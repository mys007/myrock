require 'torch'
require 'optim'
require 'myutils'
require 'strict'

----------------------------------------------------------------------
-- 1) applies caffeBiases, 2) initializes custom weights
function prepareModel(model, opt)
    -- Weight initialization
    if (opt.winit ~= 'default') then
        local rngState = torch.getRNGState()    --don't mess up repeatability
        torch.manualSeed(opt.seed)
        
        if opt.winit == 'He' then resetHe2015(model, opt)
        elseif opt.winit == 'Gauss' then resetGaussConst(model)
        elseif opt.winit == 'Nin' then resetNin(model) --TODO: legacy
        else error('unknown winit') end
        
        torch.setRNGState(rngState)
    end
    
    -- Set individual factors
    for i,module in ipairs(model:listModules()) do
        if (opt.caffeBiases and module.weight ~= nil) then
            if module.lrFactorB == nil then module.lrFactorB = 2*(module.lrFactorW or 1) end
            if module.decayFactorB == nil then module.decayFactorB = 0 end
        end
    end
end

----------------------------------------------------------------------
-- Sets up weight sharing. (needs to be called after mModel:type() !)
function moduleSharing(model, opt)
    if opt.parSharing then
        for i,module in ipairs(model:listModules()) do
            if (module.pendingSharing ~= nil) then
                module:pendingSharing()      --Module:share() needs to be called after Model:type()
            end
        end
    end
end

----------------------------------------------------------------------
-- preprocesses per-module weight and bias to incorporate: 1) per-module learning rate, 2) per-module weight decay
function prepareGradPerModule(model, opt)
    for i,module in ipairs(model:listModules()) do
        if module.weight then
            -- personalized learning rate (faster solution: edit scale in Module:backward ... but we lose weight/bias distinction) and weight decay
            if (not opt.parSharing or module.pendingSharing == nil) then
                if (module.lrFactorW ~= nil and module.lrFactorW ~= 1) then module.gradWeight:mul(module.lrFactorW) end
                if (module.lrFactorB ~= nil and module.lrFactorB ~= 1) then module.gradBias:mul(module.lrFactorB) end
            end          

            --NOTE: this is different than in optim.sgd, they ignore shared weights
            if opt.weightDecay > 0 then
                if (module.decayFactorW ~= 0) then module.gradWeight:add(opt.weightDecay*(module.decayFactorW or 1), module.weight) end
                if (module.decayFactorB ~= 0) then module.gradBias:add(opt.weightDecay*(module.decayFactorB or 1), module.bias) end
                --TODO: should also adjust err: err = crit + 0.5*x:norm^2.  But I don't need it, so no need to waste gpu time
            end
        end   
    end
end        


----------------------------------------------------------------------
-- optimize on current mini-batch
function doOptStep(model, parameters, feval, opt, config)
    if opt.optimization == 'CG' then
        config.maxIter = opt.maxIter
        optim.cg(feval, parameters, config)

    elseif opt.optimization == 'LBFGS' then
        config.learningRate = opt.learningRate
        config.maxIter = opt.maxIter
        config.nCorrection = 10
        optim.lbfgs(feval, parameters, config)

    elseif opt.optimization == 'SGD' or opt.optimization == 'SGDCaffe' then
        config.learningRateDecay = 0
        config.weightDecay = 0 --weightDecay handled in prepareGradPerModule
        config.momentum = opt.momentum
        config.dampening = opt.momdampening>=0 and opt.momdampening or opt.momentum
        config.evalCounter = config.evalCounter or 0

        if (opt.lrdPolicy == 'fixed') then
            config.learningRate = opt.learningRate
        elseif (opt.lrdPolicy == 'inv') then --torch's default ; caffe also supports denominator^power
            config.learningRate = opt.learningRate / (1 + config.evalCounter * opt.lrdGamma)
        elseif (opt.lrdPolicy == 'step') then --as in caffe (drop the learning rate stepwise by a factor of lrdGamma every lrdStep iterations)
            config.learningRate = opt.learningRate * math.pow(opt.lrdGamma, math.floor(config.evalCounter / opt.lrdStep))
        elseif (opt.lrdPolicy == 'expep') then --mine version of exponential decay ("half lr each 10 epochs"), epoch-based not iter-based
            config.learningRate = opt.learningRate * math.pow(opt.lrdGamma, (model.epoch-1) / opt.lrdStep)
        elseif (opt.lrdPolicy == 'poly') then --as in caffe (polynomial decay, fixed number of max iterations, doesn't have such a sharp decay at the start as the others)
            config.learningRate = opt.learningRate * math.max(0, math.pow(1 - config.evalCounter / opt.lrdStep, opt.lrdGamma))
        else
            assert(false, 'unknown lrdPolicy')    
        end

        local loss,step
        if opt.optimization == 'SGD' then _,loss,step = optim.sgd(feval, parameters, config) end
        if opt.optimization == 'SGDCaffe' then _,loss,step = optim.sgdcaffe(feval, parameters, config) end
            
        return loss[1], step      

    elseif opt.optimization == 'ASGD' then
        config.eta0 = opt.learningRate --(not exactly the same formula as in sgd)
        config.learningRate = opt.learningRate
        if model.epoch < opt.asgdE0 then config.t0 = 1e20 elseif config.t0 == 1e20 then config.t0 = config.t end 
        config.lambda = 0 --weightDecay handled in prepareGradPerModule
        local _,loss,avgx = optim.asgd(feval, parameters, config)
        
        return loss[1], {1,avgx}--todo 

    else
        error('unknown optimization method')
    end
end


do
    local cleanModel, cleanParameters = nil, nil
    
    -- TODO/BUG: this is practically useless if modelCreate calls sth like model:forward(expectedInput) ... then all has been filled.
        -- .. should probably use sth like sanitize(net) from FB
    
    ----------------------------------------------------------------------
    --clones a model in order to always save a clean copy
    function cleanModelInit(model, opt)
        cleanModel = model:clone():float()
        moduleSharing(cleanModel, opt) -- share parameters (same # of params as main model)
        cleanParameters = cleanModel:getParameters()
    end

    ----------------------------------------------------------------------
    -- save/log current net (a clean model without any gradients, inputs,... ; it takes less storage)
    function cleanModelSave(model, parameters, config, opt, fname)  
        local filename = paths.concat(opt.save, fname)
        os.execute('mkdir -p ' .. paths.dirname(filename))
        if paths.filep(filename) then
            os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
        end
        print('<trainer> saving network to '..filename)
        cleanParameters:copy(parameters)
        cleanModel.epoch = model.epoch
        torch.save(filename, cleanModel)
        torch.save(filename..'.optconfig', config)
        --torch.save(filename..cleanModel.epoch, cleanModel)
        --torch.save(filename..cleanModel.epoch..'.optconfig', config)   
    end
end

