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
        if (opt.caffeBiases and module.weight ~= nil and module.bias ~= nil) then
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
            if opt.weightDecay > 0 and (not opt.weightDecayMaxEpoch or model.epoch <= opt.weightDecayMaxEpoch) then
                if (module.decayFactorW ~= 0) then module.gradWeight:add(opt.weightDecay*(module.decayFactorW or 1), module.weight) end
                if (module.decayFactorB ~= 0 and module.gradBias) then module.gradBias:add(opt.weightDecay*(module.decayFactorB or 1), module.bias) end
                --TODO: should also adjust err: err = crit + 0.5*x:norm^2.  But I don't need it, so no need to waste gpu time
            end
        end   
    end
end        


--[[ based on optim.sqd

ARGS:

- `opfunc` : a function that takes a single input (X), the point
             of a evaluation, and returns f(X) and df/dX
- `x`      : the initial point
- `config` : a table with configuration parameters for the optimizer
- `config.learningRate`      : learning rate
- `config.learningRateDecay` : learning rate decay
- `config.weightDecay`       : weight decay
- `config.weightDecays`      : vector of individual weight decays
- `config.momentum`          : momentum
- `config.dampening`         : dampening for momentum
- `config.nesterov`          : enables Nesterov momentum
- `state`  : a table describing the state of the optimizer; after each
             call the state is modified
- `state.learningRates`      : vector of individual learning rates

RETURN:
- `x`     : the new x vector
- `f(x)`  : the function, evaluated before the update

(Clement Farabet, 2012)
]]
local function optimSgd(opfunc, x, config, state)
   -- (0) get/update state
   local config = config or {}
   local state = state or config
   local lr = config.learningRate or 1e-3
   local lrd = config.learningRateDecay or 0
   local wd = config.weightDecay or 0
   local mom = config.momentum or 0
   local damp = config.dampening or mom
   local nesterov = config.nesterov or false
   local lrs = config.learningRates
   local wds = config.weightDecays
   state.evalCounter = state.evalCounter or 0
   local nevals = state.evalCounter
   assert(not nesterov or (mom > 0 and damp == 0), "Nesterov momentum requires a momentum and zero dampening")

   assert(wd==0 and not wds and not lrs) --implemeted in doOptStep

   -- evaluate f(x) and df/dx
   local fx,dfdx = opfunc(x)
   
   -- learning rate decay (annealing)
   local clr = lr / (1 + nevals*lrd)
   
   if config.caffeFormula then
       -- ** Krizhevsky/Caffe style
       -- ? Btw, the effective learning rate is then ~ clr * 1/(1-momentum)  [assuming constant dfdx, rewrite the update recursion to clr*dfdx*sum_i{0,inf}(momentum^i)]
       if mom ~= 0 then
          --if not state.dfdx then          --- OLD BUT HORRIBLY WRONG BEHAVIOR
          --   state.dfdx = torch.Tensor():typeAs(dfdx):resizeAs(dfdx):copy(dfdx)
          --else
          --   state.dfdx:mul(mom):add(-clr, dfdx)
          --end       
          state.dfdx = state.dfdx or torch.Tensor():typeAs(dfdx):resizeAs(dfdx):fill(0)
          state.dfdx:mul(mom):add(-clr, dfdx)
          if nesterov then
             assert(false,'not verified how to do it yet')
             dfdx:add(mom, state.dfdx)
          else
             dfdx = state.dfdx
          end
          x:add(dfdx)
          clr = -1
       else
          x:add(-clr, dfdx)
       end   
    
    else
       -- ** Torch original
       if mom ~= 0 then
          if not state.dfdx or state.dfdx:dim()==0 then
             state.dfdx = (state.dfdx or torch.Tensor()):typeAs(dfdx):resizeAs(dfdx):copy(dfdx)
          else
             state.dfdx:mul(mom):add(1-damp, dfdx)
          end
          if nesterov then
             dfdx:add(mom, state.dfdx)
          else
             dfdx = state.dfdx
          end
       end

       x:add(-clr, dfdx)
    end   
     
   -- update evaluation counter
   state.evalCounter = state.evalCounter + 1

   -- return x*, f(x) before optimization, delta
   return x,{fx},{-clr, dfdx}
end


local function decodeCustomSched(str, epoch)
    for token in string.gmatch(str, "[^_]+") do
        if token~='sched' then
            local args = {} --last_epochxlr  (in increasing epoch order)
            for a in string.gmatch(string.trim(token), "[^x]+") do
                table.insert(args, tonumber(a))
            end
            if epoch <= args[1] then return args[2] end
        end
    end
    assert(false, 'undefined schedule string for epoch '..epoch)
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
        config.caffeFormula = (opt.optimization == 'SGDCaffe')

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
        elseif string.starts(opt.lrdPolicy,'sched') then --predefined learning schedule in format sched_20x1e-2_35x1e-3  
            local prevlr = config.learningRate
            config.learningRate = decodeCustomSched(opt.lrdPolicy, model.epoch)
            if opt.optimization == 'SGD' and prevlr ~= config.learningRate then config.dfdx = config.dfdx and config.dfdx:resize(0) or nil end --zero momentum vector on lr step (FB does it in their imagenet code)
        else
            assert(false, 'unknown lrdPolicy')    
        end

        local _,loss,step = optimSgd(feval, parameters, config) 
            
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
    ----------------------------------------------------------------------
    --clones a model in order to always save a clean copy
    function cleanModelInit(model, opt)
    	--legacy
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

		model:clearState()
        torch.save(filename, model)
        torch.save(filename..'.optconfig', config)
        --torch.save(filename..model.epoch, model)
        --torch.save(filename..model.epoch..'.optconfig', config)   
        
        if paths.filep(filename .. '.old') then
            os.execute('rm ' .. filename .. '.old')
        end
    end
end

