require 'torch'
require 'nn'
require 'nnx'
require 'optim'
require 'image'
require 'trepl'
require 'Cifar10'
require 'Sun397'
require 'myutils'
require 'model'
local _ = nil

----------------------------------------------------------------------
-- parse command-line options
--
local cmd = torch.CmdLine(nil,nil,true)  --my fix   --TODO: wont be accepted, replace with https://github.com/davidm/lua-pythonic-optparse/blob/master/lmod/pythonic/optparse.lua
cmd:option('-network', '', 'reload pretrained network')
cmd:option('-device', 1, 'CUDA device')
cmd:option('-threads', 1, 'nb of threads to use (blas on cpu)')
cmd:option('-normalizationMode', 'RgbZca', 'what normalization to use RgbZca | YuvScnChStat | none')
cmd:option('-alg', 'fast', 'what adversary alg use classic | fast')
cmd:option('-criterion', 'nll', 'nll | svm')
cmd:option('-dataset', 'Cifar10', 'dataset Cifar10 | Sun397')
cmd:option('-optimPrints', 100, 'print sgd info each x-th iteration (0 = deactivate)')
cmd:text()
local opt = cmd:parse(arg)

opt.seed = 1

--opt.network='/storage/msimonov/h/workspace/pyrann/foo-baseline/20150312-014903-full-base-16-256-128-jn2_5/cifar.net'; 
opt.network='/storage/msimonov/h/workspace/pyrann/foo-baseline/20150309-214136-full-base-16-256-128/cifar.net';
--opt.network='/storage/msimonov/h/workspace/pyrann/foo-baseline/20150317-110434-full-base-16-256-128-qr09last/cifar.net';

torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)
torch.setnumthreads(opt.threads)

-- cuda
local doCuda = opt.device > 0
if doCuda then
    print('Will use device '..opt.device)
    require 'cutorch'
    require 'cunn'    
    cutorch.setDevice(opt.device)
end

-- load model
local model = torch.load(opt.network)     
model = model:float()
local parameters,gradParameters = model:getParameters()
gradParameters:zero()

-- load data
local dataset
if opt.dataset == 'Cifar10' then
    dataset = torch.Cifar10({nSampleRatio=1, normalizationMode=opt.normalizationMode}) --, sampleAllSets=1})nSamples})
elseif opt.dataset == 'Sun397' then
    dataset = torch.Sun397({nSampleRatio=1, partition=1, minSize=32, fullTrainSet=true, squareCrop=true, nValidations=1000})
end

----
local win = nil
function plotPyraConvActivations(model, modulenr, nFeats, legend)
    local pyra = model:get(modulenr):get(2).output --:get(scale):get(1).modules
    for p=1,#pyra do
        image.display{image=pyra[p]:narrow(1,1,nFeats), zoom=6, min=-1, max=1, win=win, legend=legend, padding=1, nrow=1}
        image.save('/home/simonovm/p/feat' .. modulenr .. '_' .. resolution .. p .. '.png', dispAndZoom(pyra[p]:narrow(1,1,nFeats), 10))
        image.save('/home/simonovm/p/Xfeat' .. modulenr .. '_' .. (p+rdiff) .. resolution .. '.png', dispAndZoom(pyra[p]:narrow(1,1,nFeats), 10))
    end
end 
----

function dispAndZoom(src, zoom)
    local img = image.toDisplayTensor{input=src, min=-1, max=1}
    return image.scale(img, img:size(img:dim())*zoom, img:size(img:dim()-1)*zoom, 'simple')
end
----

-- finds an input producing an output highly similar to the one given (basically apprximative network inversion). The input is thus one of the many
function inputFromActivations(model, targetOutput, inputInit)
    local err = 10
    local iter = 0
    local config = {}
    local criterion = nn.MSECriterion()
    --local criterion = nn.AbsCriterion()
    local input = inputInit:clone():fill(1) --will pass all relus, hopefully
    
    if doCuda then
        model:cuda(); criterion:cuda(); input=input:cuda(); targetOutput=targetOutput:cuda()
    else    
        model:float(); criterion:float(); input=input:float(); targetOutput=targetOutput:float()
    end    
    
    while err > 0.01 do--and iter < 100000 do

    
         local feval = function(x)
                            local output = model:forward(x)                         
                            err = criterion:forward(output, targetOutput)
                            local df_do = criterion:backward(output, targetOutput)
                            local di = model:updateGradInput(x, df_do)        
                            return err, di
                      end
        
       --elseif opt.optimization == 'SGD' then
            config.learningRate = 40--opt.learningRate
            --config.weightDecay = opt.weightDecay
            config.momentum = 0.9
            --config.dampening = opt.momdampening>=0 and opt.momdampening or opt.momentum
            --if (opt.lrdStep > 0) then
              --  config.learningRate = opt.learningRate * math.pow(opt.lrdStep, -(model.epoch-1) / opt.lrdStepEpoch)
                --config.learningRateDecay = 0
            --else            
                config.learningRateDecay = 1e-3--opt.learningRateDecay
            --end    
            
            local input,loss,step = optim.sgd(feval, input, config)
    
            iter = iter + 1
        
            if iter%50==0 then print(iter, err, step[1]) end
    
        end

    model:float()
    
    return input:float()
end


function clsOf(output)
    local m, idx = torch.max(output, 1)
    return idx[1], m[1]
end

--Szegedy
function adversarial(model, opt, input, target, doPlot)

    -- loss function:
    local criterion
    if opt.criterion == "nll" then
        criterion = nn.ClassNLLCriterion()  --negative log-likelihood
    elseif opt.criterion == "svm" then
        criterion = nn.MultiMarginCriterion() --hinge loss
    else
        assert(false)
    end

    -- Dropout should be IMHO disabled ... otherwise I optimize AE for the whole ensemble of nets and any is sufficient (but it actually doesn't need to work for the 
    -- test-state network, does it? When disabled, I have just the single net. 
    local dropouts = model:findModules('nn.Dropout')
    for i = 1, #dropouts do dropouts[i]:setp(0) end

    if doCuda then
        model:cuda(); criterion:cuda(); input=input:cuda();
    else    
        model:float(); criterion:float(); input=input:float();
    end
    
    local delta = input:clone():fill(0)
    local found = false
    local C = 5

    while not found do
    
        local iter = 0
        local config = {}

        while iter < 20000 and not found do
    
             local feval = function(x)
                                local adversary = input + x

                                local output = model:forward(adversary)              
                                found = clsOf(output)==target   
                                
                                -- loss = classification loss + norm on delta                              
                                local err = criterion:forward(output, target) + C*math.pow(x:norm(2),2) --NOTE: I use L2^2, the paper uses L1, which is harder to diff, though
    
                                if opt.optimPrints>0 and iter%opt.optimPrints==0 then print(iter, found, err, criterion:forward(output, target), C*x:norm(2), x:norm(2)) end
                                local df_do = criterion:backward(output, target)
                                local di = model:updateGradInput(adversary, df_do)
                                
                                -- gradient = dClassifLoss/dx + dNorm/dx
                                if found then return err, di:zero() end --don't move anymore;)        
                                return err, di:add(x*2*C)
                          end
    
            --SGD
            if true then
                config.learningRate = 0.0001
                config.learningRateDecay = 1e-3
                optim.sgd(feval, delta, config)
            end       
            
            if false then
                -- TODO: use L-BFGS-B as in https://github.com/tabacof/adversarial/blob/master/overfeat/adversarial.lua
                -- but as the norms of deltas are much less than norm of input, the possible going out of bounds is IMHO negligible
            end     
            
            iter = iter + 1
        end

        -- TODO: line search for C  ; now I just decrease the penalization (it doesn't seem to be that sensitive to particular C because I use 'found' early termination and small lr)        
        if not found then C = C / 5 end
    end
    
    if doPlot then
        image.display{image=input,          zoom=6, min=-2, max=2, win=win, legend='Input', padding=1, nrow=1}
        image.display{image=input + delta,  zoom=6, min=-2, max=2, win=win, legend='Adversary', padding=1, nrow=1}
        image.display{image=delta,          zoom=6, min=-2, max=2, win=win, legend='Delta', padding=1, nrow=1}
        image.display{image=delta,          zoom=6, win=win, legend='Delta (scaled)', padding=1, nrow=1}
    end
    local c,m = clsOf(model:forward(input))
    print('Input classified as ' .. c ..  ' (probI ' .. math.exp(m) .. ')')   --exp(m) won't work for svm crit, I guess
    local c,m = clsOf(model:forward(input + delta))
    print('Adversary classified as ' .. c ..  ' (probA ' .. math.exp(m) .. ')')
    print('Delta norm: ' .. delta:norm(2))
    print('Input norm: ' .. input:norm(2))
    
    return delta:norm(2)
end
    
    
    
--Goodfellow
function adversarialGoodfellow(model, opt, input, targetCorrect, doPlot)

    -- loss function:
    local criterion
    if opt.criterion == "nll" then
        criterion = nn.ClassNLLCriterion()  --negative log-likelihood
    elseif opt.criterion == "svm" then
        assert(false, "todo: needs to change how probabilities are computed, now just exp()")
        criterion = nn.MultiMarginCriterion() --hinge loss
    else
        assert(false)
    end

    -- Dropout should be IMHO disabled ... otherwise I optimize AE for the whole ensemble of nets and any is sufficient (but it actually doesn't need to work for the 
    -- test-state network, does it? When disabled, I have just the single net. 
    local dropouts = model:findModules('nn.Dropout')
    for i = 1, #dropouts do dropouts[i]:setp(0) end
    
    local output = model:forward(input)              
    local err = criterion:forward(output, targetCorrect)
    local df_do = criterion:backward(output, targetCorrect)
    local di = model:updateGradInput(input, df_do)
    
    --di = di / torch.max(torch.abs(di)) --Linf normed di
    --di = di:normal(0,1); di = di / torch.max(torch.abs(di)) --LInf normed rand vector   
    di:apply(function(x) if x>0 then return 1 elseif x<0 then return -1 else return 0 end end) --sign (as in paper)
    
    --find the smallest eps iteratively (x paper)
    local eps = 1e-4
    local inputcls = clsOf(model:forward(input))
    local advcls = inputcls
    local iter = 0
    
    while advcls == inputcls do
        eps = eps * 1.01
        --if eps>1e-3 then eps = eps + 1e-3 else eps = eps * 1.01 end
        local lprob
        advcls, lprob = clsOf(model:forward(input + di*eps))
        if opt.optimPrints>0 and iter%opt.optimPrints==0 then print(iter, eps, math.exp(lprob)) end
        iter = iter + 1
    end  
    
    local delta = di*eps
    
    if doPlot then
        image.display{image=input,          zoom=6, min=-2, max=2, win=win, legend='Input', padding=1, nrow=1}
        image.display{image=input + delta,  zoom=6, min=-2, max=2, win=win, legend='Adversary', padding=1, nrow=1}
        image.display{image=delta,          zoom=6, min=-2, max=2, win=win, legend='Delta', padding=1, nrow=1}
        image.display{image=delta,          zoom=6, win=win, legend='Delta (scaled)', padding=1, nrow=1}
    end
    local c,m = clsOf(model:forward(input))
    print('Input classified as ' .. c ..  ' (probI ' .. math.exp(m) .. ')')      --exp(m) won't work for svm crit, I guess
    local c,m = clsOf(model:forward(input + delta))
    print('Adversary classified as ' .. c ..  ' (probA ' .. math.exp(m) .. ')')
    print('Delta norm: ' .. delta:norm(2) .. ', eps: '.. eps)
    print('Input norm: ' .. input:norm(2))
    
    return eps
end    
    
    
    
-- compares two saved logs of ae computations and prints the precentage of entries which are larger in the first file than in the second 
function rankAEOutputs(file1, file2, str)   
    local function readLogVals(fname) 
        local f = assert(io.open(fname, 'r'))
        local fstr = f:read('*all')
        f:close()
    
        local v1 = {}
        for s in string.gmatch(fstr, str.."([0-9.]+)") do
            table.insert(v1, tonumber(s))
        end
        return torch.Tensor(v1)
    end      
    
    local v1 = readLogVals(file1)
    local v2 = readLogVals(file2)
    local percV1Greater = torch.sum(torch.gt(v1,v2)) / v1:nElement()
    print('Input1 > Input2 in '..(percV1Greater*100)..'% cases')   
end    
   
    
--rankAEOutputs('/storage/msimonov/h/workspace/pyrann/foo-baseline/20150326-160302-full-32n16n8n4n2-f5-p-256-qr09last/advF.txt', '/home/simonovm/workspace/pyrann/foo-baseline/20150227-115459-full-32n16n8n4n2-f5-p-256/advF.txt', 'eps: ')
--rankAEOutputs('/storage/msimonov/h/workspace/pyrann/foo-baseline/20150326-160302-full-32n16n8n4n2-f5-p-256-qr09last/adv.txt', '/home/simonovm/workspace/pyrann/foo-baseline/20150227-115459-full-32n16n8n4n2-f5-p-256/adv.txt', 'Delta norm: ')
----    
    
    


local resolution, rdiff = 'defMy', 0
--dataset.trainData:rescale(1/opt.pyraFactor); resolution, rdiff = 'up', -1
--dataset.trainData:rescale(opt.pyraFactor); resolution, rdiff = 'down', 1

--dataset.trainData:rescale(1/opt.pyraFactor); dataset.trainData:rescale(opt.pyraFactor); resolution, rdiff = 'updown', 0
--dataset.trainData:rescale(opt.pyraFactor); dataset.trainData:rescale(1/opt.pyraFactor); resolution, rdiff = 'downup', 0

local input, target = dataset.trainData:at(1)


--Adversary examples
if true then

    if true then
        local avgDeltaNorm = 0 
        local minDelta = 1e10
        local nSamples = 50
        local nTests = 0
        for i=1,nSamples do
            print('~~ Image '..i)
            local d = 0
            local input, target = dataset.trainData:at(i)
            if target==clsOf(model:forward(input:typeAs(model.output))) then              
                if opt.alg=='fast' then
                    d = adversarialGoodfellow(model, opt, input, target, false)
                else
                    local newTarget = target==1 and 2 or 1
                    d = adversarial(model, opt, input, newTarget, false)
                end                
                avgDeltaNorm = avgDeltaNorm + d
                nTests = nTests + 1
                minDelta = math.min(minDelta, d)
            else
                print('  skipping, img misclassified, should be '..target)
            end
        end
        avgDeltaNorm = avgDeltaNorm / nTests
        print('Network:' .. opt.network)
        print('Average norm of adversaries:' .. avgDeltaNorm)
        print('Min norm of adversaries:' .. minDelta)
    
    else
        local newTarget = 1
        print('Original target: ' .. target .. ' (' .. dataset:classes()[target] .. ')')       
        if opt.alg=='fast' then
            adversarialGoodfellow(model, opt, input, target, true)
        else    
            print('New target: ' .. newTarget .. ' (' .. dataset:classes()[newTarget] .. ')')
            adversarial(model, opt, input, newTarget, true)
        end    
    end
    
--Input reconstruction
else 

    -- trim the model
    local cutoff = 7
    for i = #model.modules, cutoff, -1 do  model.modules[i] = nil end
    model.output = model.modules[#model.modules].output
    
    --local jittermod = nn.JitteringModuleGNoise(5)
    --local jittermod = nn.JitteringModuleScale(1/1.1, 1.1, false); torch.manualSeed(2)
    local jittermod = nn.JitteringModuleTranslate(1.5)
    --local jittermod = nn.Dropout(0.5)
    
    
    local output = model:forward(input):clone()
    local backo = model:backward(input, output):clone()
    
    jittermod:training()
    local outputJ = jittermod:forward(output)
    --print(jittermod.dx, jittermod.dy)
    --local t = output:clone(); t[{{},{1,2},{1,3}}]:fill(0)
    --local backoJ = model:backward(input, jittermod:backward(output, outputJ))
    local backoJ = model:backward(input, outputJ)
    
    local inputJ = inputFromActivations(model, outputJ, input)
    
    local nFeats = 10
    image.display{image=input, zoom=6, min=0, max=1, win=win, legend='InOrig', padding=1, nrow=1}
    image.display{image=backo, zoom=6, min=-2, max=2, win=win, legend='BaOrig', padding=1, nrow=1}
    image.display{image=backoJ, zoom=6, min=-2, max=2, win=win, legend='BaJitter', padding=1, nrow=1}
    image.display{image=inputJ, zoom=6, min=-2, max=2, win=win, legend='InJitter', padding=1, nrow=1}
    image.display{image=torch.abs(input-inputJ), zoom=6, min=-2, max=2, win=win, legend='InDiff', padding=1, nrow=1, saturate=false}
    --image.display{image=backo-backoJ, zoom=6, min=-2, max=2, win=win, legend='Diff', padding=1, nrow=1, saturate=false}
end    
