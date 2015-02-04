require 'nn'
require 'optim'
require 'strict'

--------------------------------------- LearningDebugger ---------------------------------------
-- print average values of each nonshared weight matrix and each weight matrix increment [inspired by cuda-convnet]

local LearningDebugger = torch.class('LearningDebugger')


function LearningDebugger:__init(config)    
    self.gradParams, self.gradParamsPrev = {}, {}
    self.params, self.paramsPrev = {}, {}
    self.gradWeightRatio, self.gradWeightRatioPrev = {}, {}
    self.gradParamsB, self.gradParamsBPrev = {}, {}
    self.paramsB, self.paramsBPrev = {}, {}
    self.gradWeightRatioB, self.gradWeightRatioBPrev = {}, {}       
    self.outputs, self.outputsPrev = {}, {}
    self.gradInputs, self.gradInputsPrev = {}, {}
end


local function updMeanStd(inp, subject, norm)
    if not inp then inp = {0, 0, 0, 0} end
    return  { inp[1] + torch.mean(subject) / norm, inp[2] + torch.std(subject) / norm, inp[3] + torch.max(subject) / norm, inp[4] + torch.min(subject) / norm }
end

function LearningDebugger:visit(model, nEvalsPerEpoch)
    assert(model ~= nil and nEvalsPerEpoch ~= nil)
    
    for i,module in ipairs(model:listModules()) do
        if (module.weight ~= nil and module.weight:nElement()>0 and module.pendingSharing==nil) then
            self.gradParams[i] = updMeanStd(self.gradParams[i], module.gradWeight, nEvalsPerEpoch)
            self.params[i] = updMeanStd(self.params[i], module.weight, nEvalsPerEpoch)
            --self.gradWeightRatio[i] = updMeanStd(self.gradWeightRatio[i], torch.cdiv(module.gradWeight,module.weight), nEvalsPerEpoch)
        end    
        if (module.bias ~= nil and module.bias:nElement()>0 and module.pendingSharing==nil) then
            self.gradParamsB[i] = updMeanStd(self.gradParamsB[i], module.gradBias, nEvalsPerEpoch)
            self.paramsB[i] = updMeanStd(self.paramsB[i], module.bias, nEvalsPerEpoch)
            --self.gradWeightRatioB[i] = updMeanStd(self.gradWeightRatioB[i], torch.cdiv(module.gradBias,module.bias), nEvalsPerEpoch)
        end   
        if (torch.isTensor(module.output)) then  
            self.outputs[i] = updMeanStd(self.outputs[i], module.output, nEvalsPerEpoch)
        end
        if (torch.isTensor(module.gradInput)) then
            self.gradInputs[i] = updMeanStd(self.gradInputs[i], module.gradInput, nEvalsPerEpoch)
        end
     end
end


local function printStat(stat, statPrev)
    for i=1,1000 do
        if stat[i] then
            print(string.format("\tModule %s:\t%s\t+- %s\t[diff\t%s\t+- %s], max %s min %s", i, stat[i][1], stat[i][2], 
                                stat[i][1] - (statPrev[i] or {0,0})[1], stat[i][2] - (statPrev[i] or {0,0})[2], stat[i][3], stat[i][4]))
        end    
    end    
end

local function printStatRatio(stat, statb)
    for i=1,1000 do
        if stat[i] then
            print(string.format("\tModule %s:\t%s\t+- %s\tmax %s\tmin %s", i, stat[i][1]/statb[i][1], stat[i][2]/statb[i][2], 
                                stat[i][3]/statb[i][3], stat[i][4]/statb[i][4]))
        end    
    end    
end



function LearningDebugger:reset()
    print('Module weight (Mean +- std, avg over batches):')
    printStat(self.params, self.paramsPrev)
    print('Module gradWeight (Mean +- std, avg over batches):')
    printStat(self.gradParams, self.gradParamsPrev)
    print('Module gradWeightRatio (Mean +- std, avg over batches):')
    printStatRatio(self.gradParams, self.params)
    --printStat(self.gradWeightRatio, self.gradWeightRatioPrev)    
    print('Module bias (Mean +- std, avg over batches):')
    printStat(self.paramsB, self.paramsBPrev)
    print('Module gradBias (Mean +- std, avg over batches):')
    printStat(self.gradParamsB, self.gradParamsBPrev)    
    print('Module gradBiasRatio (Mean +- std, avg over batches):')
    --printStat(self.gradWeightRatioB, self.gradWeightRatioBPrev)
    printStatRatio(self.gradParamsB, self.paramsB)        
    print('Module output (Mean +- std, avg over batches):')
    printStat(self.outputs, self.outputsPrev)
    print('Module gradInput (Mean +- std, avg over batches):')
    printStat(self.gradInputs, self.gradInputsPrev)

    self.gradParamsPrev = self.gradParams; self.gradParams = {}
    self.paramsPrev = self.params; self.params = {}
    self.gradWeightRatioPrev = self.gradWeightRatio; self.gradWeightRatio = {}
    self.gradParamsBPrev = self.gradParamsB; self.gradParamsB = {}
    self.paramsBPrev = self.paramsB; self.paramsB = {}
    self.gradWeightRatioBPrev = self.gradWeightRatioB; self.gradWeightRatioB = {}    
    self.outputsPrev = self.outputs; self.outputs = {}
    self.gradInputsPrev = self.gradInputs; self.gradInputs = {}
end




--------------------------------------- LearningLogger ---------------------------------------
-- logs and plots various info about the learning process

local LearningLogger = torch.class('LearningLogger')


function LearningLogger:__init(basepath, showPlots)
    assert(basepath~=nil and showPlots~=nil)    
    
    --TODO: if files exist, load self and resume
    
    self.accuracyL = optim.Logger(paths.concat(basepath, 'accuracy.log'))
    self.accuracyL:setNames{'% mean class accuracy (train)', '% mean class accuracy (valid)', '% mean class accuracy (test)'}
    self.accuracyL:style{'-','-','-'}
    self.accuracyL.showPlot = showPlots
    
    self.lossL = optim.Logger(paths.concat(basepath, 'loss.log'))
    self.lossL:setNames{'batch loss (train)'}
    self.lossL:style{'-'}
    self.lossL.showPlot = showPlots
    
    self.weightsL = optim.Logger(paths.concat(basepath, 'weights.log'))
    self.weightsL:setNames{'L2 norm', 'mean', 'max', 'min'}
    self.weightsL:style{'-','-','-','-'}
    self.weightsL.showPlot = showPlots
    
    self.gradUpdL = optim.Logger(paths.concat(basepath, 'gradUpd.log'))
    self.gradUpdL:setNames{'L2 norm', 'mean', 'max', 'min', 'learningRate'}
    self.gradUpdL:style{'-','-','-','-','-'}
    self.gradUpdL.showPlot = showPlots
       
    self.gradWeightRatioL = optim.Logger(paths.concat(basepath, 'gradWeightRatio.log'))
    self.gradWeightRatioL:setNames{'L2 norm ratio', 'mean ratio', 'max ratio', 'min ratio'}
    self.gradWeightRatioL:style{'-','-','-','-'}
    self.gradWeightRatioL.showPlot = showPlots
    
    self.iterCtr = 0
    self.epochCtr = 0
    self.ticksStr = 'set xtics ('
end


function LearningLogger:logAccuracy(trainA, validA, testA)
    assert(trainA ~= nil and validA ~= nil and testA ~= nil)
    self.accuracyL:add({trainA, validA, testA})
end

function LearningLogger:logLoss(trainL)
    assert(trainL ~= nil)
    self.lossL:add({trainL})
end

function LearningLogger:logWeights(parameters)
    assert(parameters~=nil)
    assert(self.lastWeights==nil, 'logGradUpd should follow logWeights')
    self.lastWeights = {torch.norm(parameters), torch.mean(parameters), torch.max(parameters), torch.min(parameters)}
    self.weightsL:add(self.lastWeights)
end

function LearningLogger:logGradUpd(sgdStep, learningRate)
    assert(sgdStep~=nil)
    assert(self.lastWeights~=nil, 'logGradUpd should follow logWeights')
  
    local l2norm = math.abs(sgdStep[1]) * torch.norm(sgdStep[2])
    local mean = sgdStep[1] * torch.mean(sgdStep[2])
    local ex1 = sgdStep[1] * torch.max(sgdStep[2])
    local ex2 = sgdStep[1] * torch.min(sgdStep[2])
    self.gradUpdL:add({l2norm, mean, math.max(ex1,ex2), math.min(ex1,ex2), learningRate})
    
    self.gradWeightRatioL:add({l2norm / self.lastWeights[1], mean / self.lastWeights[2], math.max(ex1,ex2) / self.lastWeights[3], math.min(ex1,ex2) / self.lastWeights[4]})
    self.lastWeights = nil
    self.iterCtr = self.iterCtr + 1
end

function LearningLogger:plot()

    --plot ticks corresponding to epochs, not iterations    
    self.epochCtr = self.epochCtr + 1
    self.ticksStr = self.ticksStr .. (self.epochCtr>1 and ',' or '') .. ' "' .. self.epochCtr .. '" ' .. self.iterCtr
    self.lossL.plotRawCmd = self.ticksStr .. ');'
    self.weightsL.plotRawCmd = self.ticksStr .. ');'
    self.gradUpdL.plotRawCmd = self.ticksStr .. ');'
    self.gradWeightRatioL.plotRawCmd = self.ticksStr .. ');'

    self.accuracyL:plot()
    self.lossL:plot()
    self.weightsL:plot()
    self.gradUpdL:plot()
    self.gradWeightRatioL:plot()
end



