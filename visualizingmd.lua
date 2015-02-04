require 'torch'
require 'nn'
require 'nnx'
require 'optim'
require 'image'
require 'cutorch'
require 'cunn'
require 'trepl'
require 'Cifar10'
require 'myutils'
require 'model'
local _ = nil

local opt = {}
opt.scalespaceStep = "exp"
opt.seed = 1
opt.network='/home/simonovm/workspace/pyrann/foo-baseline/20150303-024624-full-experLayer-F0/cifar.net'; opt.normalizationMode = 'YuvScnChStat'
--opt.network='/home/simonovm/workspace/pyrann/foo-scalespace/20150223-165224-full-fan3-fuseScalespp-Cexp/cifar.net'; opt.modelName = 'scalespace'; opt.numEpochs = 0; opt.insertAMP = 8; opt.subsampleData = 1;

--opt.network='/home/simonovm/workspace/pyrann/foo-scalespace/20150226-021159-full-fan3-fuseScalespp-Cexp-bigFilter/cifar.net'; opt.modelName = 'scalespace';
--opt.network='/home/simonovm/workspace/pyrann/foo-pyra/20150226-021206-full-amp-fan3-fuseMaxscale-bigFilter/cifar.net'; opt.modelName = 'pyra';

torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)
cutorch.manualSeed(opt.seed)

-- load model
local model = torch.load(opt.network)     
model = model:float()
local parameters,gradParameters = model:getParameters()
gradParameters:zero()

-- load data
local dataset = torch.Cifar10({nSampleRatio=25000, normalizationMode=opt.normalizationMode})







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



local resolution, rdiff = 'defMy', 0
--dataset.trainData:rescale(1/opt.pyraFactor); resolution, rdiff = 'up', -1
--dataset.trainData:rescale(opt.pyraFactor); resolution, rdiff = 'down', 1

--dataset.trainData:rescale(1/opt.pyraFactor); dataset.trainData:rescale(opt.pyraFactor); resolution, rdiff = 'updown', 0
--dataset.trainData:rescale(opt.pyraFactor); dataset.trainData:rescale(1/opt.pyraFactor); resolution, rdiff = 'downup', 0

local input = dataset.trainData:at(1)

model:forward(input)


local nFeats = 10
image.display{image=torch.abs(model:get(4):get(1).output:narrow(1,1,nFeats)), zoom=6, min=0, max=1, win=win, legend='A', padding=1, nrow=1}
image.display{image=torch.abs(model:get(4):get(2).output:narrow(1,1,nFeats)), zoom=6, min=0, max=1, win=win, legend='B', padding=1, nrow=1}
image.display{image=torch.abs(model:get(5).output:narrow(1,1,nFeats)), zoom=6, min=-1, max=1, win=win, legend='A*B', padding=1, nrow=1}


for i=1,#model.modules do
    if (torch.isTensor(model.modules[i].output)) then
        print(i .. ' ' .. torch.norm(model.modules[i].output,2))
    end
end

--local v = torch.cmul(model:get(4):get(1).output, model:get(4):get(2).output)
--image.display{image=v:narrow(1,1,nFeats), zoom=6, min=-10, max=10, win=win, legend='V', padding=1, nrow=1}









        
    --if string.starts(opt.modelName,'pyra') then dataset:toPyra(opt.pyraScales) end
    --if string.starts(opt.modelName,'scalespace') then dataset:toScalespaceTensor(opt.pyraScales) end
    
    --dataset.trainData:rescale(1/opt.pyraFactor)
    --dataset.validData:rescale(1/opt.pyraFactor)
    --dataset.testData:rescale(1/opt.pyraFactor)

    
    --[[
        -- reset gradients
                    gradParameters:zero()
        
                    -- f is the average of all criterions
                    local f = 0
        
                    -- evaluate function for complete mini batch
                    for i = 1,#inputs do
                        -- estimate f
                        local output = model:forward(inputs[i])
                        local err = criterion:forward(output, targets[i])
                        f = f + err
        
                        -- estimate df/dW
                        local df_do = criterion:backward(output, targets[i])
                        model:backward(inputs[i], df_do)
        
        --]]