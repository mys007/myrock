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




--[[     local input = torch.Tensor({
            {{4,0,1,0}, {0,0,0,0}},
            {{2,0,0,0}, {0,0,0,0}},
            {{1,0,0,1}, {1,0,1,0}},
        })

print(input)
local o = input:clone()
o = remaxnorm(o, 2, 3)
print(o)--]]

 local input = torch.Tensor(
        {{4,0,1,0}, {1,0,0,0}})

print(input)
remaxnorm(input, 2, 3)
print(input)





local opt = {}
opt.scalespaceStep = "exp"
opt.seed = 1
opt.network='/home/simonovm/workspace/pyrann/foo-scalespace/20150223-165518-full-fan3-fuseScalespp-Csqrt/cifar.net'; opt.scalespaceStep = "sqrt" opt.pyraFactor=0.7; opt.scalespaceNumSc=4; opt.pyraFanIn=3; opt.pyraFuseMode = 'scalespp';


torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)
cutorch.manualSeed(opt.seed)

local plotpath = opt.network:match("(.*/)") .. 'visu/'
os.execute('mkdir -p "' .. plotpath .. '"');

-- compute pyramid scales
--[[if (opt.pyraFactor > 0 and opt.scalespaceStep == "exp") then
    opt.pyraScales = {1}
    for i=2,opt.scalespaceNumSc do
        table.insert(opt.pyraScales, opt.pyraScales[#opt.pyraScales]*opt.pyraFactor)
    end     
end--]]

-- load model
local model = torch.load(opt.network)     
model = model:float()
local parameters,gradParameters = model:getParameters()
gradParameters:zero()

-- load data
local dataset = torch.Cifar10({nSampleRatio=25000, normalizationMode=opt.normalizationMode})




local resolution, rdiff = 'def', 0
--dataset.trainData:randRescaleCropPad(opt, 1/opt.pyraFactor, 1/opt.pyraFactor); resolution, rdiff = 'crop', -1
--dataset.trainData:randRescaleCropPad(opt, opt.pyraFactor, opt.pyraFactor); resolution, rdiff = 'pad', -1
--dataset.trainData:rescale(1/opt.pyraFactor); resolution, rdiff = 'up', -1
--dataset.trainData:rescale(opt.pyraFactor); resolution, rdiff = 'down', 1

--dataset.trainData:rescale(1/opt.pyraFactor); dataset.trainData:rescale(opt.pyraFactor); resolution, rdiff = 'updown', 0
--dataset.trainData:rescale(opt.pyraFactor); dataset.trainData:rescale(1/opt.pyraFactor); resolution, rdiff = 'downup', 0


----
local win = nil
--[[function plotPyraConvActivations(model, modulenr, nFeats, legend)
    local pyra = model:get(modulenr):get(2).output --:get(scale):get(1).modules
    for p=1,#pyra do
        image.display{image=pyra[p]:narrow(1,1,nFeats), zoom=6, min=-1, max=1, win=win, legend=legend, padding=1, nrow=1}
        image.save(plotpath..'feat' .. modulenr .. '_' .. resolution .. p .. '.png', dispAndZoom(pyra[p]:narrow(1,1,nFeats), 10))
        image.save(plotpath..'Xfeat' .. modulenr .. '_' .. (p+rdiff) .. resolution .. '.png', dispAndZoom(pyra[p]:narrow(1,1,nFeats), 10))
    end
end 
----
--
function plotMaxScaleBlock(model, modulenr, nFeats, opt)
    local pyra = model:get(modulenr):get(2):get(1):get(1)
    for p=1,opt.scalespaceNumSc do
        image.display{image=pyra:get(p).output:select(1,1):narrow(1,1,nFeats), zoom=6, min=-1, max=1, win=win, legend=p, padding=1, nrow=1}
        image.save(plotpath..'feat' .. 'Finc' .. '_' .. resolution .. p .. '.png', dispAndZoom(pyra:get(p).output:select(1,1):narrow(1,1,nFeats), 10))
        image.save(plotpath..'Xfeat' .. 'Finc' .. '_' .. (p+rdiff) .. resolution .. '.png', dispAndZoom(pyra:get(p).output:select(1,1):narrow(1,1,nFeats), 10))
    end
    image.display{image=model:get(modulenr+1).output:narrow(1,1,nFeats), zoom=6, min=-1, max=1, win=win, legend=0, padding=1, nrow=1}
    image.save(plotpath..'feat' .. 'Finc' .. '_' .. resolution ..  '.png', dispAndZoom(model:get(modulenr+1).output:narrow(1,1,nFeats), 10))
    image.save(plotpath..'Xfeat' .. 'Finc' .. '_' .. resolution .. '.png', dispAndZoom(model:get(modulenr+1).output:narrow(1,1,nFeats), 10))
end 

function plotSpatialMaxPooling(model, modulenr, nFeats)
    local pyra = model:get(modulenr)
    for p=1,opt.scalespaceNumSc do
        image.display{image=pyra:get(p).output:narrow(1,1,nFeats), zoom=6, min=-1, max=1, win=win, legend=p, padding=1, nrow=1}
        --image.save(plotpath..'feat' .. 'Finc' .. '_' .. resolution .. p .. '.png', dispAndZoom(pyra:get(p).output:select(1,1):narrow(1,1,nFeats), 10))
        --image.save(plotpath..'Xfeat' .. 'Finc' .. '_' .. (p+rdiff) .. resolution .. '.png', dispAndZoom(pyra:get(p).output:select(1,1):narrow(1,1,nFeats), 10))
    end

end--]]

function plotScalespaceConvActivations(model, modulenr, scalenr, nFeats)
    local scconv = model:get(modulenr):get(scalenr):get(1)
    for p=1,#scconv.modules do
        local im = scconv:get(p):get(2).output
        im = im:narrow(1,1,math.min(im:size(1),nFeats))
        print(torch.norm(im), scalenr, p)
        --image.display{image=im, zoom=6, min=-1, max=1, win=win, legend=modulenr..'-'..p, padding=1, nrow=1}
        image.save(plotpath..'cvact' .. modulenr .. '_' .. scalenr .. '_' .. resolution .. p .. '.png', dispAndZoom(im, 10))
        image.save(plotpath..'Xcvact' .. modulenr .. '_' .. scalenr .. '_' .. (p+rdiff) .. resolution .. '.png', dispAndZoom(im, 10))
    end
end

function plotScalespaceConvFilters(model, modulenr, scalenr, nFeats)
    local scconv = model:get(modulenr):get(scalenr):get(1)
    for p=1,#scconv.modules do
        plotSpatialConvolutionMM(scconv:get(p):get(2), win, 'stage ' .. modulenr .. ': weights ' .. p)
    end
end

function plotScalespaceTensorActivations(model, modulenr, nFeats, opt)
    local ss = model:get(modulenr).output
    local step = ss:size(1)/opt.scalespaceNumSc
    for p=0,opt.scalespaceNumSc-1 do
        local im = ss:narrow(1,p*step+1,math.min(step,nFeats))
        --image.display{image=im, zoom=6, min=-1, max=1, win=win, legend=modulenr..'-'..p, padding=1, nrow=1}
        image.save(plotpath..'act' .. modulenr .. '_' .. resolution .. p .. '.png', dispAndZoom(im, 10))
        image.save(plotpath..'Xact' .. modulenr .. '_' .. (p+rdiff) .. resolution .. '.png', dispAndZoom(im, 10))
    end
end


----

function dispAndZoom(src, zoom)
    local img = image.toDisplayTensor{input=src, min=-1, max=1}
    return image.scale(img, img:size(img:dim())*zoom, img:size(img:dim()-1)*zoom, 'simple')
end
----




local input = dataset.trainData:at(1)

model:forward(input)

local nFeats = 10

--[[


for p=1,#input do
    image.display{image=input[p], zoom=6, min=-1, max=1, win=win, legend='input', padding=1, nrow=1}
    image.save(plotpath..'img_' .. resolution .. p .. '.png', dispAndZoom(input[p], nFeats))
    image.save(plotpath..'Ximg_' .. (p+rdiff) .. resolution .. '.png', dispAndZoom(input[p], nFeats))
end

plotPyraConvActivations(model, 1, 5, 'layer activations')
plotPyraConvActivations(model, 4, 5, 'layer activations')

image.save(plotpath..'Xlastc_' .. (1+rdiff) .. resolution .. '.png', dispAndZoom(model:get(9).output, nFeats))--]]

for i=1,7 do
    plotScalespaceTensorActivations(model, i, nFeats, opt)
end

plotScalespaceConvFilters(model, 2, 2, nFeats)
--[[plotScalespaceConvActivations(model, 2, 1, nFeats)
plotScalespaceConvActivations(model, 2, 2, nFeats)
plotScalespaceConvActivations(model, 2, 3, nFeats)
plotScalespaceConvActivations(model, 2, 4, nFeats)--]]

--plotSpatialMaxPooling(model, 6, nFeats)

