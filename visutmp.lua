require 'torch'
require 'nn'
require 'nnx'
require 'optim'
require 'image'
require 'cutorch'
require 'cunn'
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
cmd:option('-layer', 0, 'layer to visualize max activations in layer x (0 = top)')
cmd:option('-plotname', '', 'plot name (subdir)')
cmd:option('-normalizationMode', 'RgbZca', 'what normalization to use RgbZca | YuvScnChStat | none')
cmd:option('-dataset', 'Cifar10', 'dataset Cifar10 | Sun397')
cmd:option('-sunMinSize', 32, 'Sun397: the shorter side of images')
cmd:text()
local opt = cmd:parse(arg)

opt.scalespaceStep = "exp"
opt.seed = 1
--opt.network='/home/simonovm/workspace/pyrann/foo-baseline/20150408-131900-full-base-16-256-128-overfeatall/cifar.net'; opt.normalizationMode = 'RgbZca'
--opt.network='/home/simonovm/workspace/pyrann/foo-baseline/20150422-122625-full-down5-f5-p2-fm64to384-m09-b12-lr5e2-svm-wd-overfeat-trainrnd/cifar.net'
--opt.network='/home/simonovm/workspace/pyrann/foo-inversepyra/20150402-150928-full-down5-f5-p2-fm64to384-m09-b12-lr5e2-svm-wd-maxscale/cifar.net'; opt.normalizationMode = 'RgbZca'
--opt.network='/home/simonovm/workspace/pyrann/foo-baseline/20150402-150919-full-down5-f5-p2-fm64to384-m09-b12-lr5e2-svm-wd/cifar.net'
--opt.network='/home/simonovm/workspace/pyrann/foo-baseline/20150331-144634-sunf-32n16n8x3n2-f3-m09-b12-lr5e3-nll-wd-he/cifar.net'
--opt.network='/home/simonovm/workspace/pyrann/foo-baseline/20150414-015328-sunf-32n16n8x3n2-f3-m09-b12-lr5e3-nll-wd-he-overfeat-od-dook/cifar.net'


torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)
cutorch.manualSeed(opt.seed)


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
    dataset = torch.Sun397({nSampleRatio=1, partition=1, minSize=opt.sunMinSize, fullTrainSet=true, squareCrop=true, nValidations=1000})
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

function dispAndZoom(src, zoom, min, max)
    local img = image.toDisplayTensor{input=src, min=min, max=max}
    return image.scale(img, img:size(img:dim())*zoom, img:size(img:dim()-1)*zoom, 'simple')
end
----

function clsOf(output)
    local m, idx = torch.max(output, 1)
    return idx[1], m[1]
end

------------- Visualization of target class neuron (as in Zeiler2014) --- 

local ReLUZeiler = torch.class('nn.ReLUZeiler', 'nn.ReLU') --ReLU acting in both directions the same

function ReLUZeiler:updateGradInput(input, gradOutput)
    if (not self.gradInput:isSameSizeAs(gradOutput)) then
        self.gradInput = self.gradInput:resizeAs(gradOutput) 
    end
    self.gradInput:copy(self:updateOutput(gradOutput))
    return self.gradInput
end


local layer = opt.layer --or 0 to remove nothing
local plotpath = '/home/simonovm/tmp/' .. opt.dataset .. (opt.dataset == 'Sun397' and opt.sunMinSize or '') .. '/' .. (opt.plotname~="" and opt.plotname or layer) .. '/'
os.execute('mkdir -p "' .. plotpath .. '"')


for i,module in ipairs(model:listModules()) do
    if (module.modules) then
        for i,mo in ipairs(module.modules) do
           if torch.type(mo) == 'nn.ReLU' then module.modules[i] = nn.ReLUZeiler() end
        end
    end
end

if torch.type(model.modules[#model.modules]) == 'nn.LogSoftMax' then model.modules[#model.modules] = nil end

while #model.modules > layer do model.modules[#model.modules] = nil end

local expname = string.gsub(opt.network, ".*/[0-9]+-[0-9]+-([^/]+)/[^/]+", "%1")

for i=1,40 do    
    local input, target = dataset.trainData:at(i)
    --local input, target = dataset.testData:at(i)
 
    local output = model:forward(input)
    print('Original target: ' .. target .. ' (' .. dataset:classes()[target] .. ')')       
    
    if layer==0 then   
        local c,m = clsOf(output)
        print('Input classified as ' .. c)
        
        output:zero()
        output[target] = 1 --neuron to visualize is the target class indicator
        local heatmap = model:backward(input, output)
        
        image.save(plotpath .. i .. '.png', dispAndZoom(input, 6))
        image.save(plotpath .. i .. '_'.. expname .. '.png', dispAndZoom(heatmap:sum(1), 6, 0))
        
    else
        local m = torch.max(output)
        local idx = torch.eq(output,m)
        output:zero()
        output[idx] = 1 --neuron to visualize is the max activation in the layer (over all maps)
        local heatmap = model:backward(input, output)
        
        image.save(plotpath .. i .. '.png', dispAndZoom(input, 6))
        image.save(plotpath .. i .. '_'.. expname .. '.png', dispAndZoom(heatmap:sum(1), 6, 0))        
    end    
end

do return end


--[[
local nFeats = 10
local m = 5 
for i=1,model:get(m).output:size(1) do
    image.display{image=torch.abs(model:get(m).output[i]:narrow(1,1,nFeats)), zoom=6, min=0, max=1, win=win, legend='A'..i, padding=1, nrow=1}
end
image.display{image=torch.abs(model:get(m+1).output:narrow(1,1,nFeats)), zoom=6, min=0, max=1, win=win, legend='MP', padding=1, nrow=1}
--]]


local nFeats = 7
local m = 15
for i=1,#model:get(m).output do
    print(model:get(m).output[i]:size())
    print(model:get(m).output[i]:narrow(1,1,nFeats):min())
    print(model:get(m).output[i]:narrow(1,1,nFeats):max())
    image.display{image=torch.abs(model:get(m).output[i]:narrow(1,1,nFeats)), zoom=6,  win=win, legend='A'..i, padding=1, nrow=1}
end

--[[image.display{image=torch.abs(model:get(14).output:narrow(1,1,nFeats)), zoom=6, min=-1, max=1, win=win, legend='MP', padding=1, nrow=1}
print(model:get(14).output:narrow(1,1,nFeats):min())
    print(model:get(14).output:narrow(1,1,nFeats):max())--]]




        
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