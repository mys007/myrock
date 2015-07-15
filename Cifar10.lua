require 'torch'
require 'nn'
require 'image'
require 'myutils'
require 'strict'
require 'nagadomi/data_augmentation'
require 'nagadomi/preprocessing'
local _ = nil

--------------------------------------- A general labeled dataset ---------------------------------------

local Dataset = torch.class('torch.Dataset')

function Dataset:__init()
   self.data = nil
   self.labels = nil
   self.batchIdx = 0
   self.nSampleVariants = 1
   self.postprocFun = function(x,t) return x,t end
end

function Dataset:size()
    assert(self~=nil)
    return self.labels:size(1)
end

function Dataset:trainBatch(batchSize, sampling)
    assert(self~=nil and batchSize~=nil and sampling~=nil)
    
    -- TODO: is it really efficient, memory is allocated again and again
    local inputs = {}
    local targets = {}
    
    if (sampling == 'seq') then
        for i=1,batchSize do
            local x,t = self.postprocFun( self.data[self.batchIdx+1], self.labels[self.batchIdx+1] )
            table.insert(inputs, x)
            table.insert(targets, t)
            self.batchIdx = (self.batchIdx + 1) % self:size()
        end
    elseif (sampling == 'randperm') then
        for i=1,batchSize do
            if (self.batchIdx==0) then self.shuffle = torch.randperm(self:size()) end   --new perm in each epoch 
            local index = self.shuffle[self.batchIdx+1]
            local x,t = self.postprocFun( self.data[index], self.labels[index] )
            table.insert(inputs, x)
            table.insert(targets, t)
            self.batchIdx = (self.batchIdx + 1) % self:size()
        end
    elseif (sampling == 'rand') then    
        for i=1,batchSize do
            local index = math.random(1,self:size())
            local x,t = self.postprocFun( self.data[index], self.labels[index] )
            table.insert(inputs, x)
            table.insert(targets, t)
        end    
    end    
        
    return inputs, targets
end

function Dataset:at(k)
    assert(self~=nil and k~=nil)
    return self.postprocFun(self.data[k], self.labels[k])
end

function Dataset:toPyra(scales)
    assert(self~=nil and scales~=nil)
    assert(torch.isTensor(self.data[1]) and self.data[1]:type()~='torch.CudaTensor')   --before toPyra(), cuda()
    
    local newData = {}
    for i=1,self:size() do
        table.insert(newData, image.gaussianpyramid(self.data[i], scales))
    end
    self.data = newData
end

function Dataset:cuda()
    assert(self~=nil)
    
    --self.data = funcOnTensors(self.data, function (x) return x:cuda() end)     --currently disabled due to high GPU mem requirement. should put just batches to GPU mem
    --self.labels = self.labels:cuda()
end

-- will change input dimensions, no cropping/padding (unlike randRescaleCropPad)
function Dataset:rescale(factor)
    assert(self~=nil and factor~=nil)
    assert(torch.isTensor(self.data[1]) and self.data[1]:type()~='torch.CudaTensor')   --before toPyra(), cuda()
    
    local newW = math.floor(self.data[1]:size(3)*factor + 0.5)
    local newH = math.floor(self.data[1]:size(2)*factor + 0.5)
    local newData = torch.Tensor(self:size(), 3, newH, newW)
    for i=1,self:size() do
        newData[i] = image.scale(self.data[i], newW, newH)
    end
    self.data = newData    
end

-- note: this is randomized and thus will produce different test results with different seeds!
function Dataset:randRescale(opt, minFactor, maxFactor)
    assert(self~=nil and minFactor~=nil and maxFactor~=nil)
    assert(torch.isTensor(self.data[1]) and self.data[1]:type()~='torch.CudaTensor')   --before toPyra(), cuda()

    local rngState = torch.getRNGState()    --don't mess up repeatability based on whether we randomize dataset. and don't get messed by e.g. module initialization
    torch.manualSeed(opt.seed)
    
    local newData = {}
    for i=1,self:size() do
        local factor = torch.uniform(minFactor,maxFactor)
        local src = self.data[i]
        local resc = image.scale(src, math.ceil(src:size(3)*factor), math.ceil(src:size(2)*factor))
        table.insert(newData, resc)
    end    
    self.data = newData
      
    torch.setRNGState(rngState)
end

-- note: this is randomized and thus will produce different test results with different seeds!
function Dataset:randGBlur(opt, minSigma, maxSigma)
    assert(self~=nil and minSigma~=nil and maxSigma~=nil)
    assert(torch.isTensor(self.data[1]) and self.data[1]:type()~='torch.CudaTensor')   --before toPyra(), cuda()

    local rngState = torch.getRNGState()    --don't mess up repeatability based on whether we randomize dataset. and don't get messed by e.g. module initialization
    torch.manualSeed(opt.seed)
   
    for i=1,self:size() do
        local sigma = torch.uniform(minSigma, maxSigma)
        local k = gaussianfilter2D(9, sigma)
        self.data[i] = image.convolve(self.data[i], k, 'same')
    end    
      
    torch.setRNGState(rngState)
end

-- note: this is randomized and thus will produce different test results with different seeds!
function Dataset:randRescaleCropPad(opt, minFactor, maxFactor)
    assert(self~=nil and minFactor~=nil and maxFactor~=nil)
    assert(torch.isTensor(self.data[1]) and self.data[1]:type()~='torch.CudaTensor')   --before toPyra(), cuda()

    local rngState = torch.getRNGState()    --don't mess up repeatability based on whether we randomize dataset. and don't get messed by e.g. module initialization
    torch.manualSeed(opt.seed)

    for i=1,self:size() do
        local factor = torch.uniform(minFactor,maxFactor)
        local src = self.data[i]
        local resc = image.scale(src, math.ceil(src:size(3)*factor), math.ceil(src:size(2)*factor))
        local dx = math.abs(resc:size(3) - src:size(3))
        local dy = math.abs(resc:size(2) - src:size(2))
        
        -- upsampling -> central crop         
        if (factor>1) then
            self.data[i] = image.crop(resc, math.ceil(dx/2), math.ceil(dy/2), resc:size(3)-math.floor(dx/2), resc:size(2)-math.floor(dy/2))
            --image.display{image=self.data[i], legend='aft', zoom=4}
        -- downsampling -> pad with 0          
        else
            src:fill(0)
            src[{ {}, {math.ceil(dy/2)+1, src:size(2)-math.floor(dy/2)}, {math.ceil(dx/2)+1, src:size(3)-math.floor(dx/2)} }] = resc
        end   
    end

    torch.setRNGState(rngState)
end


--------------------------------------- Cifar10---------------------------------------
-- based on https://raw.githubusercontent.com/torch/demos/master/train-on-cifar/train-on-cifar.lua

local Cifar10 = torch.class('torch.Cifar10')

function Cifar10:__init(config)
    assert(config ~= nil)
    
    _, self.dir, self.nValidations, self.nSampleRatio, self.augmentCropScaleFlip, self.normalizationMode, self.sampleAllSets = 
        xlua.unpack({config}, 'torch.Cifar10', nil,
        {arg='dir', type='string', help='', req=false, default=os.getenv('HOME')..'/datasets/cifar-10-batches-t7'},
        {arg='nValidations', type='number', help='', req=false, default=5000},
        {arg='nSampleRatio', type='number', help='', req=false, default=1},
        {arg='augmentCropScaleFlip', type='number', help='', req=false, default=0}, -- 0=no, 1=train, 2=all
        {arg='normalizationMode', type='string', help='', req=false, default='YuvScnChStat'},
        {arg='sampleAllSets', type='number', help='', req=false, default=0}
    )
    if (self.nSampleRatio<1) then self.nSampleRatio=1 end
    
    -- try to load self from cache
    local paramstr = self.nValidations..'_'..self.nSampleRatio..'_'..self.augmentCropScaleFlip..'_'..self.normalizationMode..'_'..self.sampleAllSets
    local success, cachedSelf = pcall(torch.load, os.getenv('HOME')..'/datasets/cache/'..torch.type(self)..'_'..paramstr..'.bin')
    if (success) then 
        print('Reusing cache')
        self.trainData = cachedSelf.trainData; self.validData = cachedSelf.validData; self.testData = cachedSelf.testData
        assert(self.testData.nSampleVariants~=nil)
        return 
    end
    
    self:load()
    print('Starting augmentation/normalization ' .. self.trainData:size() .. ' ' .. self.validData:size() .. ' '.. self.testData:size())
    
    if (self.augmentCropScaleFlip>0) then
        -- 36x increase in datasize! Image size reduces to 24x24
        -- TODO: the transformation should be actually drawn on the fly, nothing should be precomputed.. 
        self.trainData.data, self.trainData.labels = data_augmentation(self.trainData.data, self.trainData.labels); self.trainData.nSampleVariants = 36
        if (self.augmentCropScaleFlip==2) then
            self.validData.data, self.validData.labels = data_augmentation(self.validData.data, self.validData.labels); self.validData.nSampleVariants = 36
            self.testData.data, self.testData.labels = data_augmentation(self.testData.data, self.testData.labels); self.testData.nSampleVariants = 36
        else
            self.validData.data = self.validData.data:narrow(4, 5, 24):narrow(3, 5, 24):clone()
            self.testData.data = self.testData.data:narrow(4, 5, 24):narrow(3, 5, 24):clone()
        end
    end
    
    if (self.normalizationMode == 'RgbZca') then
        -- performs z-score normalization per element (but it should be gcn, as indicated in the drop-out paper...)
        -- followed by zca-whitening (http://stats.stackexchange.com/questions/117427/what-is-the-difference-between-zca-whitening-and-pca-whitening)
        --                           (http://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)
        local params = preprocessing(self.trainData.data)
        preprocessing(self.validData.data, params)
        preprocessing(self.testData.data, params)
    elseif (self.normalizationMode == 'RgbZcaPylearn') then
        --per-example normalization as in pylearn followed by zca-whitening (hopefully similar to pylearn:P)  
        local params = preprocessing_pylearn(self.trainData.data)
        preprocessing_pylearn(self.validData.data, params)
        preprocessing_pylearn(self.testData.data, params)        
    elseif (self.normalizationMode == 'YuvScnChStat') then
        -- converts to yuv, performs per-channel uv normalization and local contrast normalization on y 
        self:normalizeYuvScnChStat()
    elseif (self.normalizationMode == 'none') then
    
    else
        assert(false, 'unknown normalizationMode')
    end    
    
    torch.save(os.getenv('HOME')..'/datasets/cache/'..torch.type(self)..'_'..paramstr..'.bin', self)
    
    --> self.trainData ; self.validData ; self.testData
end

function Cifar10:cuda()
    for _,s in pairs{self.trainData, self.validData, self.testData} do
        s:cuda()
    end
end

function Cifar10:toPyra(scales)
    for _,s in pairs{self.trainData, self.validData, self.testData} do
        s:toPyra(scales)
    end
end

function Cifar10:toScalespaceTensor(scales)
    for _,s in pairs{self.trainData, self.validData, self.testData} do
        s:toScalespaceTensor(scales)
    end
end

function Cifar10:setPostprocFun(fn)
    for _,s in pairs{self.trainData, self.validData, self.testData} do
        s.postprocFun = fn
    end
end

function Cifar10:classes()
    return {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}
end


function Cifar10:load()
    assert(self~=nil)
    
    -- download dataset
    if not paths.dirp(self.dir) then
       local www = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar-10-torch.tar.gz'
       local tar = paths.basename(www)
       os.execute('wget ' .. www .. '; '.. 'tar xvf ' .. tar)
    end

    -- load train dataset
    self.trainData = torch.Dataset()
    self.trainData.data = torch.Tensor(50000, 32*32*3)
    self.trainData.labels = torch.Tensor(50000)

    for i = 0,4 do
       local subset = torch.load(self.dir .. '/data_batch_' .. (i+1) .. '.t7', 'ascii')
       self.trainData.data[{ {i*10000+1, (i+1)*10000} }] = subset.data:t():float()
       self.trainData.labels[{ {i*10000+1, (i+1)*10000} }] = subset.labels:float() + 1
    end
    
    -- validation dataset as the first self.nValidations samples (my own def)
    --  (out of technical reasons, it has to have at least one sample:)
    self.validData = torch.Dataset()
    self.validData.data = self.trainData.data[{ {1, math.max(1,self.nValidations)} }]
    self.validData.labels = self.trainData.labels[{ {1, math.max(1,self.nValidations)} }]
    self.trainData.data = self.trainData.data[{ {self.nValidations+1, 50000} }]
    self.trainData.labels = self.trainData.labels[{ {self.nValidations+1, 50000} }]
    
    -- load test dataset
    local subset = torch.load(self.dir .. '/test_batch.t7', 'ascii')
    self.testData = torch.Dataset()
    self.testData.data = subset.data:t():float()
    self.testData.labels = subset.labels[1]:float() + 1
    
    -- crop and reshape data
    for i,s in pairs{self.trainData, self.validData, self.testData} do
        if i==1 or self.sampleAllSets>0 then
            s.data = s.data[{ {1, s:size()/self.nSampleRatio} }]
            s.labels = s.labels[{ {1, s:size()/self.nSampleRatio} }]
        end    
        s.data = s.data:reshape(s:size(),3,32,32)
    end
end


----------------------------------------------------------------------
-- normalize sets by the training set
-- train-on-cifar version: the result is in YUV (!), SCN for Y and per-channel (!) normalization for U,V
function Cifar10:normalizeYuvScnChStat()
    assert(self~=nil)
    
    -- preprocess trainSet  --TODO: could also use image.lcn() ??
    local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
    for i = 1,self.trainData:size() do
       -- rgb -> yuv
       local rgb = self.trainData.data[i]
       local yuv = image.rgb2yuv(rgb)
       -- normalize y locally:
       yuv[{1}] = normalization:forward(yuv[{{1}}])
       self.trainData.data[i] = yuv
    end
    -- normalize u globally:
    local mean_u = self.trainData.data[{ {},2,{},{} }]:mean()
    local std_u = self.trainData.data[{ {},2,{},{} }]:std()
    self.trainData.data[{ {},2,{},{} }]:add(-mean_u)
    self.trainData.data[{ {},2,{},{} }]:div(-std_u)
    -- normalize v globally:
    local mean_v = self.trainData.data[{ {},3,{},{} }]:mean()
    local std_v = self.trainData.data[{ {},3,{},{} }]:std()
    self.trainData.data[{ {},3,{},{} }]:add(-mean_v)
    self.trainData.data[{ {},3,{},{} }]:div(-std_v)
    
    -- preprocess valid/testSet
    for _,s in pairs{self.validData, self.testData} do
        for i = 1,s:size() do
           -- rgb -> yuv
           local rgb = s.data[i]
           local yuv = image.rgb2yuv(rgb)
           -- normalize y locally:
           yuv[{1}] = normalization:forward(yuv[{{1}}])
           s.data[i] = yuv
        end
        -- normalize u globally:
        s.data[{ {},2,{},{} }]:add(-mean_u)
        s.data[{ {},2,{},{} }]:div(-std_u)
        -- normalize v globally:
        s.data[{ {},3,{},{} }]:add(-mean_v)
        s.data[{ {},3,{},{} }]:div(-std_v)
    end
end




--------------------------------------- Cifar10ClrReg---------------------------------------
-- data are greyscale images (1x32x32), labels are color images (rgb, 3x32x32)

local Cifar10ClrReg, Cifar10ClrReg_parent = torch.class('torch.Cifar10ClrReg', 'torch.Cifar10')

function Cifar10ClrReg:classes()
    return {}
end

function Cifar10ClrReg:load()
    Cifar10ClrReg_parent.load(self)
    
    for i,s in pairs{self.trainData, self.validData, self.testData} do
        s.labels = s.data
        s.data = torch.Tensor(s:size(),1,32,32)
        s.data:zero():add(0.299, s.labels:select(2,1)):add(0.587, s.labels:select(2,2)):add(0.114, s.labels:select(2,3))
    end
end    



--------------------------------------- TEST ---------------------------------------
--[[
local mytest = {}
local OFFmytest = {}
local tester = torch.Tester()

function OFFmytest.testRescale()
    torch.setdefaulttensortype('torch.FloatTensor')
    torch.manualSeed(1)

    local dataset = torch.Cifar10({nSampleRatio=500})
    dataset.trainData:randRescale(1.5, 2)
    --dataset.trainData:randRescale(0.5, 0.9)
end

function mytest.testCifar10ClrReg()
    torch.setdefaulttensortype('torch.FloatTensor')
    torch.manualSeed(1)
    
    local dataset = torch.Cifar10ClrReg({nSampleRatio=1, normalizationMode='RgbZca', nValidations='5000'})
    local d, i = dataset.trainData:at(1)
    image.display{image=d, zoom=6}
    image.display{image=i, zoom=6}                   
end


tester:add(mytest)
tester:run()--]]