require 'Cifar10'
require 'strict'
local _ = nil

local Sun397 = torch.class('torch.Sun397')

function Sun397:__init(config)
    assert(config ~= nil)
    
    _, self.dir, self.nValidations, self.nSampleRatio, self.partition, self.minSize, self. normalizationMode, self.squareCrop, self.fullTrainSet, self.sampleAllSets = 
        xlua.unpack({config}, 'torch.Cifar10', nil,
        {arg='dir', type='string', help='', req=false, default='/media/simonovm/Slow/datasets/SUN'},
        {arg='nValidations', type='number', help='', req=false, default=5000},
        {arg='nSampleRatio', type='number', help='', req=false, default=1},
        {arg='partition', type='number', help='', req=true, default=1},
        {arg='minSize', type='number', help='', req=true, default=64},
        {arg='normalizationMode', type='string', help='', req=false, default='zscore'},
        {arg='squareCrop', type='boolean', help='', req=true, default=true},
        {arg='fullTrainSet', type='boolean', help='', req=true, default=true},
        {arg='sampleAllSets', type='boolean', help='', req=false, default=false}
    )
    if (self.nSampleRatio<1) then self.nSampleRatio=1 end
    
    -- try to load self from cache
    local b = function (p) return p and '1' or '0' end
    local paramstr = self.nValidations..'_'..self.nSampleRatio..'_'..self.partition..'_'..self.minSize..'_'..self.normalizationMode..'_'..b(self.squareCrop)..'_'..b(self.fullTrainSet)..'_'..b(self.sampleAllSets)
    local success, cachedSelf = pcall(torch.load, '/home/simonovm/datasets/cache/sun_'..paramstr..'.bin')
    if (success) then 
        print('Reusing cache')
        self.trainData = cachedSelf.trainData; self.validData = cachedSelf.validData; self.testData = cachedSelf.testData; self.classNames = cachedSelf.classNames
        return 
    end
    
    self:load()
    print('Starting augmentation/normalization ' .. self.trainData:size() .. ' ' .. self.validData:size() .. ' '.. self.testData:size())
    
    if torch.isTensor(self.trainData.data) then
        if (self.normalizationMode == 'RgbZca') then
            -- performs z-score normalization per element (but it should be gcn, as indicated in the drop-out paper...)
            -- followed by zca-whitening (http://stats.stackexchange.com/questions/117427/what-is-the-difference-between-zca-whitening-and-pca-whitening)
            --                           (http://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)
            local params = preprocessing(self.trainData.data)
            preprocessing(self.validData.data, params)
            preprocessing(self.testData.data, params)
        elseif (self.normalizationMode == 'zscore') then
            --per element z-score normalization
            local m, s = global_contrast_normalization(self.trainData.data)
            global_contrast_normalization(self.validData.data, m, s)
            global_contrast_normalization(self.testData.data, m, s)  
        elseif (self.normalizationMode == 'none') then
                 
        else
            assert(false, 'unknown normalizationMode')
        end    
    else
        -- TODO: per-image contrast normalization only? or global mean/std, not per element?
    
    end
    
    torch.save('/home/simonovm/datasets/cache/sun_'..paramstr..'.bin', self)
end


function Sun397:cuda()
    for _,s in pairs{self.trainData, self.validData, self.testData} do
        s:cuda()
    end
end

function Sun397:toPyra(scales)
    for _,s in pairs{self.trainData, self.validData, self.testData} do
        s:toPyra(scales)
    end
end

function Sun397:toScalespaceTensor(scales)
    for _,s in pairs{self.trainData, self.validData, self.testData} do
        s:toScalespaceTensor(scales)
    end
end

function Sun397:setPostprocFun(fn)
    for _,s in pairs{self.trainData, self.validData, self.testData} do
        s.postprocFun = fn
    end
end

function Sun397:classes()
    return self.classNames
end


function Sun397:load()    
    --load classes
    local f = assert(io.open(self.dir .. '/Partitions/ClassName.txt', 'r'))
    local fstr = f:read('*all')
    f:close()
    
    local dict = {}
    self.classNames = {}
    for s in string.gmatch(fstr, "%S+") do
        table.insert(self.classNames, s)
        dict[s] = #self.classNames
    end     
 
    --load train dataset
    if self.fullTrainSet then
        self.trainData = self:loadList(self.dir .. '/Partitions/all.txt', string.format('%s/Partitions/Testing_%02d.txt', self.dir, self.partition), dict)
    else    
        self.trainData = self:loadList(string.format('%s/Partitions/Training_%02d.txt', self.dir, self.partition), nil, dict)
    end
    self.testData = self:loadList(string.format('%s/Partitions/Testing_%02d.txt', self.dir, self.partition), nil, dict)
    
    -- validation dataset as the first self.nValidations samples (my own def)
    --  (out of technical reasons, it has to have at least one sample:)
    self.validData = torch.Dataset()
    if torch.isTensor(self.trainData.data) then
        self.validData.data = self.trainData.data[{ {1, math.max(1,self.nValidations)}, {},{},{} }]
        self.trainData.data = self.trainData.data[{ {self.nValidations+1, self.trainData:size()}, {},{},{} }]
    else
        --TODO tables
    end
    self.validData.labels = self.trainData.labels[{ {1, math.max(1,self.nValidations)} }]
    self.trainData.labels = self.trainData.labels[{ {self.nValidations+1, self.trainData:size()} }]
    
    -- crop data
    for i,s in pairs{self.trainData, self.validData, self.testData} do
        if i==1 or self.sampleAllSets then
            if torch.isTensor(s.data) then
                s.data = s.data[{ {1, s:size()/self.nSampleRatio}, {},{},{} }]
            else
                --TODO tables
            end        
            s.labels = s.labels[{ {1, s:size()/self.nSampleRatio} }]
        end    
    end

end    
    
    
function Sun397:readListDiff(listpath, victimspath)
    assert(listpath~=nil)
    
    --load victims as dictionary
    local victims = {}
    local nVictims = 0;
    if victimspath~=nil then
        local f = assert(io.open(victimspath, 'r'))
        local fstr = f:read('*all')
        f:close()
    
        for s in string.gmatch(fstr, "%S+") do
            victims[s] = 1
            nVictims = nVictims + 1
        end     
    end
    
    --load list and ignore victims
    local f = assert(io.open(listpath, 'r'))
    local fstr = f:read('*all')
    f:close()        

    local paths = {}
    local nPaths = 0;
    for s in string.gmatch(fstr, "%S+") do
        if not victims[s] then table.insert(paths, s) end
        nPaths = nPaths + 1
    end  
    
    print('List #' .. nPaths .. ', victims #' .. nVictims .. ', result #' .. #paths)
    
    return paths
end


function Sun397:loadList(listpath, victimspath, dict)
    assert(listpath~=nil and dict~=nil)
    
    local dataset = torch.Dataset()
    local paths = self:readListDiff(listpath, victimspath)
    local n = #paths
    
    -- we randomize the order
    local rngState = torch.getRNGState()
    torch.manualSeed(4321)
    local shuffle = torch.randperm(n)
    torch.setRNGState(rngState)    
     
    if self.squareCrop then
        dataset.data = torch.Tensor(n, 3, self.minSize, self.minSize)
        dataset.labels = torch.Tensor(n)
        dataset.filenames = {}

        for i,path in ipairs(paths) do
            xlua.progress(i, #paths)
            
            -- try all tricks to load the image and use a black one if all failed
            local filename = self.dir .. '/SUN397' .. path
            local ok, img = pcall(image.load, filename, 3)
            if not ok then ok, img = pcall(image.loadPNG, filename, 3) end
            if not ok then os.execute('convert "' .. filename .. '" "' .. filename .. '"'); ok, img = pcall(image.load, filename, 3) end
            if not ok then print('Unsupported file format (not jpg or png and conversion failed): '..dataset.filenames[i]); img = dataset.data[1]:clone():zero() end
            
            -- rescale & central crop
            local factor = self.minSize / math.min(img:size(2), img:size(3)) 
            img = image.scale(img, math.floor(img:size(3)*factor+0.5), math.floor(img:size(2)*factor+0.5))
            local x1 = math.floor((img:size(3)-self.minSize)/2) + 1
            local y1 = math.floor((img:size(2)-self.minSize)/2) + 1
            
            local idx = shuffle[i]
            dataset.data[idx] =img:narrow(3, x1, self.minSize):narrow(2, y1, self.minSize) 
            dataset.labels[idx] = dict[string.match(path, "(.*)/[^/]+$")]
            dataset.filenames[idx] = filename
                
            if (i % 100 == 0) then collectgarbage() end
        end        
    
    else
    
            --TODO!
    end    

    return dataset
end



--------------------------------------- TEST ---------------------------------------
--[[
local mytest = {}
local OFFmytest = {}
local tester = torch.Tester()

function OFFmytest.testSun()
    torch.setdefaulttensortype('torch.FloatTensor')
    torch.manualSeed(1)
    
    local dataset = torch.Sun397({nSampleRatio=1, 
                                   partition=1, 
                                   minSize=64,
                                   squareCrop=true,
                                   fullTrainSet=true,
                                   nValidations=1000,
                                   normalizationMode='zscore'})                        
end


function mytest.classHistograms()
    torch.setdefaulttensortype('torch.FloatTensor')
    torch.manualSeed(1)
    
    local dataset = torch.Sun397({nSampleRatio=1, 
                                   partition=1, 
                                   minSize=32,
                                   squareCrop=true,
                                   fullTrainSet=true,
                                   nValidations=1000,
                                   normalizationMode='zscore'})      
               
    local nsamp = torch.histc(dataset.validData.labels, #dataset:classes(), 1, #dataset:classes()+0.01)
    print(nsamp)
end



tester:add(mytest)
tester:run()
--]]