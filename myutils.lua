require 'torch'
require 'nn'
require 'strict'

----------------------------------------------------------------------
--Matlab's "Break on error"
function breakpt(err)
    print(err)  --set breakpoint here
end    
--xpcall(function() model:getParameters() end, breakpt)

----------------------------------------------------------------------
function string.starts(String,Start)
   return string.sub(String,1,string.len(Start))==Start
end

function string.ends(String,End)
   return End=='' or string.sub(String,-string.len(End))==End
end

function string.trim(s)
  return (s:gsub("^%s*(.-)%s*$", "%1"))
end

function string.split(str, delim, maxNb)
    -- Eliminate bad cases...
    if string.find(str, delim) == nil then
        return { str }
    end
    if maxNb == nil or maxNb < 1 then
        maxNb = 0    -- No limit
    end
    local result = {}
    local pat = "(.-)" .. delim .. "()"
    local nb = 0
    local lastPos
    for part, pos in string.gfind(str, pat) do
        nb = nb + 1
        result[nb] = part
        lastPos = pos
        if nb == maxNb then break end
    end
    -- Handle the last field
    if nb ~= maxNb then
        result[nb + 1] = string.sub(str, lastPos)
    end
    return result
end

----------------------------------------------------------------------
--Creates isotropic 2D gaussian. Unlike image.gaussian this is consistent with matlab's fspecial
function gaussianfilter2D(size, sigma)
    assert((size+1) % 2 == 0 and sigma~=nil)
    
    local gauss = torch.Tensor(size, size)
    local mean = (size-1)/2
    for i=1,size do
        for j=1,size do
            gauss[i][j] = math.exp(-math.pow((i-1-mean)/sigma,2)/2 - math.pow((j-1-mean)/sigma,2)/2)
        end
    end
    gauss:div(gauss:sum())
    return gauss
end

----------------------------------------------------------------------
--Applies function f(arg) to a tensor. Can deal with tensors in recursive tables.
function funcOnTensors(input, f)
    assert(input~=nil and f~=nil)
    
    if (torch.isTensor(input)) then
        return f(input)
    elseif (torch.type(input) == 'table') then
        local output = {}
        for k,v in pairs(input) do 
            output[k] = funcOnTensors(v, f)
        end
        return output
    else
        error('funcOnTensors: unknown type ' .. torch.type(input))
    end
end

----------------------------------------------------------------------
-- Prints the memory consumed by a model by recursively crawling its tensors. Prints pointer to storages for reasoning about sharing.
function printModelMemUse(model)
    local function recursivePrintMem(str,param)
        if (torch.isTensor(param)) then
            local ptr = param:storage() and string.format('0x%X',torch.pointer(param:storage())) or ''
            print(str..'['..ptr..']:\t'..(param:nElement()*4/1024/1024)..'MB')
        elseif (torch.type(param) == 'table') then
            for k,v in pairs(param) do recursivePrintMem(str..'/'..k,v) end
        elseif torch.isTypeOf(param, 'nn.Module') then
            str = str..'('..torch.typename(param)..')'
            for k,v in pairs(param) do recursivePrintMem(str..'/'..k,v) end            
        end
    end
    recursivePrintMem('/', model)
end

----------------------------------------------------------------------
-- Prints a Lua table
function printTable(input)
    local function recursivePrint(input)
        if torch.type(input) == 'table' then
            local str = '{'
            for k,v in pairs(input) do 
                str = str..k..'='..recursivePrint(v)..', '
            end
            return string.sub(str, 1, #str-2)..'}'
        elseif torch.type(input) == 'boolean' then
            return input and 'true' or 'false'
        else    
            return input
        end
    end
    print(recursivePrint(input))
end

----------------------------------------------------------------------
-- Returns a string of tensor dimension AxBxCx..
function formatSizeStr(input)
    local function sizeStr(x) 
        if x:nDimension() == 0 then
            return 'empty'
        else
            local str = ''
            for i=1,x:nDimension() do
                str = str .. x:size(i) .. (i ~= x:nDimension() and 'x' or '')
            end
            return str
        end
    end
    
    local ret = funcOnTensors(input, sizeStr)
    if (torch.type(input) == 'table') then
        return table.concat(ret, ",") 
    end
    return ret
end

----------------------------------------------------------------------
--Efficiently converts a table of tensors to another type (esp. cuda<->ram): only one memory transfer call is made
-- (nx uploading b bytes is much more expensive than 1x uploading nb bytes)
function tensortableType(input, type)
    assert(torch.type(input) == 'table' and type~=nil)
    
    local n = 0
    for k,v in pairs(input) do n = n + v:nElement() end

    local storage = torch.Tensor(n):typeAs(input[1])
    n = 1
    for k,v in pairs(input) do  --copy into a single storage 
        storage:narrow(1, n, v:nElement()):copy(v)
        n = n + v:nElement()
    end 
    
    storage = storage:type(type) --convert the storage
    
    local output = {}
    n = 1
    for k,v in pairs(input) do --reconstruct the tensors (all will be viewing the single storage)
        output[k] = storage:narrow(1, n, v:nElement()):viewAs(v)
        n = n + v:nElement()
    end
    return output
end

----------------------------------------------------------------------
-- Adds a 0-padding layer to model if the input isn't exactly divisible by max-poling and max-pool would floor-down the size.
--TODO: maybe should use Sergey's image-nn version... (avoids padding)
function zeroPadMPCeil(model, w, h, kW, kH, dW, dH)
    assert(model, w and h and kW and kH)
    dW = dW or kW; dH = dH or kH
    local padW = math.max(0, w - kW) % dW
    local padH = math.max(0, h - kH) % dH
    if (padW>0 or padH>0) then
        model:add(nn.SpatialZeroPadding(0, padW>0 and (dW - padW) or 0, 0, padH>0 and (dH - padH) or 0))
    end
end

----------------------------------------------------------------------
-- Intersection and Union of axis-aligned boxes. Input format of indexes as for Tensor.operator[]
function boxIntersectionUnion(idx1, idx2)
    assert(#idx1==#idx2)
    local inter, area1, area2 = 1, 1, 1
    for i=1,#idx1 do
        inter = inter * math.max(0, math.min(idx1[i][2],idx2[i][2])+1 - math.max(idx1[i][1],idx2[i][1]))
        area1 = area1 * (idx1[i][2]+1 - idx1[i][1])
        area2 = area2 * (idx2[i][2]+1 - idx2[i][1])
    end
    local union = area1 + area2 - inter
    return inter, union
end 

----------------------------------------------------------------------
-- L2-Distance between box centers. Input format of indexes as for Tensor.operator[]
function boxCenterDistance(idx1, idx2)
    assert(#idx1==#idx2)
    local dist = 0
    for i=1,#idx1 do
        dist = dist + ((idx1[i][2] + idx1[i][1])/2 - (idx2[i][2] + idx2[i][1])/2)^2
    end
    return math.sqrt(dist)
end 

------------------------------
-- pads a patch with pad at each side (or with degenPad for the degenerate dim)
function boxPad(indices, pad, degenPad)
    local out = {}
    for i=1,#indices do
        if indices[i][1]==indices[i][2] then 
            out[i] = {indices[i][1] - degenPad, indices[i][2] + degenPad}
        else
            out[i] = {indices[i][1] - pad, indices[i][2] + pad}
        end
    end
    return out
end

----------------------------------------------------------------------
-- Plots weights in nn.SpatialConvolutionMM
function plotSpatialConvolutionMM(convmm, win, legend)
    --local delta = torch.Tensor(3,20,20):zero()
    --delta[1][10][10] = 1; delta[2][10][10] = 1; delta[3][10][10] = 1
    --image.display{image=convmm:forward(delta:cuda()):float(), zoom=4}
    local weights = convmm.weight:float():view(convmm.nOutputPlane, convmm.nInputPlane, convmm.kH, convmm.kW)
    
    -- split by output layers
    local weightLayers = {}
    for i=1,weights:size(1) do
        local fmap = image.toDisplayTensor({input=weights[i], nrow=convmm.nInputPlane, padding=1}) --, min=-1, max=1
        table.insert(weightLayers, fmap)
    end    
    
    print('L1 norm of weights: ' .. legend .. ' :' .. torch.sum(torch.abs(weights)))
     
    return image.display{image=weightLayers, zoom=6, min=-1, max=1, win=win, legend=legend, padding=1, nrow=1} --nrow=10
end    

----------------------------------------------------------------------
-- Initializes model weights as in He2015
function resetHe2015(model, opt)
    for _,module in ipairs(model:listModules()) do
        if module:parameters() ~= nil and (module.weight ~= nil or module.bias~=nil) then
            local n = 0
            if torch.typename(module) == 'nn.SpatialConvolutionMM' or torch.typename(module) == 'nn.SpatialConvolutionLRScale' then    
                n = module.kW*module.kH*module.nInputPlane
                --if (opt and opt.modelName == 'pyra') then n = n*opt.pyraFanIn end --(multiple conv results are summed up if createPyramidConvolutionPostscale used)
            elseif torch.typename(module) == 'nn.VolumetricConvolution' then
                n = module.kT*module.kW*module.kH*module.nInputPlane             
            elseif torch.typename(module) == 'nn.Linear' then    
                n = module.weight:size(2)
            else
                assert(false, 'Unknown module with parameters ' .. torch.typename(module))
            end
            local stdv = math.sqrt(2/n)
            module.weight:normal(0, stdv)  
            module.bias:zero()
        end
    end
end

----------------------------------------------------------------------
-- Plain old Gaussian noise with fixed stddev and 0 bias (as in network-in-network)
function resetNin(model)
    for _,module in ipairs(model:listModules()) do
        if module:parameters() ~= nil and (module.weight ~= nil or module.bias~=nil) then
            assert(torch.typename(module) ~= 'ccn2.SpatialConvolution', 'not impl yet')
            module.weight:normal(0, 0.05)  
            module.bias:zero()
        end
    end
end

----------------------------------------------------------------------
-- Plain old Gaussian noise with stddev and bias (per module)
function resetGaussConst(model)
    for _,module in ipairs(model:listModules()) do
        if module:parameters() ~= nil and (module.weight ~= nil or module.bias~=nil) then
            --typically convolutions
            if module.resetGstddev ~= nil then
                if torch.typename(module) == 'ccn2.SpatialConvolution' then
                    local Wt = module.weight:view(-1, module.kH ,module.kH, module.nOutputPlane)
                    Wt = Wt:transpose(1, 4):transpose(2, 4):transpose(3, 4)
                    Wt:normal(0, module.resetGstddev)
                else   
                    module.weight:normal(0, module.resetGstddev)
                end      
                module.bias:fill(module.resetGbias or 0)
            --typically batchnorm, prelu, ..
            else
                module:reset()
            end
        end
    end
end

----------------------------------------------------------------------
-- Prepares parameters for resetGaussConst()
function gaussConstInit(module, wstddev, bval)
    assert(module and wstddev)
    module.resetGstddev = wstddev
    if bval then module.resetGbias = bval end
    return module        
end

----------------------------------------------------------------------
--Leaky ReLu (as in "Rectifier Nonlinearities Improve Neural Network Acoustic Models")
-- observation?: even if the weights of input modules are set to produce only negative input to the nonlinearity
-- due to large step size, there is still a hope for improvement (classical ReLu will never update the predecessors any more)
local LReLU, LReLU_parent = torch.class('nn.LReLU', 'nn.Module')

function LReLU:__init(alpha)
    LReLU_parent.__init(self)
    self.reluP = nn.ReLU()
    self.reluN = nn.ReLU()
    self.alpha = alpha or 0.01
end

function LReLU:updateOutput(input)
    if (not self.output:isSameSizeAs(input)) then
        self.output = self.output:resizeAs(input) 
    end
    
    self.reluP:updateOutput(input)
    self.reluN:updateOutput(-input)
    self.output:add(self.reluP.output, -self.alpha, self.reluN.output)
    return self.output
end

function LReLU:updateGradInput(input, gradOutput)
    if (not self.gradInput:isSameSizeAs(gradOutput)) then
        self.gradInput = self.gradInput:resizeAs(gradOutput) 
    end
    
    self.reluP:updateGradInput(input, gradOutput)
    self.reluN:updateGradInput(-input, gradOutput)
    self.gradInput:add(self.reluP.gradInput, self.alpha, self.reluN.gradInput)
    return self.gradInput
end

function LReLU:type(type)
   LReLU_parent.type(self,type)
   self.reluP:type(type)
   self.reluN:type(type)
   return self
end










local QuantileReLU, QuantileReLU_parent = torch.class('nn.QuantileReLU', 'nn.Threshold')

function QuantileReLU:__init(quantileToDrop, negsToo)
    assert(quantileToDrop~=nil and negsToo~=nil)
    QuantileReLU_parent.__init(self,0,0)
    self.sorted = torch.Tensor()
    self.quantile = quantileToDrop
    self.negsToo = negsToo 
end

function QuantileReLU:updateOutput(input)
    if self.negsToo then
        local inputVec = input:view(-1)
        local sorted,idx=torch.sort(torch.abs(inputVec))
        
        local qel = math.floor(inputVec:nElement() * self.quantile)+1
        local thresholdP, thresholdN = 0, 0
        for i=qel,1,-1 do --towards smaller elements
            if thresholdP == 0 and inputVec[idx[i]] >= 0 then thresholdP = inputVec[idx[i]] end
            if thresholdN == 0 and inputVec[idx[i]] <= 0 then thresholdN = inputVec[idx[i]] end
            if thresholdP > 0 and thresholdN < 0 then break end
        end    
        
        if (not self.sieve or not self.sieve:isSameSizeAs(input)) then self.sieve = input:clone() end
        self.sieve:map(input, function(_,xt) return (xt < thresholdN or xt > thresholdP) and 1 or 0 end)
        torch.cmul(self.output, input, self.sieve) 
        return self.output
    else
        if input:type()=="torch.CudaTensor" then
            self.sorted = self.sorted:float()
            local tmp = input:view(-1):float()
            self.sorted:sort(tmp)
        else    
            self.sorted:sort(input:view(-1))
        end    
        self.threshold = self.sorted[math.floor(self.sorted:nElement() * self.quantile)+1]
        return QuantileReLU_parent.updateOutput(self,input)
    end    
end

function QuantileReLU:updateGradInput(input, gradOutput)
    if self.negsToo then
        torch.cmul(self.gradInput, gradOutput, self.sieve)
        return self.gradInput
    else    
        return QuantileReLU_parent.updateGradInput(self,input,gradOutput)
    end
end








-- Multiclass hinge loss as in wiki (only the most violating class adjusted). Torch's MultiMarginCriterion implements sum 
--  over all margins and produces dense gradient (tries to compensate for all violations 'ahead'/cumulatively?)
-- Found out: diff of outputs allows them to grow to inf -> explodes without some other balancing (such as softmax)
local CrammerSingerCriterion = torch.class('nn.CrammerSingerCriterion', 'nn.Criterion')

function CrammerSingerCriterion:updateOutput(input, target)
    local backup = input[target]
    input[target] = -1e10
    local m, i = torch.max(input,1)
    input[target] = backup
    
    self.ma = 1 - input[target] + m[1]; 
    self.mai = i[1]
    return math.max(self.ma, 0)
end

function CrammerSingerCriterion:updateGradInput(input, target)
    if (not self.gradInput:isSameSizeAs(input)) then
        self.gradInput = self.gradInput:resizeAs(input) 
    end
    
    self.gradInput:zero()
    if self.ma > 0 then
        self.gradInput[target] = -1
        self.gradInput[self.mai] = 1
    end
    return self.gradInput
end

-- SVM Criterion as in caffe (http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1HingeLossLayer.html). There is no difference 
-- between outputs but the outputs are forced to be >1 for the correct class and <-1 for the others. Thus, they are not allowed
-- to grow to inf. But is more strict
local CaffeSVMCriterion = torch.class('nn.CaffeSVMCriterion', 'nn.Criterion')

function CaffeSVMCriterion:updateOutput(input, target)
    if (not self.margins or not self.margins:isSameSizeAs(input)) then
        self.margins = input:clone() 
    end
    self.margins:copy(input)
    self.margins[target] = -input[target]
    self.margins = self.margins + 1
    self.margins:clamp(0,1e20)
    
    if false then        --sparse variant (note: can be impl more efficiently)
        local m, i = torch.max(self.margins,1)
        local backup = self.margins[i[1]]
        self.margins:zero()
        self.margins[i[1]] = backup
    end    
    
    return self.margins:sum()
end

function CaffeSVMCriterion:updateGradInput(input, target)
    if (not self.gradInput:isSameSizeAs(input)) then
        self.gradInput = self.gradInput:resizeAs(input) 
    end
    
    self.gradInput:copy(torch.gt(self.margins,0))
    self.gradInput[target] = -self.gradInput[target]
    return self.gradInput
end



-- SpatialConvolutionMM with user-defined 'scale' parameter for accGradParameters (i.e. a learning rate factor)
local SpatialConvolutionLRScale, SpatialConvolutionLRScale_parent = torch.class('nn.SpatialConvolutionLRScale', 'nn.SpatialConvolutionMM')

function SpatialConvolutionLRScale:__init(scale,...)
   SpatialConvolutionLRScale_parent.__init(self, ...)
   self.scale = scale
end
function SpatialConvolutionLRScale:accGradParameters(input, gradOutput, scale)
   SpatialConvolutionLRScale_parent.accGradParameters(self, input, gradOutput, (scale or 1) * self.scale)
end




--follows https://github.com/BVLC/caffe/blob/master/src/caffe/layers/pooling_layer.cu 
-- ! works only for non-negative activations (so needs to follow ReLU)
local SpatialStochasticPooling, SpatialStochasticPooling_parent = torch.class('nn.SpatialStochasticPooling', 'nn.Module')

function SpatialStochasticPooling:__init(kW, kH, dW, dH)
   SpatialStochasticPooling_parent.__init(self)

   self.kW = kW
   self.kH = kH
   self.dW = dW or kW
   self.dH = dH or kH
   self.thresholds = torch.Tensor()
   self.indices = torch.Tensor()
   self.tmp1 = torch.Tensor()
   self.tmp2 = torch.Tensor()
end

function SpatialStochasticPooling:updateOutput(input)
    input.nn.SpatialAveragePooling_updateOutput(self, input)
    
    if self.train then
        self.output:mul(self.kW*self.kH):float() --avg to sum
        self.thresholds:resizeAs(self.output):uniform(0,1):cmul(self.output)
        self.indices = self.indices:resize(input:size()):int()
        
        assert(input:isContiguous() and self.output:isContiguous())
        local input_data = torch.data(input)
        local output_data = torch.data(self.output)
        local thresholds_data = torch.data(self.thresholds)
        local indices_data = torch.data(self.indices)
        
        local function sample(idx_o, idx_i)
            local cumsum = 0
            for h=0,self.kH-1 do
                for w=0,self.kW-1 do
                    cumsum = cumsum + input_data[idx_i+w]
                    if cumsum >= thresholds_data[idx_o] then
                        output_data[idx_o] = input_data[idx_i+w]
                        indices_data[idx_o] = idx_i+w
                        return
                    end
                end
                idx_i = idx_i + input:size(3)
            end               
        end
        
        local idx_o = 0
        for c=0,self.output:size(1)-1 do
            for y=0,self.output:size(2)-1 do
                for x=0,self.output:size(3)-1 do
                    local idx_i = c*input:size(2)*input:size(3) + y*self.dH*input:size(3) + x*self.dW
                    sample(idx_o, idx_i)
                    idx_o = idx_o + 1
                end
            end
        end
    else
        --sum_i(a_i^2)/sum_i(a_i)
        self.tmp1:resizeAs(self.output):copy(self.output):add(1e-30) --(avoid divide-by-zero problems)
        self.tmp2:resizeAs(input):copy(input):pow(2)
        self.tmp2.nn.SpatialAveragePooling_updateOutput(self, self.tmp2)
        self.output:cdiv(self.tmp1)
    end
   
    return self.output
end

function SpatialStochasticPooling:updateGradInput(input, gradOutput)
    if self.train then
        self.gradInput:resizeAs(input):zero()
        
        assert(gradOutput:isContiguous() and self.gradInput:isContiguous())
        local gradOutput_data = torch.data(gradOutput)
        local gradInput_data = torch.data(self.gradInput)
        local indices_data = torch.data(self.indices)
        
        local idx_o = 0
        for c=0,gradOutput:size(1)-1 do
            for y=0,gradOutput:size(2)-1 do
                for x=0,gradOutput:size(3)-1 do        
                    local idx = indices_data[idx_o]
                    gradInput_data[idx] = gradInput_data[idx] + gradOutput_data[idx_o]
                    idx_o = idx_o + 1
                end
            end
        end
    else
        assert(false, 'No backprop defined for testmode')
    end
    return self.gradInput    
end














--[[function remaxnorm(matrix, dim, maxnorm)    -- works for any D, but copies memory 
  local m1 = matrix:transpose(dim, 1):contiguous()
  -- collapse non-dim dimensions
  local m2 = m1:view(m1:size(1), -1)
  local norms = m2:abs():max(2):add(1e-7)
  -- clip
  local new_norms = norms:clone()
  new_norms[torch.gt(norms, maxnorm)] = maxnorm
  new_norms:cdiv(norms)
  -- renormalize
  m2:cmul(new_norms:expandAs(m2))
  return m1:transpose(dim, 1)
end--]]

-- Renormalizes the sub-tensors along dimension dim such that they do not exceed norm maxnorm.
function remaxnorm(matrix, dim, maxnorm)
    assert(matrix:dim()==2)
    local m1 = matrix:transpose(dim, 1)
    -- collapse non-dim dimensions
    local norms = m1:abs():max(2):add(1e-7)
    -- clip
    local new_norms = norms:clone()
    new_norms:apply(function(x) return math.min(x,maxnorm) end)
    
    --new_norms:maskedFill(torch.gt(norms, maxnorm), maxnorm)
    --new_norms[torch.gt(norms, maxnorm)] = maxnorm
    new_norms:cdiv(norms)
    -- renormalize
    m1:cmul(new_norms:expandAs(m1))
end






--note: get a path to this file from curdir: local f = string.gsub(debug.getinfo(1,'S').source, '@(.+)/[^/]+', '%1')


---

--[[
function image.gaussianpyramidMy(...)
   local dst,src,scales
   local args = {...}
   if select('#',...) == 3 then
      dst = args[1]
      src = args[2]
      scales = args[3]
   elseif select('#',...) == 2 then
      dst = {}
      src = args[1]
      scales = args[2]
   else
      print(dok.usage('image.gaussianpyramid',
                       'construct a Gaussian pyramid from an image', nil,
                       {type='torch.Tensor', help='input image', req=true},
                       {type='table', help='list of scales', req=true},
                       '',
                       {type='table', help='destination (list of Tensors)', req=true},
                       {type='torch.Tensor', help='input image', req=true},
                       {type='table', help='list of scales', req=true}))
      dok.error('incorrect arguments', 'image.gaussianpyramid')
   end
   if src:nDimension() == 2 then
      for i = 1,#scales do
         dst[i] = dst[i] or torch.Tensor()
         dst[i]:resize(src:size(1)*scales[i], src:size(2)*scales[i])
      end
   elseif src:nDimension() == 3 then
      for i = 1,#scales do
         dst[i] = dst[i] or torch.Tensor()
         dst[i]:resize(src:size(1), src:size(2)*scales[i], src:size(3)*scales[i])
      end
   else
      dok.error('src image must be 2D or 3D', 'image.gaussianpyramid')
   end
   
   local tmp = src
   for i = 1,#scales do
      if scales[i] == 1 then
         dst[i][{}] = tmp
      else
         image.scale(dst[i], tmp, 'simple')
      end
      local sigma = i==#scales and 1 or math.sqrt(scales[i]/scales[i+1])
      local k = gaussianfilter2D(9, sigma)
      tmp = image.convolve(dst[i], k, 'same')
   end
   return dst
end
--]]

--------------------------------------- TEST ---------------------------------------

--[[
local mytest = {}
local OFFmytest = {}
local tester = torch.Tester()
 
function OFFmytest.testTensortableType()
    torch.setdefaulttensortype('torch.FloatTensor')
    torch.manualSeed(1)
    
    local input = {}
    local expected = {}
    for i=1,5 do 
        input[i] = torch.rand(torch.uniform(1,3),torch.uniform(1,3),torch.uniform(1,3))
        expected[i] = input[i]:double() 
    end
    
    local output = tensortableType(input, 'torch.DoubleTensor')
    for i=1,#expected do 
        tester:assertTensorEq(output[i], expected[i], 1e-6)
    end
end

function OFFmytest.testLReLU()
    torch.setdefaulttensortype('torch.FloatTensor')
    torch.manualSeed(1)
    
    local input = torch.Tensor({1,2,0,-1,-2})
    local expected = torch.Tensor({1,2,0,-0.01,-0.02})
    local module = nn.LReLU()
    tester:assertTensorEq(module:forward(input), expected, 1e-6)
end

function OFFmytest.testLReLUJac()

    local ini = math.random(3,5)
    local inj = math.random(3,5)
    local ink = math.random(3,5)
    local input = torch.Tensor(ini,inj,ink):normal(0, 5)
    local module = nn.LReLU()
    local err = nn.Jacobian.testJacobian(module,input)      --needs to enable non-radnomized code for this
    tester:assertlt(err,1e-5, 'error on state ')
    local ferr,berr = nn.Jacobian.testIO(module,input)
    tester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
    tester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function OFFmytest.testQuantileReLU()
    torch.setdefaulttensortype('torch.FloatTensor')
    torch.manualSeed(1)
    
    --the value at quantile gets discarded too (strict inequality)
    local input = torch.Tensor({3,1,2,4})
    local expected1 = torch.Tensor({3,0,0,4})
    local expected2 = torch.Tensor({0,0,0,4})
    local module = nn.QuantileReLU(0.25, false)    
    tester:assertTensorEq(module:forward(input), expected1, 1e-6)
    module = nn.QuantileReLU(0.74, false)
    tester:assertTensorEq(module:forward(input), expected2, 1e-6)
    
    local input1 = torch.Tensor({-3,1,2,4})
    local expected1 = torch.Tensor({-3,0,0,4})
    local input2 = torch.Tensor({-3,1,2,-4})
    local expected2 = torch.Tensor({-3,0,0,-4})
    local input3 = torch.Tensor({3,-1,-2,4})
    local expected3 = torch.Tensor({3,0,0,4})    
    local module = nn.QuantileReLU(0.25, true)
    tester:assertTensorEq(module:forward(input1), expected1, 1e-6)    
    module = nn.QuantileReLU(0.49, true)
    tester:assertTensorEq(module:forward(input2), expected2, 1e-6)
    tester:assertTensorEq(module:forward(input3), expected3, 1e-6)             
end

function OFFmytest.testQuantileReLUJac()

    for i=1,2 do
        local ini = math.random(3,5)
        local inj = math.random(3,5)
        local ink = math.random(3,5)
        local input = torch.Tensor(ini,inj,ink):normal(0, 5)
        local module = nn.QuantileReLU(0.25, i==1)
        local err = nn.Jacobian.testJacobian(module,input)      --needs to enable non-radnomized code for this
        tester:assertlt(err,1e-5, 'error on state ')
        local ferr,berr = nn.Jacobian.testIO(module,input)
        tester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
        tester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
    end
end

function OFFmytest.testCrammerSingerCriterion()
    local crit = nn.CrammerSingerCriterion()
    local input = torch.Tensor({5, 4, 4.5, 5})
    tester:asserteq(crit:forward(input, 3), 1.5)
    tester:assertTensorEq(crit:backward(input, 3), torch.Tensor({1, 0, -1, 0}), 1e-6)
    
    local input = torch.Tensor({5, 4, 4.5})
    tester:asserteq(crit:forward(input, 1), 0.5)
    tester:assertTensorEq(crit:backward(input, 1), torch.Tensor({-1, 0, 1}), 1e-6)
    
    local input = torch.Tensor({5, 4, 4})
    tester:asserteq(crit:forward(input, 1), 0)
    tester:assertTensorEq(crit:backward(input, 1), torch.Tensor({0, 0, 0}), 1e-6)
    
    local input = torch.Tensor({5, -3, 3})
    tester:asserteq(crit:forward(input, 1), 0)
    tester:assertTensorEq(crit:backward(input, 1), torch.Tensor({0, 0, 0}), 1e-6)     
end


function OFFmytest.testCaffeSVMCriterion()

    local crit = nn.CaffeSVMCriterion()
    local input = torch.Tensor({-1, -0.75, 0.7, -0.9})
    
    print(crit:forward(input, 3))
    print(crit:backward(input, 3))
    
    tester:asserteq(crit:forward(input, 3), 0.25+0.3+0.1)
    tester:assertTensorEq(crit:backward(input, 3), torch.Tensor({0, 1, -1, 1}), 1e-6)
    
    local input = torch.Tensor({-1, -2, 1, -1.1})
    
    print(crit:forward(input, 3))
    print(crit:backward(input, 3))
    
    tester:asserteq(crit:forward(input, 3), 0)
    tester:assertTensorEq(crit:backward(input, 3), torch.Tensor({0, 0, 0, 0}), 1e-6)    
end


function OFFmytest.testRemaxnorm()
    torch.setdefaulttensortype('torch.FloatTensor')
    torch.manualSeed(1)
   
    if false then 
       local input = torch.Tensor({
                {{4,0,1,0}, {0,0,0,0}},
                {{2,0,0,0}, {0,0,0,0}},
                {{1,0,0,1}, {1,0,1,0}},
            })
    
        print(input)
        local o = input:clone()
        print(torch.renorm(input, math.huge, 2, 3))
        o = remaxnorm(o, 2, 3)
        print(o)
    else
        local input = torch.Tensor(
            {{4,0,1,0}, {1,0,0,0}})
    
        print(input)
        
        print(torch.renorm(input, math.huge, 1, 3))
        
        --remaxnorm(input, 1, 3)
        print(input)
    end    
end

function mytest.testSpatialStochasticPooling()
    torch.setdefaulttensortype('torch.FloatTensor')
    torch.manualSeed(1)
    
    local module = nn.SpatialStochasticPooling(2,2,2,2)
    local input = torch.Tensor({
                {{0,10,5,0}, {0,5,0,1}},
                {{1,5,0,1}, {0,0,2,0}}
            })  
    local back = torch.Tensor({
                {{1,2}},
                {{3,4}}
            })              
    
    module:training()
    print(input)
    for i=1,20 do
    print(module:forward(input))
    print(module:backward(input, back))
    end
    
    module:evaluate()
    print(module:forward(input))
    
    local module = nn.SpatialStochasticPooling(3,3,2,2)
    local input = torch.Tensor({
                {{0,5,10,0,2}, {0,0,0,0,0}, {0,1,0,1,5}},
                {{1,5,0,1,0}, {0,0,0,0,0}, {0,0,2,0,0}}
            })
    local back = torch.Tensor({
                {{1,2}},
                {{3,4}}
            })
            
    module:training()
    print(input)
    for i=1,20 do
    print(module:forward(input))
    print(module:backward(input, back))
    end
    
    module:evaluate()
    print(module:forward(input))
end





math.randomseed(os.time())
tester:add(mytest)
tester:run()
--]]