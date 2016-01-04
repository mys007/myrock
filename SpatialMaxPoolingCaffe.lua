
--- LEGACY. From end of June15 implemented in standard nn.SpatialMaxPooling

myrock.SpatialMaxPoolingCaffe = nn.SpatialMaxPooling

--local ffi = require 'ffi'
--require 'cunn'

--ffi.cdef[[
--void SpatialMaxPoolingCaffe_updateOutput(THCState* state, THCudaTensor* input, 
--    THCudaTensor* output, THCudaTensor* indices, int kW, int kH, int dW, int dH, bool train);
--void SpatialMaxPoolingCaffe_updateGradInput(THCState* state, THCudaTensor* input,
--    THCudaTensor* gradInput, THCudaTensor* gradOutput, THCudaTensor* indices, int kW, int kH, int dW, int dH);
--]]

--[[local C = ffi.load(package.searchpath('libmyrock', package.cpath))

-- MaxPooling from Caffe: deterministic behavior for overlapping kernels and ceil()-output size  
local SpatialMaxPoolingCaffe, parent = torch.class('myrock.SpatialMaxPoolingCaffe', 'nn.SpatialMaxPooling')

function SpatialMaxPoolingCaffe:__init(kW, kH, dW, dH)
    parent.__init(self, kW, kH, dW, dH)
end


function SpatialMaxPoolingCaffe:updateOutput(input)
    if torch.isTypeOf(self.output, 'torch.CudaTensor') then 
        assert(torch.isTypeOf(input, 'torch.CudaTensor'))
        C.SpatialMaxPoolingCaffe_updateOutput(cutorch.getState(), input:cdata(), self.output:cdata(),
                                              self.indices:cdata(), self.kW, self.kH, self.dW, self.dH, true)    
    else
  	    local padW = math.max(0, input:size(input:dim()) - self.kW) % self.dW
        local padH = math.max(0, input:size(input:dim()-1) - self.kH) % self.dH
        if (padW>0 or padH>0) then
            if not self.padder then self.padder = nn.SpatialZeroPadding(0) end
            self.padder.pad_r = padW>0 and (self.dW - padW) or 0
            self.padder.pad_b = padH>0 and (self.dH - padH) or 0
            input.nn.SpatialMaxPooling_updateOutput(self, self.padder:updateOutput(input))
        else
            input.nn.SpatialMaxPooling_updateOutput(self, input)
        end
    end
    return self.output
end

function SpatialMaxPoolingCaffe:updateGradInput(input, gradOutput)
    if torch.isTypeOf(self.output, 'torch.CudaTensor') then 
      assert(torch.isTypeOf(input, 'torch.CudaTensor'))
      assert(torch.isTypeOf(gradOutput, 'torch.CudaTensor'))
      assert(gradOutput:nElement()==self.indices:nElement(), 'Wrong gradOutput size')
      C.SpatialMaxPoolingCaffe_updateGradInput(cutorch.getState(), input:cdata(), self.gradInput:cdata(),
      	gradOutput:cdata(), self.indices:cdata(), self.kW, self.kH, self.dW, self.dH)
    else
        local padW = math.max(0, input:size(input:dim()) - self.kW) % self.dW
        local padH = math.max(0, input:size(input:dim()-1) - self.kH) % self.dH
        if (padW>0 or padH>0) then
            if self.gradInputMy then self.gradInput = self.gradInputMy end
            input.nn.SpatialMaxPooling_updateGradInput(self, self.padder.output, gradOutput)
            self.gradInputMy = self.gradInput
            self.gradInput = self.padder:updateGradInput(input, self.gradInput)
        else
            input.nn.SpatialMaxPooling_updateGradInput(self, input, gradOutput)
        end
    end      	
    return self.gradInput
end

function SpatialMaxPoolingCaffe:type(type)
   self.indices = torch.Tensor()
   self.gradInput = torch.Tensor()
   self.output = torch.Tensor()
   self.padder = nil
   return parent.type(self,type)
end

function SpatialMaxPoolingCaffe:ceil(doCeil)
   assert(doCeil, 'floor not implemented yet')
   return self
end--]]

--------------------------------------- TEST ---------------------------------------
--[[
local mytest = {}
local OFFmytest = {}
local tester = torch.Tester()

function OFFmytest.testDeterminismccn2()
    local inp = torch.CudaTensor(32,16,18,18):zero()
    inp[1][1][3][3] = 1 --will force adding up 3 numbers
    inp = nn.Transpose({1,4},{1,3},{1,2}):cuda():forward(inp)
    
    local fw = torch.CudaTensor(32,16,8,8):fill(0)
    fw[1][1][2][1] = -0.00055786536540836
    fw[1][1][2][2] = 0.00075417151674628
    fw[1][1][1][2] = -0.00029314149287529
    fw=nn.SpatialZeroPadding(0,1,0,1):cuda():forward(fw)
    fw = nn.Transpose({1,4},{1,3},{1,2}):cuda():forward(fw)

    require 'ccn2'
    local model = ccn2.SpatialMaxPooling(3,2):cuda(); 
    model:forward(inp)
    local bw = model:backward(inp, fw):clone()
    for i=1,100 do
        local diff = bw - model:backward(inp, fw)
        print(diff:sum()) --sometimes, the input is not zero for nn.SpatialMaxPooling!
    end
end

function OFFmytest.testDeterminism()
    torch.setdefaulttensortype('torch.FloatTensor')
    torch.manualSeed(1)
    
    local inp = torch.CudaTensor(1,18,18):zero()
    inp[1][3][3] = 1 --will force adding up 3 numbers
    
    local fw = torch.CudaTensor(1,8,8):fill(0)
    fw[1][2][1] = -0.00055786536540836
    fw[1][2][2] = 0.00075417151674628
    fw[1][1][2] = -0.00029314149287529
      
    --require 'cudnn'
    --local model = cudnn.SpatialMaxPooling(3, 3, 2, 2):ceil():cuda(); fw=nn.SpatialZeroPadding(0,1,0,1):cuda():forward(fw)
    local model = myrock.SpatialMaxPoolingCaffe(3, 3, 2, 2):cuda(); fw=nn.SpatialZeroPadding(0,1,0,1):cuda():forward(fw)
    --local model = nn.SpatialMaxPooling(3, 3, 2, 2):cuda()
    model:forward(inp)
    local bw = model:backward(inp, fw):clone()
    for i=1,100 do
        local diff = bw - model:backward(inp, fw)
        print(diff:sum()) --sometimes, the input is not zero for nn.SpatialMaxPooling!
    end
end

function OFFmytest.SpatialStochasticPooling()
   local from = math.random(1,5)
   local ki = math.random(1,4)
   local kj = math.random(1,4)
   local si = math.random(1,3)
   local sj = math.random(1,3)
   local outi = math.random(4,5)
   local outj = math.random(4,5)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   local module = myrock.SpatialMaxPoolingCaffe(ki,kj,si,sj):cuda()
   local input = torch.rand(from,ini,inj):cuda()

   local err = nn.Jacobian.testJacobian(module, input, nil, nil, 1e-3)
   tester:assertlt(err, 1e6, 'error on state ')

   local ferr, berr = nn.Jacobian.testIO(module, input)
   tester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   tester:assertlt(berr, 1e-6, torch.typename(module) .. ' - i/o backward err ')

   -- batch
   local nbatch = math.random(2,5)
   input = torch.rand(nbatch,from,ini,inj):cuda()
   module = myrock.SpatialMaxPoolingCaffe(ki,kj,si,sj):cuda()

   local err = nn.Jacobian.testJacobian(module, input, nil, nil, 1e-3)
   tester:assertlt(err, 1e6, 'error on state (Batch) ')

   local ferr, berr = nn.Jacobian.testIO(module, input)
   tester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err (Batch) ')
   tester:assertlt(berr, 1e-6, torch.typename(module) .. ' - i/o backward err (Batch) ')
end

function mytest.testEquiv()
    torch.setdefaulttensortype('torch.FloatTensor')
    torch.manualSeed(1)
    cutorch.setDevice(2)
    
    for it=1,50 do
           local from = 128--math.random(1,5)
           local nbatch = 128--math.random(1,5)
           local ki = math.random(1,4)
           local kj = math.random(1,4)
           local si = math.random(1,3)
           local sj = math.random(1,3)
           local outi = math.random(4,5)
           local outj = math.random(4,5)
           local ini = (outi-1)*si+ki
           local inj = (outj-1)*sj+kj
          
            require 'cunn'
            require 'cudnn'
            local input = torch.rand(nbatch,from,inj,ini):cuda()
            local module1 = myrock.SpatialMaxPoolingCaffe(ki,kj,si,sj):cuda()
            local module2 = cudnn.SpatialMaxPooling(ki,kj,si,sj):ceil():cuda()
            local out = module1(input):clone()
            tester:assertTensorEq(out, module2(input), 1e-6)
            tester:assertTensorEq(module1(input,out+1), module2(input,out+1), 1e-6)
            collectgarbage()
     end       
end

function OFFmytest.testEquivCPU()
    torch.setdefaulttensortype('torch.FloatTensor')
    torch.manualSeed(1)
    
    for it=1,50 do
           local from = math.random(1,5)
           local nbatch = math.random(1,5)
           local ki = math.random(1,4)
           local kj = math.random(1,4)
           local si = math.random(1,3)
           local sj = math.random(1,3)
           local outi = math.random(4,5)
           local outj = math.random(4,5)
           local ini = (outi-1)*si+ki
           local inj = (outj-1)*sj+kj
          
            require 'cunn'
            local input = torch.rand(nbatch,from,inj,ini)
            local module1 = myrock.SpatialMaxPoolingCaffe(ki,kj,si,sj)
            local module2 = myrock.SpatialMaxPoolingCaffe(ki,kj,si,sj):cuda()
            local out = module1(input):clone()
            tester:assertTensorEq(out, module2(input:cuda()):float(), 1e-6)
            tester:assertTensorEq(module1(input,out), module2(input:cuda(),out:cuda()):float(), 1e-6)
            collectgarbage()
     end       
end

math.randomseed(os.time())
tester:add(mytest)
tester:run()--]]