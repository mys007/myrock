require 'nn'
require 'myrock'
require 'DebugModule'
require 'strict'

--------------------------------------- SpatialScaling---------------------------------------
-- Upscales image by float scale by bilinear interpolation. This is implemented as 2x matrix 
-- multiplication. The weights are precomputed, so the process is nicely inversible.
-- When downsampling, the gradient is bilin interp. When upsampling, the input is bilin interp.
-- For upsampling, this should show equivalent behaviour to nnx.SpatialUpSampling().

-- TODO: full CUDA implementation (not matrix mult) ... because this is still very slow!
 
-- TODO: better deal with varying input sizes: have big WC, WR and just take submatrices if the scaling ratio stays the same??? 

local SpatialScaling, parent = torch.class('nn.SpatialScaling', 'nn.Module')
   
function SpatialScaling:__init(outW, outH)
    parent.__init(self)
    
    if (outW~=nil and outH~=nil) then
        self.outW = outW
        self.outH = outH
    elseif (outW~=nil) then
        self.getSizeFunc = outW
    else
        assert(false,'wrong arguments')
    end    
end

function SpatialScaling:createWeights(inW, inH, outW, outH)
    assert(inW ~= nil and inH ~= nil and outW ~= nil and outH ~= nil)
    
    -- upsampling -> use bilinear interp in the fw-pass
    -- downsampling -> use bilinear interp in the bw-pass   (in fw-pass it would be just subsampling without pre-bluring)
        
    if (outH>=inH) then
        self.WC = self:createBlerpWeights(outH, inH)
    else    
        self.WC = self:createBlerpWeights(inH, outH):transpose(2,3)
    end        
    
    if (outW==outH and inW==inH) then
        self.WR = self.WC:transpose(2,3)    --same weights so shared mem
    elseif (outW>=inW) then 
        self.WR = self:createBlerpWeights(outW, inW):transpose(2,3)
    else    
        self.WR = self:createBlerpWeights(inW, outW)
    end    

end

function SpatialScaling:createBlerpWeights(d, s)
    local W = torch.Tensor(1, d, s):zero()
    if (s==1) then W:fill(1); return W end
    
    --from image.scale
    local scale = (s-1) / (d-1)    
    for di = 0, d-2 do
        local si_f = di * scale; 
        local si_i = math.floor(si_f);
        si_f = si_f - si_i;
        
        W[1][di+1][si_i+1] = 1 - si_f;
        W[1][di+1][si_i+2] = si_f;
    end
    W[1][d][s] = 1;

    return W
end

function SpatialScaling:updateOutput(input)
    assert(input ~= nil)
    
    if self.getSizeFunc then 
        self.outW, self.outH = self.getSizeFunc()
    end    
    
    -- batchmode to 3D (because of bmm)
    local bs = input:dim()==4 and input:size(1) or 0
    if bs>0 then
        input = input:view(-1,input:size(3), input:size(4))
        if self.output:dim()==4 then self.output  = self.output:view(-1, self.outH, self.outW) end
    end
 
    if input:dim() ~= self.output:dim() or self.output:size(1)~=input:size(1) or self.output:size(2)~=self.outH or self.output:size(3)~=self.outW or 
       self.WC==nil or input:size(2)~=self.WC:size(3) or input:size(3)~=self.WR:size(2) then
        
        self:createWeights(input:size(3), input:size(2), self.outW, self.outH)
        self.WC = self.WC:expand(input:size(1), self.WC:size(2), self.WC:size(3)):typeAs(input)
        self.WR = self.WR:expand(input:size(1), self.WR:size(2), self.WR:size(3)):typeAs(input) 
    
        self.output:resize(input:size(1), self.outH, self.outW)
        self.tmpFW = torch.Tensor(input:size(1), self.WC:size(2), input:size(3)):typeAs(input)
        assert(input:size(2)==self.WC:size(3) and input:size(3)==self.WR:size(2))
    end
    
    self.tmpFW:bmm(self.WC, input)
    self.output:bmm(self.tmpFW, self.WR)
    
    if bs>0 then
        self.output = self.output:view(bs, -1, self.outH, self.outW)
    end
    
    return self.output
end

function SpatialScaling:updateGradInput(input, gradOutput)
    assert(input ~= nil and gradOutput ~= nil)
    
    -- batchmode to 3D (because of bmm)
    local bs = input:dim()==4 and input:size(1) or 0
    if bs>0 then
        input = input:view(-1,input:size(3), input:size(4))
        if not gradOutput:isContiguous() then --e.g. from concat
            self._gradOutput = self._gradOutput or gradOutput.new()
            self._gradOutput:resizeAs(gradOutput):copy(gradOutput)
            gradOutput = self._gradOutput
        end        
        gradOutput = gradOutput:view(-1,gradOutput:size(3), gradOutput:size(4))
        if self.gradInput:dim()==4 then self.gradInput = self.gradInput:view(-1, self.gradInput:size(3), self.gradInput:size(4)) end
    end
    
    if not input:isSameSizeAs(self.gradInput) then
        self.gradInput:resizeAs(input)
        self.tmpBW = torch.Tensor(gradOutput:size(1), self.WC:size(3), gradOutput:size(3)):typeAs(input)
    end
    
    self.tmpBW:bmm(self.WC:transpose(2,3), gradOutput)
    self.gradInput:bmm(self.tmpBW, self.WR:transpose(2,3))
    
    if bs>0 then
        self.gradInput = self.gradInput:view(bs, -1, self.gradInput:size(2), self.gradInput:size(3))
    end
    
    return self.gradInput
end


--------------------------------------- Wrong Impl ---------------------------------------
-- Relying on image.scale doesn't work because the forward value distribution is bilinear
-- but the backward one is just unweighted averaging -> the process is not reversible

--[[function SpatialScaling:__init(outW, outH)
    assert(outW ~= nil and outH ~= nil)
    
    parent.__init(self)
    self.outW = outW
    self.outH = outH
end

function SpatialScaling:updateOutput(input)
    assert(input ~= nil)
    
    --Input gets copied
    self.output = image.scale(input, self.outW, self.outH)
    return self.output
end

function SpatialScaling:updateGradInput(input, gradOutput)
    assert(input ~= nil and gradOutput ~= nil)
    
    --Gradients gets summed/distributed (but the magnitude remains)
    local inW = input:size(input:dim())
    local inH = input:size(input:dim()-1)
    local sumFactor = (self.outW/inW) * (self.outH/inH) 
    --TODO: this is broken!
    self.gradInput = image.scale(gradOutput, inW, inH) * sumFactor
    return self.gradInput
end--]]


--------------------------------------- TEST ---------------------------------------

--[[
require 'image'

local mytest = {}
local OFFmytest = {}
local tester = torch.Tester()


function OFFmytest.testToy2()
    local inp = torch.Tensor(1,1,2); inp[1][1][1]=1; inp[1][1][2]=0;
    local err = torch.Tensor(1,1,5); err[1][1][1]=0; err[1][1][2]=0; err[1][1][3]=0; err[1][1][4]=20; err[1][1][5]=0;
    local expD = torch.Tensor(1,1,2); expD[1][1][1]=5; expD[1][1][2]=15;
    
    local supMy = nn.SpatialScaling(function () return 5,1 end)
  
    local oMy = supMy:forward(inp)
    local dMy = supMy:backward(inp, err)
    
    tester:assertlt(torch.norm(dMy-expD), 1e-5)
end

function OFFmytest.testBlerp()
    local inp = torch.rand(1,20,30)
    local expO = image.scale(inp, 256, 128)
  
    local supMy = nn.SpatialScaling(function () return 256,128 end)
  
    local oMy = supMy:forward(inp)
  
    tester:assertlt(torch.norm(expO-oMy), 1e-4)
end

function OFFmytest.testUpscaleFloat()
    local inp = torch.rand(1,4,4)
    local err = torch.rand(1,5,5)
  
    local supMy = nn.SpatialScaling(function () return 5,5 end)
  
    local oMy = supMy:forward(inp)
    local dMy = supMy:backward(inp, err)

    tester:assertlt(torch.norm(err,1) - torch.norm(dMy,1), 1e-5)
end

function OFFmytest.testBatch()
    local inp = torch.rand(2,5,4,4)
    local err = torch.rand(2,5,5,5)
  
    local supMy = nn.SpatialScaling(function () return 5,5 end)
  
    local oMy = supMy:forward(inp):clone()
    local dMy = supMy:backward(inp, err):clone()

    for i=1,2 do
        tester:assertTensorEq(oMy[i], supMy:forward(inp[i]), 1e-5)
        tester:assertTensorEq(dMy[i], supMy:backward(inp[i], err[i]), 1e-5)
    end
end

    
function OFFmytest.testScaleInteractive()
    local img = image.lena()
    img = img:narrow(2, 1, 400)
    
    local model = nn.Sequential()
    model:add(myrock.DebugModule{name="befo", plot=true})
    model:add(nn.SpatialScaling(function () return 200,150 end))
    model:add(myrock.DebugModule{name="after", plot=true})   
    model:training()           
    local res = model:forward(img)
    model:backward(img, res*1.2)    
end

function OFFmytest.testJitteringModuleScale()

    local ini = math.random(3,5)
    local inj = math.random(3,5)
    local ink = math.random(3,5)
    local outj = math.random(2,4)
    local outk = math.random(6,10)    
    local input = torch.Tensor(ini,inj,ink):normal(0, 5)
    local module = nn.SpatialScaling(function () return outj, outk end)
    module:training()      
    local err = nn.Jacobian.testJacobian(module,input)      --needs to enable non-radnomized code for this
    tester:assertlt(err,1e-5, 'error on state ')
    local ferr,berr = nn.Jacobian.testIO(module,input)
    tester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
    tester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end    
    
    
    
math.randomseed(os.time())
tester:add(mytest)
tester:run()
--]]
