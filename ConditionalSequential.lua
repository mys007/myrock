require 'nn'
require 'myutils'
require 'strict'

--------------------------------------- ConditionalSequential ---------------------------------------
-- Behaves like nn.Sequential if 'inputCheckFunc(input)' is fulfiled. 
-- Otherwise the sequence is skipped and the module outputs a Tensor() in fw- and zeros in bw-pass.

local ConditionalSequential, ConditionalSequential_parent = torch.class('nn.ConditionalSequential', 'nn.Sequential')

function ConditionalSequential:__init(inputCheckFunc, emptyOutput)
    assert(inputCheckFunc ~= nil)
    
    ConditionalSequential_parent.__init(self)
    self.inputCheckFunc = inputCheckFunc
    self.emptyOutput = emptyOutput
    self.checkResult = true
end

function ConditionalSequential:updateOutput(input)
    self.checkResult = self.inputCheckFunc(input) 
    if (self.checkResult) then
        return ConditionalSequential_parent.updateOutput(self, input)
    else
        self.output = self.emptyOutput
        return self.output
    end
end 

function ConditionalSequential:updateGradInput(input, gradOutput)
    if (self.checkResult) then
        return ConditionalSequential_parent.updateGradInput(self, input, gradOutput)
    else
        -- input was ignored so gradInput will be zeros in the same form and size as input
        self.gradInput = funcOnTensors(input, function (x) return x*0 end)
        return self.gradInput 
    end    
end        
    
function ConditionalSequential:accGradParameters(input, gradOutput, scale)
    if (self.checkResult) then
        return ConditionalSequential_parent.accGradParameters(self, input, gradOutput, scale)
    end    
end

function ConditionalSequential:accUpdateGradParameters(input, gradOutput, lr)
    if (self.checkResult) then
        return ConditionalSequential_parent.accUpdateGradParameters(self, input, gradOutput, lr)
    end   
end

function ConditionalSequential:__tostring__()
    return ConditionalSequential_parent.__tostring__(self) .. '/Cond'
end











--------------------------------------- TEST ---------------------------------------

--[[local mytest = {}
local tester = torch.Tester()

function mytest.testCM()
    torch.setdefaulttensortype('torch.FloatTensor')
    torch.manualSeed(1)
    
    local mod = nn.SpatialMaxPooling(2, 2, 2, 2)
    local modExp = nn.SpatialMaxPooling(2, 2, 2, 2)
    
    local subject = nn.ConditionalSequential(function (x) return x[1][1][1]>1 end, torch.Tensor())
    subject:add(mod)
    
    local inputT = torch.Tensor({{{2,1,0,0}, {0,0,0,3}}})
    local gradOutput = torch.Tensor({{{10,15}}})
    local inputF = torch.Tensor({{{1,0,0,0}, {0,0,0,3}}})
    local expectedF = torch.Tensor()
    
    tester:assertTensorEq(subject:forward(inputT), modExp:forward(inputT), 1e-6)
    tester:assertTensorEq(subject:backward(inputT, gradOutput), modExp:backward(inputT, gradOutput), 1e-6)

    tester:assert(subject:forward(inputF):nElement() == 0, 1e-6)
    tester:assert(torch.sum(subject:backward(inputF, gradOutput)) == 0, 1e-6)
end



tester:add(mytest)
tester:run()--]]
