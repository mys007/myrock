require 'nn'

-----------
-- Weights backpropagation of different samples in batch by specified factors.
local SampleWeighter, SampleWeighter_parent = torch.class('myrock.SampleWeighter', 'nn.Module')

function SampleWeighter:__init()
    SampleWeighter_parent.__init(self)
    self.factors = torch.Tensor() --set externally
end

function SampleWeighter:updateOutput(input)
   self.output = input
   return self.output
end

function SampleWeighter:updateGradInput(input, gradOutput)
    if self.train then
        local factors = gradOutput:dim()==2 and self.factors:view(-1,1) or self.factors
        local L1orig = gradOutput:norm(1) + 1e-6
        gradOutput:cmul(factors:expandAs(gradOutput))
        local L1new = gradOutput:norm(1) + 1e-6
        gradOutput:mul(L1orig/L1new) --keep the L1-norm of gradient ("its energy") for consistency
        self.gradInput = gradOutput
    end
    self.gradInput = gradOutput
    return self.gradInput
end

function SampleWeighter:setFactors(factors)
    self.factors:resize(factors:size()):copy(factors)
end


-----TEST
--[[
local mytest = {}
local OFFmytest = {}
local tester = torch.Tester()

function mytest.test1()

   local input = torch.Tensor{1,2,3,0,4,0}
   local output = torch.Tensor{10,20,30,0,40,0}

   local module = myrock.SampleWeighter()
   module:training()
   module:setFactors(torch.Tensor{0,0.1,0.1,0,1,0})

   module:forward(input)
   module:backward(input, output)

   print(module.gradInput)
end
   
tester:add(mytest) 
tester:run()   --]]