require 'nn'
require 'myutils'
require 'strict'

--------------------------------------- NilModule---------------------------------------
-- Creates a module that returns zeros or nil as output and zeros as gradInput.
-- This is useful when combined with the module ParallelTable in case one wishes to 
-- ignore a certain element in the table.


local NilModule, parent = torch.class('nn.NilModule', 'nn.Module')

function NilModule:__init(oSize)
    parent.__init(self)
    if oSize == nil then
        self.output = nil
    else    
        self.output = torch.Tensor(torch.LongStorage(oSize)):zero()
    end
    self.gradInput = nil
end

function NilModule:updateOutput(input)
    assert(input ~= nil)
        
    return self.output
end

function NilModule:updateGradInput(input, gradOutput)
    assert(input ~= nil)
    
    -- input was nothing so gradInput will be zeros in the same form and size as input
    if (self.gradInput == nil) then 
        self.gradInput = funcOnTensors(input, function (x) return x*0 end) 
    end    
        
    return self.gradInput
end

function NilModule:type(type, tensorCache)
   parent.type(self, type, tensorCache)
   self.gradInput = nil
end