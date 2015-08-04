require 'nn'
require 'strict'

--------------------------------------- CudaAdapter---------------------------------------
-- Enables to run CPU-only modules in CUDA models and vice versa. 
-- 
-- CudaAdapter keeps its output and gradInput tensors (which are public) always in the model's type.
-- If the type of the adaptee's tensors differ, a conversion is performed. Note that the conversion
-- is of course expensive and causes a major delay (although implemented efficiently).

local CudaAdapter, parent = torch.class('myrock.CudaAdapter', 'nn.Container')

function CudaAdapter:__init(module)
    assert(module ~= nil)
    parent.__init(self)
    self.modules[1] = module
    self.inputF = torch.Tensor()
    self.gradOutputF = torch.Tensor()
end

function CudaAdapter:updateOutput(input)
    if input:type()==self.modules[1].output:type() then
        self.output = self.modules[1]:updateOutput(input)
    else
        self.inputF = self.inputF:typeAs(self.modules[1].output)
        self:convert(self.inputF, input)    
        self:convert(self.output, self.modules[1]:updateOutput(self.inputF))
    end
    return self.output
end

function CudaAdapter:updateGradInput(input, gradOutput)
    if input:type()==self.modules[1].output:type() then
        self.gradInput = self.modules[1]:updateGradInput(input, gradOutput)
    else
        self.gradOutputF = self.gradOutputF:typeAs(self.modules[1].output)
        self:convert(self.gradOutputF, gradOutput)    
        self:convert(self.gradInput, self.modules[1]:updateGradInput(self.inputF, self.gradOutputF))
    end
    return self.gradInput
end

function CudaAdapter:convert(out, x)
    assert(out~=nil and x ~= nil)
    
    if (torch.isTensor(x)) then
        out:resize(x:size()):copy(x)
    elseif (torch.type(x) == 'table') then
        for k,v in pairs(x) do 
            if (out[k]~=nil) then out[k]:resize(x[k]:size()):copy(x[k]) end
        end
    else
        error('CudaAdapter: unknown type ' .. torch.type(x))
    end 
end

function CudaAdapter:type(type)
    self.output = torch.Tensor():type(type)
    self.gradInput = torch.Tensor():type(type)
    return self --the internal module has been spared of type conversion
end

function CudaAdapter:__tostring__()
   return 'myrock.CudaAdapter' .. ' {' .. tostring(self.modules[1]) .. '}'
end