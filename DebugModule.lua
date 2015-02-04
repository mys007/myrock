require 'nn'
require 'image'
require 'trepl'
require 'strict'
local _

--------------------------------------- DebugModule---------------------------------------
-- Plots and prints incoming fw/bw data 
 
--TODO: would be nice to add logging to a Logger (a special DebugLogging instance shared among all DebugModules). User can specify aggregation function of input/gradOutput->single number

local DebugModule, parent = torch.class('nn.DebugModule', 'nn.Module')

local help_desc = [[todo]]

function DebugModule:__init(config)
    assert(config ~= nil)
    parent.__init(self)

    _, self.name, self.plot, self.print = 
        xlua.unpack({config}, 'nn.DebugModule',  help_desc,
        {arg='name', type='string', help='User identifier of the module', req=true},
        {arg='plot', type='boolean', help='Plot yes/no', req=false, default=false},
        {arg='print', type='boolean', help='Full print yes/no', req=false, default=false}
    )
end

function DebugModule:updateOutput(input)
    assert(input ~= nil)
    
    self:displayData(input, 'FW');
    self.output = input
    --torch.save('/home/simonovm/workspace/pyrann/'..self.name..'FW', input)
    
    return self.output
end

function DebugModule:updateGradInput(input, gradOutput)
    assert(input ~= nil and gradOutput ~= nil)
    
    self:displayData(gradOutput, 'BW');
    self.gradInput = gradOutput
    --torch.save('/home/simonovm/workspace/pyrann/'..self.name..'BW', gradOutput)
        
    return self.gradInput
end

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

function DebugModule:displayData(input, fwbw)
    assert(input ~= nil and fwbw ~= nil)
    
    if (torch.isTensor(input)) then
        print('DebugModule ' .. self.name .. ': ' .. fwbw .. ' input size ' .. sizeStr(input))
        if (self.print) then
            print(input)
        end         
        if (self.plot) then
            if (input:dim()==4) then
                for i=1,input:size(1) do
                    image.display{image=input[i], legend=self.name .. ' /slice' .. i .. '_' .. fwbw}
                end
            else    
                image.display{image=input, legend=self.name .. '_' .. fwbw}
            end    
        end
    elseif (torch.type(input) == 'table') then
        print('DebugModule ' .. self.name .. ': ' .. fwbw)
        print(input)
        if (self.plot) then
            for i=1,#input do
                if (input[i]:dim()>0) then
                    image.display{image=input[i], legend=self.name .. '_' .. fwbw .. ' (' .. i .. ')'}
                end
            end
        end        
    else
        print('DebugModule ' .. self.name .. ': ' .. fwbw .. 'unknown type ' .. torch.type(input))
    end
    
    return input
end
