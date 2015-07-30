require 'torch'
require 'strict'

--------------------------------------- CmdLine---------------------------------------
-- Modifies handling of bool arguments, they have to be explicitely specified.
-- Possible alternative: https://github.com/davidm/lua-pythonic-optparse/blob/master/lmod/pythonic/optparse.lua

local CmdLine, parent = torch.class('myrock.CmdLine', 'torch.CmdLine')

local function strip(str)
   return string.match(str, '%-*(.*)')
end

function CmdLine:__readOption__(params, arg, i)
   local key = arg[i]
   local option = self.options[key]
   if not option then
      self:error('unknown option ' .. key)
   end

   if not self.boolsfull and option.type and option.type == 'boolean' then
      params[strip(key)] = not option.default
      return 1
   else
      local value = arg[i+1]
      if not value then
         self:error('missing argument for option ' .. key)
      end
      if not option.type or option.type == 'string' then
         --
      elseif option.type == 'number' then
         value = tonumber(value)
      elseif option.type == 'boolean' then
         value = (string.lower(value) == 'true')
      else
         self:error('unknown required option type ' .. option.type)
      end
      if value==nil then
         self:error('invalid type for option ' .. key .. ' (should be ' .. option.type .. ')')
      end
      params[strip(key)] = value
      return 2
   end
end

function CmdLine:__init(argseparator_,keyseparator_)
   parent.__init(self,argseparator_,keyseparator_)
   self.boolsfull = true
end