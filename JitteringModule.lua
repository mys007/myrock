require 'nn'
require 'nnx'
require 'DebugModule'
require 'image'
require 'myutils'
require 'strict'

--------------------------------------- JitteringModuleGNoise ---------------------------------------

local JitteringModuleGNoise, JitteringModuleGNoise_parent = torch.class('nn.JitteringModuleGNoise', 'nn.Module')

function JitteringModuleGNoise:__init(factor)
    assert(factor~=nil)
    JitteringModuleGNoise_parent.__init(self)
    self.factor = factor
end

function JitteringModuleGNoise:updateOutput(input)
    if (not self.output or not self.output:isSameSizeAs(input)) then
        self.output = input:clone()
    end

    if self.train then
        --for unit test
        --torch.manualSeed(1)
        --self.output:normal(0, 10)
        
        self.output:normal(0, math.max(torch.std(input) * self.factor, 1e-7))   --TODO: maybe per-element value instead of std?
        self.output:add(input)
    else
        self.output:copy(input) --TODO: should boost sth like dropout does?
    end    
    return self.output
end

function JitteringModuleGNoise:updateGradInput(input, gradOutput)
    assert(input ~= nil and gradOutput ~= nil and self.train)
    self.gradInput = gradOutput -- derivative of additive noise = 0
    return self.gradInput
end



--------------------------------------- JitteringModuleScale ---------------------------------------

local JitteringModuleScale, JitteringModuleScale_parent = torch.class('nn.JitteringModuleScale', 'nn.Module')

function JitteringModuleScale:__init(fmin,fmax,fixAspectRatio,rndCrop)
    assert(fmax>=1 and fmin<=1 and fixAspectRatio~=nil and rndCrop~=nil)
    
    JitteringModuleScale_parent.__init(self)
    self.fmin = fmin
    self.fmax = fmax
    self.fixAspectRatio = fixAspectRatio
    self.rndCrop = rndCrop
    self.resample = nn.SpatialReSampling{rwidth=1, rheight=1}       --TODO: replace with SpatialScaling (makes a difference when downsampling in deeper layers)
end

function JitteringModuleScale:updateOutput(input)
    assert(input ~= nil)
    if (self.output:dim()==0) then
        self.output = self.output:resizeAs(input)
    end      
        
    if self.train then    
        self.output:fill(0)
        
        self.resample.rwidth = math.max(torch.uniform(self.fmin,self.fmax), 1/input:size(3))
        self.resample.rheight = math.max(torch.uniform(self.fmin,self.fmax), 1/input:size(2))
        if self.fixAspectRatio then self.resample.rheight = self.resample.rwidth end
        --if (torch.bernoulli(0.5)==1) then self.resample.rheight = 1; self.resample.rwidth = 1; end
        
        local resc = self.resample:updateOutput(input)
        local dx = resc:size(3) - input:size(3)
        local dy = resc:size(2) - input:size(2)
        
        self.dx2 = self.rndCrop and torch.uniform(0,dx) or math.ceil(dx/2) 
        self.dy2 = self.rndCrop and torch.uniform(0,dy) or math.ceil(dy/2)        
        
        if (dx>=0 and dy>=0) then --crop w,h
            local r = resc:narrow(3, 1+self.dx2, input:size(3)):narrow(2, 1+self.dy2, input:size(2))
            self.output:copy(r)
        elseif (dx>=0 and dy<0) then --crop w, pad h
            local r = resc:narrow(3, 1+self.dx2, input:size(3))
            self.output:narrow(2, 1-self.dy2, r:size(2)):copy(r)
        elseif (dx<0 and dy>=0) then --crop h, pad w
            local r = resc:narrow(2, 1+self.dy2, input:size(2))
            self.output:narrow(3, 1-self.dx2, r:size(3)):copy(r)        
        else --pad w,h
            self.output:narrow(3, 1-self.dx2, resc:size(3)):narrow(2, 1-self.dy2, resc:size(2)):copy(resc)        
        end
        
        self.dx = dx
        self.dy = dy
    else
        self.output:copy(input) --TODO: should boost sth like dropout does? (e.g. count the freq of used/blanked pixels at train time, multiply at test tim)
    end            
    return self.output
end

function JitteringModuleScale:updateGradInput(input, gradOutput)
    assert(input ~= nil and gradOutput ~= nil and self.train)
    
    local dx = self.dx
    local dy = self.dy
    local resc = torch.Tensor(gradOutput:size(1), gradOutput:size(2)+dy, gradOutput:size(3)+dx):zero()
    
    if (dx>=0 and dy>=0) then --crop w,h -> pad it now
        resc:narrow(3, 1+self.dx2, gradOutput:size(3)):narrow(2, 1+self.dy2, gradOutput:size(2)):copy(gradOutput)
    elseif (dx>=0 and dy<0) then --crop w, pad h -> crow h, pad w
        local r = gradOutput:narrow(2, 1-self.dy2, resc:size(2))
        resc:narrow(3, 1+self.dx2, gradOutput:size(3)):copy(r)     
    elseif (dx<0 and dy>=0) then --crop h, pad w -> crop w, pad h
        local r = gradOutput:narrow(3, 1-self.dx2, resc:size(3))
        resc:narrow(2, 1+self.dy2, gradOutput:size(2)):copy(r)    
    else --pad w,h -> crop it now
        local r = gradOutput:narrow(3, 1-self.dx2, resc:size(3)):narrow(2, 1-self.dy2, resc:size(2))
        resc:copy(r)
    end
    
    self.gradInput = self.resample:updateGradInput(input, resc)
    return self.gradInput
end


--------------------------------------- JitteringModuleTranslate ---------------------------------------

local JitteringModuleTranslate, JitteringModuleTranslate_parent = torch.class('nn.JitteringModuleTranslate', 'nn.Module')

function JitteringModuleTranslate:__init(tmax)
    assert(tmax~=nil)
    
    JitteringModuleTranslate_parent.__init(self)
    self.tmax = tmax
    self.renorm = false
end

function JitteringModuleTranslate:updateOutput(input)
    assert(input ~= nil)
    if (self.output:dim()==0) then
        self.output = self.output:resizeAs(input)
    end      
    
    if self.train then    
        self.output:fill(0)
        
        --for unit test
        --torch.manualSeed(1)   
        
        self.dx = torch.uniform(-self.tmax,self.tmax)
        self.dy = torch.uniform(-self.tmax,self.tmax)    
        image.translate(self.output, input, self.dx, self.dy)
        
        if self.renorm then
            self.signalloss = 1 - (math.abs(self.dx)*input:size(2) + math.abs(self.dy)*input:size(3) - math.abs(self.dx)*math.abs(self.dy)) / (input:size(3)*input:size(2))
            --local signalloss2 = torch.norm(self.output,1) / (torch.norm(input,1) + 1e-8)
            self.output =  self.output / self.signalloss --boost output by the amount of blacked-out features (~inverse dropout)
        end
    else
        self.output:copy(input)
    end            
    return self.output
end

function JitteringModuleTranslate:updateGradInput(input, gradOutput)
    assert(input ~= nil and gradOutput ~= nil and self.train)
    if (self.gradInput:dim()==0) then
        self.gradInput = self.gradInput:resizeAs(gradOutput)
    end      
    self.gradInput:fill(0)    
    
    image.translate(self.gradInput, gradOutput, -self.dx, -self.dy)
    if self.renorm then
        self.gradInput = self.gradInput / self.signalloss
    end
    return self.gradInput
end










--------------------------------------- TEST ---------------------------------------

--[[
local mytest = {}
local OFFmytest = {}
local tester = torch.Tester()


function OFFmytest.testJitteringModuleGNoiseInteractive()
    local img = image.lena()
    img = img:narrow(2, 1, 400)
    
    local model = nn.Sequential()
    model:add(nn.DebugModule{name="befo", plot=true})
    model:add(nn.JitteringModuleGNoise(2))
    model:add(nn.DebugModule{name="after", plot=true})        
    local res = model:forward(img)    
end

function OFFmytest.testJitteringModuleGNoise()

    local ini = math.random(3,5)
    local inj = math.random(3,5)
    local ink = math.random(3,5)
    local input = torch.Tensor(ini,inj,ink):fill(1)
    local module = nn.JitteringModuleGNoise(5)
    module:training() 
    local err = nn.Jacobian.testJacobian(module,input)      --needs to enable non-radnomized code for this
    tester:assertlt(err,1e-5, 'error on state ')
    local ferr,berr = nn.Jacobian.testIO(module,input)
    tester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
    tester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function OFFmytest.testJitteringModuleScaleInteractive()
    local img = image.lena()
    img = img:narrow(2, 1, 400)
    
    local model = nn.Sequential()
    model:add(nn.DebugModule{name="befo", plot=true})
    --model:add(nn.JitteringModuleScale(0.5, 1, true, true))
    model:add(nn.JitteringModuleScale(1, 1.5, true, true))
    model:add(nn.DebugModule{name="after", plot=true})   
    model:training()           
    local res = model:forward(img)
    model:backward(img, res*1.2)    
end

function OFFmytest.testJitteringModuleScale()

    local ini = math.random(3,5)
    local inj = math.random(3,5)
    local ink = math.random(3,5)
    local docrop = math.random(1,2)
    local input = torch.Tensor(ini,inj,ink):normal(0, 5)
    local module = nn.JitteringModuleScale(1/1.5, 1.5, false, docrop==1)
    module:training()      
    local err = nn.Jacobian.testJacobian(module,input)      --needs to enable non-radnomized code for this
    tester:assertlt(err,1e-5, 'error on state ')
    local ferr,berr = nn.Jacobian.testIO(module,input)
    tester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
    tester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function OFFmytest.testJitteringModuleTranslateInteractive()
    local img = image.lena()
    img = img:narrow(2, 1, 400)
    
    local model = nn.Sequential()    
    model:add(nn.DebugModule{name="befo", plot=true})
    model:add(nn.JitteringModuleTranslate(50))
    model:add(nn.DebugModule{name="after", plot=true})      
    model:training()      
    local res = model:forward(img)
    model:backward(img, res*1.2)    
end

function OFFmytest.testJitteringModuleTranslate()

    local ini = math.random(3,5)
    local inj = math.random(3,5)
    local ink = math.random(3,5)
    local input = torch.Tensor(ini,inj,ink):normal(0, 5)
    local module = nn.JitteringModuleTranslate(2)
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
