function nn.Module:__init()
   self.gradInput = torch.Tensor()
   self.output = torch.Tensor()
   self.doBackProp = true
   self.requiresGradients = true
   self.train = true
end

function nn.Module:getDisposableTensors()
	return {self.output, self.gradInput}
end

function nn.Module:clean()
   self.gradInput=self.gradInput.new()
   self.output=self.output.new()
   if self.modules then
      for i=1,#self.modules do 
         self.modules[i]:clean()
      end 
   end
   collectgarbage()
end

function nn.Module:doINeedGradientsMyself()
   if self.modules then -- a container needs gradients if one of its modules does
      local switch = false
      for i=1,#self.modules do 
         switch = switch or self.modules[i]:doINeedGradientsMyself()
      end 
      return switch      
   else
      if (self.weight or self.bias) then
         return self.train
      else
         return false
      end   
   end
end

function nn.Module:setBackProp(boolean)
   self.requiresGradients=self:doINeedGradientsMyself()
   self.doBackProp = boolean or false
   return self.requiresGradients or self.doBackProp
end

-- do I have to perform backpropagation ?
-- 
-- 

-- Sequential


function nn.Sequential:setBackProp(boolean)
   self.doBackProp=boolean or false
   self.requiresGradients=self:doINeedGradientsMyself()
   local switch=self.doBackProp
   for i=1,#self.modules do 
      switch = self.modules[i]:setBackProp(switch)
   end 
   return self.requiresGradients or self.doBackProp
end

function nn.Sequential:updateOutput(input)
   local currentOutput = input
   for i=1,#self.modules do 
      currentOutput = self.modules[i]:updateOutput(currentOutput)
      if self.modules[i+1] and self.modules[i+1].doBackProp == false and self.modules[i+1].requiresGradients == false then
         self.modules[i].output = self.modules[i].output.new()
      end
   end 
   self.output = currentOutput
   collectgarbage()
   return currentOutput
end


function nn.Sequential:updateGradInput(input, gradOutput)
   local currentGradOutput = gradOutput
   local currentModule = self.modules[#self.modules]
   for i=#self.modules-1,1,-1 do
      local previousModule = self.modules[i]
      if currentModule.doBackProp or currentModule.modules then
         currentGradOutput = currentModule:updateGradInput(previousModule.output, currentGradOutput)
      end
      currentModule = previousModule
   end
   if currentModule.doBackProp or currentModule.modules then
      currentGradOutput = currentModule:updateGradInput(input, currentGradOutput)
      self.gradInput = currentGradOutput
   end
   return self.gradInput
end


function nn.Sequential:accGradParameters(input, gradOutput, scale)
   scale = scale or 1

   local currentGradOutput = gradOutput
   local currentModule = self.modules[#self.modules]
   for i=#self.modules-1,1,-1 do
      local previousModule = self.modules[i]
      if currentModule.requiresGradients then
         currentModule:accGradParameters(previousModule.output, currentGradOutput, scale)
      end
      currentGradOutput = currentModule.gradInput
      currentModule = previousModule
   end
   
   if currentModule.requiresGradients then
      currentModule:accGradParameters(input, currentGradOutput, scale)
   end
end

