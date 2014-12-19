local function deepcopy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in next, orig, nil do
            copy[deepcopy(orig_key)] = deepcopy(orig_value)
        end
        setmetatable(copy, deepcopy(getmetatable(orig)))
    else -- number, string, boolean, etc
        copy = orig
    end
    return copy
end

function nn.Module:weightOptimSet(optimizer, config)
   if self.weight then
   	self.weightOptimizer=optimizer
	   self.weightOptimConfig=deepcopy(config)
   end
end

function nn.Module:biasOptimSet(optimizer, config)
   if self.bias then
	   self.biasOptimizer=optimizer
   	self.biasOptimConfig=deepcopy(config)
   end
end

function nn.Module:optimSet(optimizer, config)
	self:weightOptimSet(optimizer, config)
	self:biasOptimSet(optimizer, config)
end

function nn.Module:optimClear()
   self.weightOptimizer=nil
   self.biasOptimizer=nil
end

function nn.Module:scaleLearningRates(scale)
   if self.modules then
      for i,module in ipairs(self.modules) do
         module:scaleLearningRates(scale)
      end
   end
   if self.weightOptimizer then
      self.weightOptimConfig.learningRate = self.weightOptimConfig.learningRate * scale
   end
   if self.biasOptimizer then
      self.biasOptimConfig.learningRate = self.biasOptimConfig.learningRate * scale
   end
end

function nn.Module:optimStep()
   -- propagate inside
   if self.modules then
      for i,module in ipairs(self.modules) do
         module:optimStep()
      end
   end

   -- this function assumes you have computed gradWeight and gradBias already
   if self.weightOptimizer then
      local function fevalW(x)
         return nil, self.gradWeight
      end
      self.weightOptimizer(fevalW, self.weight, self.weightOptimConfig)
   end

   if self.biasOptimizer then
      local function fevalB(x)
         return nil, self.gradBias
      end
      self.biasOptimizer(fevalB, self.bias, self.biasOptimConfig)
   end
end
