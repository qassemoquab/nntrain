require 'optim'

local NeuralNet = torch.class('nn.NeuralNet2')

-- init 

function NeuralNet:__init()
	self.epochIdx=0

	self.trainLogger=optim.Logger('/tmp/trainLog.log')
	self.testLogger=optim.Logger('/tmp/testLog.log')
   self.lastTestCost=0
end

-- call as model:setDiagFiles('model.t7', 'log_train.txt', 'log_val.txt') before training AND before resuming
function NeuralNet:setDiagFiles(checkpointPath, trainLogPath, testLogPath)
	self.checkpointPath = checkpointPath
	self.trainLogger = optim.Logger(trainLogPath)
	self.testLogger = optim.Logger(testLogPath)
end

function NeuralNet:setNetwork(net)
	self.network = net
end

function NeuralNet:setCriterion(criterion)
	self.criterion=criterion
	self.criterion.sizeAverage=false
end




-- data

function NeuralNet:setDataset(dataset)
	self.dataset=dataset
	self.batchSize=self:getDatasetBatchSize()
end

function NeuralNet:getDatasetBatchSize()
	if self.dataset.getBatchSize then 
	   return self.dataset:getBatchSize()
	else
	   return self.dataset.batchSize
	end
end

function NeuralNet:getDatasetNumSamples()
	if self.dataset.getNumSamples then 
	   return self.dataset:getNumSamples()
	else
	   return self.dataset.numSamples
	end
end

function NeuralNet:setBatchSize(batchSize)
   self.batchSize=batchSize
end

function NeuralNet:getBatchSize()
   return self.batchSize
end

function NeuralNet:getNumBatches()
   return math.ceil(self:getDatasetNumSamples() / self:getBatchSize())
end

function NeuralNet:setTrainSetRange(first, last)
	assert(first >= 1 and last >= first and last <= self:getNumBatches(), ' ... ')
	self.trainset={first, last}
	self.trainsetsize=last-first+1
end

function NeuralNet:setTestSetRange(first, last)
	assert(first >= 1 and last >= first and last <= self:getNumBatches(), ' ... ')
	self.testset={first, last}
end

function NeuralNet:shuffleTrainSet()
	self.batchshuffle=torch.randperm(self.trainsetsize)
end

function NeuralNet:getBatchNum(idx)
	return self.trainset[1]+self.batchshuffle[idx]-1
end

function NeuralNet:getBatch(batchidx, test)
	return self:getBatchOfSize(batchidx, self.batchSize)
end

function NeuralNet:getBatchOfSize(batchidx, requestedSize)
   local batchsize = self:getDatasetBatchSize()
   local sample_start = (batchidx-1)*requestedSize+1
   local sample_end = math.min((batchidx)*requestedSize, self:getDatasetNumSamples())
   
   local firstbatch = math.ceil(sample_start/batchsize)
   local firstsampleinfirstbatch = sample_start % batchsize
   if firstsampleinfirstbatch==0 then 
      firstsampleinfirstbatch=batchsize 
   end
   local numexamplesfromfirstbatch = batchsize-firstsampleinfirstbatch+1
   
   local lastbatch  = math.ceil(sample_end/batchsize)
   local lastsampleinlastbatch = sample_end % batchsize
   if lastsampleinlastbatch==0 then 
      lastsampleinlastbatch=batchsize 
   end
   local numexamplesfromlastbatch = lastsampleinlastbatch
   
   local b,t = self.dataset:getBatch(firstbatch)
   
   local out_b
   local out_t
   
   if firstbatch==lastbatch then
      out_b = b:narrow(1, firstsampleinfirstbatch, sample_end - sample_start + 1)
      out_t = t:narrow(1, firstsampleinfirstbatch, sample_end - sample_start + 1)
   else
      local dimsBatch = b:size()
      local dimsTarget = t:size()
   
      dimsBatch[1] = sample_end - sample_start + 1
      dimsTarget[1] = sample_end - sample_start + 1
   
      out_b = b.new(dimsBatch)
      out_t = t.new(dimsTarget)
      
      -- first batch 
      out_b:narrow(1,1,numexamplesfromfirstbatch):copy(b:narrow(1,firstsampleinfirstbatch,numexamplesfromfirstbatch))
      out_t:narrow(1,1,numexamplesfromfirstbatch):copy(t:narrow(1,firstsampleinfirstbatch,numexamplesfromfirstbatch))
      
      -- intermediate batches
      local count = 0
      for currentbatch = firstbatch+1, lastbatch-1 do
         b, t = self.dataset:getBatch(currentbatch)
         out_b:narrow(1,numexamplesfromfirstbatch+1+count*batchsize,batchsize):copy(b)
         out_t:narrow(1,numexamplesfromfirstbatch+1+count*batchsize,batchsize):copy(t)
         count=count+1
      end
      
      -- last batch
      b,t = self.dataset:getBatch(lastbatch)
      out_b:narrow(1,numexamplesfromfirstbatch+1+count*batchsize,numexamplesfromlastbatch):copy(b:narrow(1,1,numexamplesfromlastbatch))
      out_t:narrow(1,numexamplesfromfirstbatch+1+count*batchsize,numexamplesfromlastbatch):copy(t:narrow(1,1,numexamplesfromlastbatch))
      
   end

	return out_b, out_t
end






-- logging

function NeuralNet:setNumClasses(nclasses)
   self.nclasses=nclasses
   self.confusion=optim.ConfusionMatrix(nclasses)
end

function NeuralNet:updateConfusion(target)
   if self.confusion then
      if target:size(1) ~= self.network.output:size(1) then 
         error('network output and target sizes are inconsistent') 
      end
      if self.network.output:dim()==2 then
         for k=1,target:size(1) do
            self.confusion:add(self.network.output[{k,{}}], target[{k}])
         end
      end
   end
end

function NeuralNet:updateTrainLogger(cost, valid)
   local lastTestCost = self.lastTestCost or 0
   self.trainLogger:add{['cost'] = cost, ['validation cost'] = lastTestCost}
end

function NeuralNet:plotTrainLogger(cost, valid)
   self.trainLogger:style{['cost']='-', ['validation cost'] = '-'}
   self.trainLogger:plot()
end





-- testing / validation

function NeuralNet:test(batchstart, batchend)
   self.network:evaluate()

   local batchstart = batchstart or self.testset[1]
   local batchend = batchend or self.testset[2]
	local timer=torch.Timer()
	
	local meancost=0
	local numexamples=0
	if self.confusion then
	   self.confusion:zero()
   end
	-- run on validation set :
	for batchIdx=batchstart,batchend do
	   local input, target = self:getBatch(batchIdx)
		local output = self.network:forward(input)
		local cost = self.criterion:forward(output, target)
		meancost=meancost+cost
		numexamples=numexamples+input:size(1)
		if self.confusion then
		   self:updateConfusion(target)
      end
	end
	meancost=meancost/numexamples
	
	local avgvalid
   if self.confusion then 
		self.confusion:updateValids() 
		avgvalid = self.confusion.averageValid
      self.confusion:zero()
	end

	self:printTestString(self.epochIdx, self.currentBatch, batchstart, batchend, meancost, timer:time().real, avgvalid)
	self.lastTestCost=meancost

   self.network:training()
end

function NeuralNet:printTestString(currentEpoch, currentBatch, batchstart, batchend, meancost, time, avgvalid)
   if avgvalid then
      print('test ->'..currentEpoch..'.'..currentBatch
      ..', test batches: '..batchstart..'-'..batchend
      ..', cost: '..string.format("%.3f", meancost)
      ..', valid % : '..string.format("%.1f", avgvalid*100)
      ..', time: '..string.format("%.0f", time*1000)..' ms')
   else
      print('test ->'..currentEpoch..'.'..currentBatch
      ..', test batches: '..batchstart..'-'..batchend
      ..', cost: '..string.format("%.3f", meancost)
      ..', time: '..string.format("%.0f", time*1000)..' ms')
   end
end






-- training

function NeuralNet:newEpoch()
	self:shuffleTrainSet()
	self.currentBatch=self.trainset[1]
	self.epochIdx=self.epochIdx+1
end

function NeuralNet:printTrainString(epochend, batchend, cost, time)
      print('->'..epochend..'.'..batchend
      ..', cost: '..string.format("%.3f", cost)
      ..', time: '..string.format("%.0f", time*1000)..' ms'
      ..', last val cost : '..string.format("%.3f", self.lastTestCost))
end

function NeuralNet:trainOnBatch(batchIdx)
	local timer=torch.Timer()

   self.network:zeroGradParameters()

	local input, target = self:getBatch(batchIdx)
	local output = self.network:forward(input)
	local cost = self.criterion:forward(output, target)
	local df_do = self.criterion:backward(output, target)
	self.network:backward(input, df_do)

   self.network:optimStep()

	return cost/input:size(1), timer:time().real
end


function NeuralNet:train(nepochs, freqs)
   local writefreq=freqs.writeFreq or 1
   local testfreq=freqs.testFreq or 100*writefreq
   local savefreq=freqs.saveFreq

   -- initializing stuff :
	if self.epochIdx==0 then
		self:newEpoch()
	end

	self.network:training()
	self.network:setBackProp()
	--self.network:clean()

	local timer=torch.Timer()

   local count=0
   local meancost = 0
   
   self:test()

	while self.epochIdx <= nepochs do
   	while self.currentBatch <= self.trainset[2] do
         
         local batchNum = self:getBatchNum(self.currentBatch)
			local cost, computeTime = self:trainOnBatch(batchNum)
			
			meancost = meancost + cost

         count = count + 1

         if count % writefreq == 0 then
            meancost = meancost / writefreq
            self:printTrainString(self.epochIdx, self.currentBatch, meancost, timer:time().real)
            self:updateTrainLogger(cost)
            timer:reset()
            meancost = 0
         end
		 
		 if savefreq ~= nil and count % savefreq == 0 then
			print('saving the net')
			torch.save(self.checkpointPath, self)
		 end

         if count % testfreq == 0 then
            timer:stop()
	         self:test()
	         timer:resume()
         end

         self.currentBatch = self.currentBatch+1
		end

      timer:stop()
	   self:newEpoch()
      timer:resume()
	end
end

