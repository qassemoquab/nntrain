local ffi = require 'ffi'

local Dataset = torch.class('nn.Dataset2')

local doc = [[
What should be the API ?
nn.Dataset(sampleGenerator, defTableGenerator, numSamples, batchSize, targetFolder)
- defTableGenerator(idx) is a function that returns a table containing whatever is needed to generate idx-th sample (with absolute paths)
- sampleGenerator(defTable) is a function that returns a sample using what's in the defTable, with sizes consistent across the dataset

This call would generate, in targetFolder :
- a bunch of files : batch_***_def.t7
   - they contain all defTables for the samples that we are interested in
   - they contain the sampleGenerator function
- a computeMean.t7 file
- a batchLoader.t7 file


Such that you can do : 
for each batch *** :
   foo = torch.load('batch_***_def.t7')
   foo:generate() to obtain :
   - targetFolder/batch_***.t7 
   - targetFolder/batch_***_mean.t7
=> we want each batch definition file to be self-contained
=> we want to be able to generate on the fly if needed
=> the real goal is to run that in parallel on the cluster



then, after :
   m = torch.load('computeMean.t7')
   m:computeMean() to obtain :
   - targetFolder/datasetMean.t7
=> this should be run only once


then, after, it would connect to nn.NeuralNet2 in the following way :
   ds = torch.load('batchLoader.t7')
   ds:setBatchDir('batchFolder') 
      - just in case we copy to another folder a dataset that was already generated.
   ds:setMean('someOtherDatasetMean.t7')
      - just in case we want to substract another mean from the dataset
   ds:getNumSamples()
   ds:setBatchSize(requestedBatchSize) 
      - default batchsize would be dataset real batch size
      - function nn.NeuralNet2.getBatchOfSize can help for different batch sizes.
   ds:getBatchSize()
   ds:getNumBatches()
   ds:initialize()
      - this would check if the mean exists, and load whatever other stuff that seems important
   ds:getBatch(idx)
      - this would call ds:initialize() if necessary
      - if the batch file doesn't exist, it generates it on the fly
      
   nn.NeuralNet2:setDataset(ds)

]]


function Dataset:__init()

end

function Dataset:setSampleGenerator(sampleGenerator)
   self.sampleGenerator = sampleGenerator
end

function Dataset:setDefTableGenerator(defTableGenerator)
   self.defTableGenerator = defTableGenerator
end

function Dataset:setNumSamples(numSamples)
   self.numSamples = numSamples
end

function Dataset:setBatchSize(batchSize)
   self.batchSize = batchSize
end

function Dataset:setTargetFolder(targetFolder)
   self.targetFolder = targetFolder
end

function Dataset:shuffleOrder()
   self.shuffle = torch.randperm(self.numSamples)
end

function Dataset:getNumBatches()
   self.numBatches = math.ceil(self.numSamples / self.batchSize)
   return self.numBatches
end

function Dataset:getNumSamples()
   return self.numSamples
end

function Dataset:start()
   self:shuffleOrder()
   -- batch definition files
   for batchIdx = 1, self:getNumBatches() do
      local batchTable = {}
      minIdx = 1 + (batchIdx-1)*self.batchSize
      maxIdx = math.min(batchIdx*self.batchSize, self.numSamples)

      for sampleIdx = minIdx, maxIdx do
         local realIdx = self.shuffle[sampleIdx]
         batchTable[sampleIdx - minIdx + 1] = self.defTableGenerator(realIdx)
      end

      local defFileName = paths.concat(self.targetFolder, 'batch_'..batchIdx..'_def.t7')
      local runFileName = paths.concat(self.targetFolder, 'batch_'..batchIdx..'_run.lua')

      local batch = {}
      batch.sampleGenerator = self.sampleGenerator
      batch.batchTable = batchTable
      batch.batchIdx = batchIdx
      batch.batchFileName = paths.concat(self.targetFolder, 'batch_'..batchIdx..'.t7')
      batch.meanFileName = paths.concat(self.targetFolder, 'batch_'..batchIdx..'_mean.t7')
      batch.batchSize = maxIdx - minIdx + 1
      function batch:generate() 
         local sampleExample, targetExample = self.sampleGenerator(batchTable[1])

         local sampleExampleDims=#(#sampleExample)
         local targetExampleDims=#(#targetExample)

         local batchSampleDims=torch.LongStorage(1+sampleExampleDims)
         local batchTargetDims=torch.LongStorage(1+targetExampleDims)

         batchSampleDims[1]=self.batchSize
         for d=1,sampleExampleDims do
            batchSampleDims[1+d]=(#sampleExample)[d]
         end

         batchTargetDims[1]=self.batchSize
         for d=1,targetExampleDims do
            batchTargetDims[1+d]=(#targetExample)[d]
         end

         local sampleBatch=sampleExample.new(batchSampleDims)
         local targetBatch=targetExample.new(batchTargetDims)

         sampleBatch:select(1,1):copy(sampleExample)
         targetBatch:select(1,1):copy(targetExample)

         for sampleIdx = 2, self.batchSize do
            local sampleExample, targetExample = self.sampleGenerator(batchTable[sampleIdx])
            sampleBatch:select(1,sampleIdx):copy(sampleExample)
            targetBatch:select(1,sampleIdx):copy(targetExample)
         end
         
         torch.save(self.batchFileName, {sampleBatch, targetBatch})
         torch.save(self.meanFileName, {sampleBatch:float():sum(1), self.batchSize})
      end
      
      print('generated def file for batch '..batchIdx..' : '..defFileName)
      torch.save(defFileName, batch)
      
      local runFile=io.open(runFileName, 'w')
      runFile:write("foo=torch.load('".. defFileName .."')\n")
      runFile:write("foo:generate()\n")
      runFile:close()

      print('generated run file for batch '..batchIdx..' : '..runFileName)


   end

   local scriptFileName = paths.concat(self.targetFolder, 'generateDataset.sh')
   local scriptFile=io.open(scriptFileName, 'w')
   scriptFile:write("#!/bin/bash\n")
   scriptFile:write('(for file in *run.lua ;do echo "$file"; done;) | xargs parallel th --\n')
   scriptFile:write('th computeMean.lua\n')
   scriptFile:close()

   print('generated dataset generation script : '..scriptFileName)

   -- mean computation file
   local meanCompFileName = paths.concat(self.targetFolder, 'computeMean.t7')
   local meanFileName = paths.concat(self.targetFolder, 'datasetMean.t7')
   local meanComp = {}
   meanComp.targetFolder=self.targetFolder
   meanComp.numBatches = self:getNumBatches()
   meanComp.meanFileName = meanFileName
   function meanComp:computeMean()
      local m = torch.load(paths.concat(self.targetFolder, 'batch_1_mean.t7'))
      local mean = m[1]:clone()
      local count = m[2]
      for batchIdx = 2, self.numBatches do
         m = torch.load(paths.concat(self.targetFolder, 'batch_'..batchIdx..'_mean.t7'))
         mean:add(m[1])
         count = count + m[2]
      end
      mean:div(count)
      torch.save(self.meanFileName, mean)
      print('generated mean : '..self.meanFileName..', samples : '..count..', batches : '..self.numBatches)
   end
   torch.save(meanCompFileName, meanComp)
   print('generated mean computation file : '..meanCompFileName)

   local meanRunFileName=paths.concat(self.targetFolder, 'computeMean.lua')
   local meanRunFile=io.open(meanRunFileName, 'w')
   meanRunFile:write("foo=torch.load('".. meanCompFileName .."')\n")
   meanRunFile:write("foo:computeMean()\n")
   meanRunFile:close()

   print('generated run file for mean computation '..meanRunFileName)


   -- batch loader
   local batchLoaderFileName = paths.concat(self.targetFolder, 'batchLoader.t7')
   local batchLoader = {}
   batchLoader.batchSize = self.batchSize
   batchLoader.requestedBatchSize = self.batchSize
   batchLoader.targetFolder = self.targetFolder
   batchLoader.numSamples = self.numSamples
   batchLoader.meanFileName = paths.concat(self.targetFolder, 'datasetMean.t7')


   function batchLoader:setBatchDir(targetFolder)
      self.targetFolder = targetFolder
   end

   function batchLoader:setMeanFileName(meanFileName)
      self.meanFileName = meanFileName
   end

   function batchLoader:getNumSamples()
      return self.numSamples
   end

   function batchLoader:getBatchSize()
      return self.requestedBatchSize
   end

   function batchLoader:setBatchSize(requestedBatchsize)
      self.requestedBatchSize = requestedBatchsize
   end

   function batchLoader:getNumBatches()
      return math.ceil(self.numSamples / self.requestedBatchSize)
   end

   function batchLoader:loadMean(sampleSize)
      if not self.oldmean then 
         self.oldmean = torch.load(self.meanFileName)
      end
      if (not self.mean) or (self.mean:size(1) ~= sampleSize) then
         --self.mean should be a 4D tensor (1,y,x,c)
         local batchDims=#self.oldmean
         batchDims[1]=sampleSize
         self.mean = self.oldmean.new(batchDims)
         for i = 1, sampleSize do
            self.mean:select(1,i):copy(self.oldmean:select(1,1))
         end
      end
   end

   function batchLoader:loadDataBatch(idx)
      local batchFile = torch.load(paths.concat(self.targetFolder, 'batch_'..idx..'.t7'))
      local b = batchFile[1]:float()
      local t = batchFile[2]
      return b,t
   end

   function batchLoader:getBatch(batchIdx)
      if batchIdx > self:getNumBatches() then error('batch index out of range') end
      local batchsize = self.batchSize
      local requestedSize = self.requestedBatchSize
      local sample_start = (batchIdx-1)*requestedSize+1
      local sample_end = math.min((batchIdx)*requestedSize, self.numSamples)
      
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
      
      local b,t = self:loadDataBatch(firstbatch)
      
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
            b, t = self:loadDataBatch(currentbatch)
            out_b:narrow(1,numexamplesfromfirstbatch+1+count*batchsize,batchsize):copy(b)
            out_t:narrow(1,numexamplesfromfirstbatch+1+count*batchsize,batchsize):copy(t)
            count=count+1
         end
         
         -- last batch
         b,t = self:loadDataBatch(lastbatch)
         out_b:narrow(1,numexamplesfromfirstbatch+1+count*batchsize,numexamplesfromlastbatch):copy(b:narrow(1,1,numexamplesfromlastbatch))
         out_t:narrow(1,numexamplesfromfirstbatch+1+count*batchsize,numexamplesfromlastbatch):copy(t:narrow(1,1,numexamplesfromlastbatch))
         
      end

      self:loadMean(out_b:size(1))
      out_b:add(-1, self.mean)

      return out_b, out_t
   end
   
   torch.save(batchLoaderFileName, batchLoader)
   print('generated batch loader : '..batchLoaderFileName)
  
   
   
end








