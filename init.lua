require 'torch'
require 'nn'

local nntrain = {}

nntrain.optimHooks = require 'nntrain.optimHooks'
nntrain.NeuralNet2 = require 'nntrain.NeuralNet2'
nntrain.NeuralNet2 = require 'nntrain.Dataset2'
nntrain.memorySave = require 'nntrain.memorySave'

return nntrain
