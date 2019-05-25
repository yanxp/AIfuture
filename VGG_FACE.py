
import torch
import torch.nn as nn
# import torch.legacy.nn as lnn

from functools import reduce

import torchfile
from torch.autograd import Variable

class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func,self.forward_prepare(input)))

class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func,self.forward_prepare(input))


VGG_FACE = nn.Sequential( # Sequential,
	nn.Conv2d(3,64,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(64,64,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
	nn.Conv2d(64,128,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
	nn.Conv2d(128,256,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
	nn.Conv2d(256,512,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
	nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
	Lambda(lambda x: x.view(x.size(0),-1)), # View,
	nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(25088,4096)), # Linear,
	nn.ReLU(),
	nn.Dropout(0.5),
	nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(4096,4096)), # Linear,
	nn.ReLU(),
	nn.Dropout(0.5),
	nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(4096,2622)), # Linear,
	nn.Softmax(),
)

block_size = [2, 2, 3, 3, 3]

def load_weights(net , path="VGG_FACE.t7"):
	""" Function to load luatorch pretrained

    Args:
        path: path for the luatorch pretrained
    """
	model = torchfile.load(path)
	counter = 1
	block = 1
	k = 0

	for i, layer in enumerate(model.modules):
		self_layer = None
		if layer.weight is not None:
			if block <= 5:
				while self_layer is None:
					if isinstance(net[k],nn.Conv2d):
						self_layer = net[k]
						k += 1
					else:
						k += 1
				counter += 1
				if counter > block_size[block - 1]:
					counter = 1
					block += 1
				self_layer.weight.data[...] = torch.Tensor(layer.weight).view_as(self_layer.weight)[...]
				self_layer.bias.data[...] = torch.Tensor(layer.bias).view_as(self_layer.bias)[...]
			else:

				while self_layer is None:
					if not isinstance(net[k], Lambda) and isinstance(net[k], nn.Sequential):
						self_layer = net[k][1]
						k += 1
					else:
						k += 1
				block += 1
				self_layer.weight.data[...] = torch.Tensor(layer.weight).view_as(self_layer.weight)[...]
				self_layer.bias.data[...] = torch.Tensor(layer.bias).view_as(self_layer.bias)[...]
