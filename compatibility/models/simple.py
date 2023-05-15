import torch
import torch.nn as nn

class ReLU_MLP(nn.Module):
    def __init__(self, layer_dims, output="linear", bias=True, layernorm=False):
        '''
        A generic ReLU MLP network.

        Arguments:
            - layer_dims: a list [d1, d2, ..., dn] where d1 is the dimension of input vectors and d1, ..., dn
                        is the dimension of outputs of each of the intermediate layers.
            - output: output activation function, either "sigmoid" or "linear".
            - layernorm: if True, apply layer normalization to the input of each layer.
        '''
        super(ReLU_MLP, self).__init__()
        layers = []
        for i in range(1, len(layer_dims) - 1):
            if (layernorm and i != 1):
                layers.append(nn.LayerNorm(layer_dims[i - 1]))
            layers.append(nn.Linear(layer_dims[i - 1], layer_dims[i], bias=bias))
            layers.append(nn.ReLU(layer_dims[i - 1]))
        if (output == "sigmoid"):
            layers.append(nn.Linear(layer_dims[-2], layer_dims[-1], bias=bias))
            layers.append(nn.Sigmoid())
        if (output == "linear"):
            layers.append(nn.Linear(layer_dims[-2], layer_dims[-1], bias=bias))

        self.layers = layers
        self.out = nn.Sequential(*layers)

    def forward(self, inp, *args):
        if (type(inp) == tuple):
            args = inp[1:]
            inp = inp[0]
        if (len(args) > 0):
            inp = torch.cat([inp] + list(args), dim=1)
        return self.out(inp)

    def clip_weights(self, c):
        for layer in self.layers:
            if (isinstance(layer, nn.Linear) or isinstance(layer, nn.LayerNorm)):
                layer.weight.data = torch.clamp(layer.weight.data, -c, c)
                layer.bias.data = torch.clamp(layer.bias.data, -c, c)