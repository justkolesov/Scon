import torch
import torch.nn as nn
from .simple import ReLU_MLP

class FCImageCritic(torch.nn.Module):
    def __init__(self, input_im_size, input_channels, hidden_layer_dims, bias=True):
        super(FCImageCritic, self).__init__()
        
        self.input_W = input_im_size
        self.input_C = input_channels
        self.hidden_layer_dims = hidden_layer_dims
        
        self.inp_projector = torch.nn.Linear(in_features=self.input_W ** 2 * self.input_C,
                                       out_features=hidden_layer_dims[0],
                                       bias=bias)
        
        self.outp_projector = torch.nn.Linear(in_features = hidden_layer_dims[-1], out_features=1, bias=bias)
        
        self.hidden = ReLU_MLP(layer_dims=hidden_layer_dims,  bias=bias)
        
    def forward(self, inp_image):
        
        """
        inp_image - torch.Size([])
        """
        x = inp_image.flatten(start_dim=1)
        x = self.inp_projector(x)
        x = self.hidden(x)
        x = nn.functional.relu(x)
     
        return self.outp_projector(x)