import torch
import torch.nn as nn
import math
from compatibility.models import  ReLU_MLP
#from ncsn.models.normalization import get_normalization
#from ncsn.models.layers import get_act, ResidualBlock, RefineBlock
#from datasets import data_transform

# so, do you like jazz?
def get_bary(config):
    
    if(config.target.data.dataset.upper() in ["GAUSSIAN", "GAUSSIAN-HD"]):
        source_dim = config.source.data.dim
        target_dim = config.target.data.dim
        layers = [source_dim] + config.baryproj.model.hidden_layers + [target_dim]
        return ReLU_MLP(layer_dims=layers, layernorm=False).to(config.device)
    
    
    if(config.baryproj.model.architecture == "fcn"):
        return FCSampler(
            input_im_size=config.source.data.image_size,
            input_channels=config.source.data.channels,
            output_im_size=config.target.data.image_size,
            output_channels=config.target.data.channels,
            hidden_layer_dims=config.model.hidden_layer_dims
        ).to(config.device)
    
    else:
        raise ValueError(f"{config.model.architecture} is not a recognized architecture.")


class FCSampler(nn.Module):
    def __init__(self, input_im_size, input_channels, output_im_size, output_channels, hidden_layer_dims):
        super(FCSampler, self).__init__()
        
        self.input_W = input_im_size
        self.input_C = input_channels
        self.output_W = output_im_size
        self.output_C = output_channels
        self.hidden_layer_dims = hidden_layer_dims
        
        self.inp_projector = nn.Linear(in_features=self.input_W ** 2 * self.input_C,
                                       out_features=hidden_layer_dims[0])
        
        self.outp_projector = nn.Linear(in_features = hidden_layer_dims[-1],
                                        out_features=self.output_W **2 * self.output_C)
        
        self.hidden = ReLU_MLP(layer_dims=hidden_layer_dims, layernorm=False)

    def forward(self, inp_image):
        inp_img_scale = 2 * inp_image - 1
        x = self.outp_projector(nn.functional.relu(self.hidden(self.inp_projector(inp_img_scale.flatten(start_dim=1)))))
        return (1/2) * (torch.tanh(x).reshape((-1, self.output_C, self.output_W, self.output_W)) + 1)



 