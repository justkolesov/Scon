import torch
import logging
from compatibility.models.imagecritic import FCImageCritic
from compatibility.models.compatibility import Compatibility
from compatibility.models import get_compatibility
from torch.utils.data import DataLoader
#from ncsn.losses import get_optimizer
from compatibility.models import get_cost
from datasets import get_dataset
from tqdm import tqdm
import os
from datasets import data_transform
import numpy as np
import util

import torch.optim as optim 
import wandb
 

"""
import sys
sys.path.append('..')
from eot_benchmark.rotated_gaussian_benchmark import get_rotated_gaussian_sampler
from eot_benchmark.gaussian_mixture_benchmark import get_guassian_mixture_benchmark_sampler 
from eot_benchmark.gaussian_mixture_benchmark import LoaderFromSampler
#from eot_benchmark.image_benchmark import get_image_benchmark
"""

#===================================#
def get_optimizer(config, parameters):
    if config.compatibility.optim.optimizer == 'Adam':
        if(hasattr(config.compatibility.optim, "beta2")):
            beta2 = config.compatibility.optim.beta2
        else:
            beta2 = 0.999

        return optim.Adam(parameters, lr=config.compatibility.optim.lr, weight_decay=config.compatibility.optim.weight_decay,
                          betas=(config.compatibility.optim.beta1, beta2), amsgrad=config.compatibility.optim.amsgrad,
                          eps=config.compatibility.optim.eps)
    
    elif config.compatibility.optim.optimizer == "LBFGS":
        return optim.LBFGS(parameters, lr=config.compatibility.optim.lr)
    elif config.compatibility.optim.optimizer == 'RMSProp':
        return optim.RMSprop(parameters, lr=config.compatibility.optim.lr, weight_decay=config.compatibility.optim.weight_decay)
    elif config.compatibility.optim.optimizer == 'SGD':
        return optim.SGD(parameters, lr=config.compatibility.optim.lr, momentum=0.9)
    else:
        raise NotImplementedError('Optimizer {} not understood.'.format(config.compatibility.optim.optimizer))
#===================================#        
        
        
#============ CPAT runner ==========#
class CpatRunner():
    
    def __init__(self, config):
        self.config = config

        
    def train(self):
        
        
        #============================DATA==============================#
        if self.config.meta.problem_name.startswith("gaussian"):
            
            source_sampler = get_rotated_gaussian_sampler("input", self.config.source.data.dim, with_density=False,
                                         batch_size=self.config.compatibility.training.batch_size, device="cpu") 
            source_loader = source_sampler.loader
            
            target_sampler = get_rotated_gaussian_sampler("target",self.config.target.data.dim, with_density=False,
                                         batch_size=self.config.compatibility.training.batch_size, device="cpu")
            target_loader = target_sampler.loader
            
        elif self.config.meta.problem_name.startswith("mixgauss"):
            
            source_sampler = get_guassian_mixture_benchmark_sampler ('input',  self.config.source.data.dim,  self.config.transport.coeff, self.config.compatibility.training.batch_size,  device="cpu", download=True)
            
            target_sampler = get_guassian_mixture_benchmark_sampler ('target',  self.config.target.data.dim,  self.config.transport.coeff, self.config.compatibility.training.batch_size,  device="cpu", download=False)
            
            #source_loader = source_sampler.loader
            #target_loader = target_sampler.loader
            
            source_loader = LoaderFromSampler(source_sampler,batch_size=self.config.compatibility.training.batch_size,
                                             num_batches=self.config.compatibility.training.num_batches)
            target_loader = LoaderFromSampler(target_sampler,batch_size=self.config.compatibility.training.batch_size,
                                             num_batches=self.config.compatibility.training.num_batches)
        
        
        elif self.config.meta.problem_name.endswith("images"):
            
            benchmark = get_image_benchmark(batch_size=self.config.compatibility.training.batch_size,
                                             eps=self.config.transport.coeff,glow_device=self.config.device_gpus.glow_device,
                                             samples_device=self.config.device_gpus.samples_device,download=True
                                            )
            
            source_sampler = benchmark.X_sampler
            source_test_sampler = benchmark.X_test_sampler
            
           
            target_sampler = benchmark.Y_sampler
            target_test_sampler = benchmark.Y_test_sampler
            
            source_loader = LoaderFromSampler(source_sampler,batch_size=self.config.compatibility.training.batch_size,
                                             num_batches=self.config.compatibility.training.num_batches)
            
            target_loader = LoaderFromSampler(target_sampler,batch_size=self.config.compatibility.training.batch_size,
                                             num_batches=self.config.compatibility.training.num_batches)
            
        
   
        else:
            
            source_dataset, source_test_dataset = get_dataset(self.config.source)
            
            source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=self.config.compatibility.training.batch_size,
                                   shuffle=True, num_workers=0, drop_last=True)
            

            target_dataset, target_test_dataset = get_dataset( self.config.target)
            
            target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=self.config.compatibility.training.batch_size,
                                       shuffle=True, num_workers=0, drop_last=True)
             
             
                
        source_batches = iter(source_loader)    
        target_batches = iter(target_loader)
        #==============================DATA=========================#
        
        
        
        
        
        #==============================CPAT=========================#
        print(util.green("cpat_runner.py : get_compatibility..."))
        cpat = get_compatibility(self.config)
        cpat_opt = get_optimizer(self.config, cpat.parameters())
        print(util.yellow("cpat is ready"))
        #===========================================================#
        
        
        
        
        
        #=======UPLOAD_CKPT======#
        if(self.config.compatibility.logging.resume_training):
            states = torch.load(os.path.join(self.new_config.compatibility.logging.log_path,                                                 f'checkpoint_dim_{self.config.source.data.dim}_eps_{self.config.transport.coeff}.pth'))
            cpat.load_state_dict(states[0])
            cpat_opt.load_state_dict(states[1]) 
            logging.info(f"Resuming training after {states[2]} steps.")
            print(util.green(" cpat_runner.py : ckpt of cpat is uploaded"))
        #========================#

            
            
            
        #==========Training=======#
        logging.info("Optimizing the compatibility function.")
        print(util.yellow(" cpat_runner.py : training of compatibility "))
        with tqdm(total=self.config.compatibility.training.n_iters) as progress:
            for d_step in range(self.config.compatibility.training.n_iters):

                try:
                    (Xs, ys) = next(source_batches)
                    (Xt, yt) = next(target_batches)
                    #Xs  = next(source_batches)
                    #Xt = next(target_batches)
                except StopIteration:
                    # Refresh after one epoch
                    source_batches = iter(source_loader)
                    target_batches = iter(target_loader)
                    (Xs, ys)= next(source_batches)
                    (Xt, yt) = next(target_batches)
                    #Xs  = next(source_batches)
                    #Xt = next(target_batches)

                #Xs = data_transform(self.config.source, Xs)
                Xs = Xs.to(self.config.device)

                #Xt = data_transform(self.config.target, Xt)
                Xt = Xt.to(self.config.device)

                obj = cpat_opt.step(lambda: self._cpat_closure(Xs, Xt, cpat, cpat_opt))
                avg_density = torch.mean(cpat.forward(Xs, Xt))

                obj_val = round(obj.item(), 5)
                avg_density_val = round(avg_density.item(), 5)
                progress.update(1)
                progress.set_description_str(f"Objective: {obj_val}. Average Density: {avg_density_val}")
                
                """
                self.config.tb_logger.add_scalars('Optimization', {
                    'Objective': obj_val,
                    'Average Density': avg_density_val
                }, d_step)
                """
                wandb.log({"Entropy OT loss": - obj_val},step=d_step)
                wandb.log({"Average density": avg_density_val},step=d_step)
            
                if(d_step  % self.config.compatibility.training.snapshot_freq == 0):
                    states = [
                        cpat.state_dict(),
                        cpat_opt.state_dict(),
                        d_step
                    ]
                     
                    torch.save(states, os.path.join(self.config.compatibility.logging.log_path,
                    f'checkpoint_dim_{self.config.source.data.dim}_eps_{self.config.transport.coeff}_step_{d_step}.pth'))
                    torch.save(states, os.path.join(self.config.compatibility.logging.log_path,  
                    f'checkpoint_dim_{self.config.source.data.dim}_eps_{self.config.transport.coeff}.pth'))

                    
    def _cpat_closure(self, Xs, Xt, cpat, cpat_opt):
        cpat_opt.zero_grad()
        density_real_inp = cpat.inp_density_param(Xs)
        density_real_outp = cpat.outp_density_param(Xt)
        density_reg = cpat.penalty(Xs, Xt)
        obj = torch.mean(density_real_inp + density_real_outp - density_reg)
        (-obj).backward() # for gradient ascent rather than descent
        return obj

    def cov_est(self, Xs, Xt, cpat):
        source_dim = self.config.source.data.dim
        target_dim = self.config.target.data.dim
        density = cpat.density(Xs, Xt).view((-1, 1, 1))
        joint = torch.cat((Xs.view((-1, source_dim, 1)), Xt.view((-1, target_dim, 1))), dim=1)
        covs = joint @ joint.transpose(1, 2)
        est = torch.mean(density * covs, dim=0)
        return est