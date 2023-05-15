import numpy as np
import tqdm
import logging
import torch
import copy
import os
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from scones.models import GaussianScore, GaussianCpat, JointMarginalGaussianCpat, MixtureGaussianScore
#from datasets import get_dataset
from scones.models.langevin_dynamics import Langevin_dynamics, compare_Langevin_dynamics, joint_marginal_Langevin_dynamics
from baryproj.models import get_bary as _get_bary
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from compatibility.models import get_compatibility as _get_compatibility
import util
from tqdm import tqdm
import wandb
"""
import sys
sys.path.append('..')
from eot_benchmark.rotated_gaussian_benchmark import get_rotated_gaussian_sampler
from eot_benchmark.rotated_gaussian_benchmark import get_rotated_gaussian_benchmark_stats
from EntropicOTBenchmark.benchmark.metrics import compute_BW_UVP_with_gt_stats
from eot_benchmark.gaussian_mixture_benchmark import get_guassian_mixture_benchmark_sampler 
from eot_benchmark.gaussian_mixture_benchmark import LoaderFromSampler
"""
                                   
__all__ = ['GaussianRunner']

 

def get_bary(config):
    cnf_for_bproj = copy.deepcopy(config.compatibility)
    cnf_for_bproj.source = config.source
    cnf_for_bproj.target = config.target
    cnf_for_bproj.transport = config.transport
    return _get_bary(cnf_for_bproj)



#========= get score function ========#
def get_scorenet(config, target):
    
    if config.meta.problem_name.startswith("gaussian"):
        cnf_for_ncsn = copy.deepcopy(config.ncsn)
        cnf_for_ncsn.data = config.target.data
        mu_X, mu_Y, cov_X, cov_Y,_,_ = get_rotated_gaussian_benchmark_stats(config.target.data.dim, config.transport.coeff,                                                                                    device="cpu", download=False)
        cnf_for_ncsn.data.mean = mu_Y
        cnf_for_ncsn.data.cov =  cov_Y
        cnf_for_ncsn.device = config.device

        return GaussianScore(cnf_for_ncsn)
    
    elif config.meta.problem_name.startswith("mixgauss"):
        cnf_for_ncsn = config.scones
        cnf_for_ncsn.device = config.device
        cnf_for_ncsn.target = config.target
        return MixtureGaussianScore(cnf_for_ncsn, target)
    
    else:
        raise NotImplementedError
#=====================================#



#=========== get cpat ================#
def get_compatibility(config):
    
    config_for_cpat = copy.deepcopy(config.compatibility)
    config_for_cpat.source = config.source
    config_for_cpat.target = config.target
    config_for_cpat.transport = config.transport
    config_for_cpat.compatibility = config.compatibility
    config_for_cpat.device = config.device
    return _get_compatibility(config_for_cpat)
#=====================================#


#========== Gaussian Runner ==========#
class GaussianRunner():
    
    def __init__(self, config):
         
        self.config = config
        config.scones.logging.log_sample_path = os.path.join(config.scones.logging.log_path, 'samples')
        os.makedirs(config.scones.logging.log_sample_path, exist_ok=True)

    def sample(self):
        
        
        #============ CPAT is set ==============#
        if self.config.scones.compatibility.ckpt_id is None:    
            cpat_states = torch.load(os.path.join(self.config.compatibility.logging.log_path,
               f'checkpoint_dim_{self.config.source.data.dim}_eps_{self.config.transport.coeff}.pth'),
                                     map_location=self.config.device)
        else:
            cpat_states = torch.load(os.path.join(self.config.compatibility.logging.log_path,
              f'checkpoint_dim_{self.config.source.data.dim}_eps_{self.config.transport.coeff}_step_{self.config.scones.compatibility.ckpt_id}.pth'),
                                     map_location=self.config.device)
        cpat = get_compatibility(self.config)
        
       
        if(self.config.compatibility.model.architecture == "comparison"):
            cpat[0].load_state_dict(cpat_states[0])
        else:
            cpat.load_state_dict(cpat_states[0])
        print(util.green("Compatibility is set"))
        #=======================================#
        
        
        
        
        #==========Barycentric_Projection========#
        """
        baryproj_data_init = (hasattr(self.config, "baryproj") and self.config.ncsn.sampling.data_init)

        if(baryproj_data_init):
            if(self.config.baryproj.ckpt_id is None):
                bproj_states = torch.load(
                    os.path.join('scones', self.config.baryproj.log_path, 'checkpoint.pth'),
                    map_location=self.config.device)
            else:
                bproj_states = torch.load(os.path.join('scones', self.config.baryproj.log_path,
                                                      f'checkpoint_{self.config.baryproj.ckpt_id}.pth'),
                                         map_location=self.config.device)

            bproj = get_bary(self.config)
            bproj.load_state_dict(bproj_states[0])
            bproj = torch.nn.DataParallel(bproj)
            bproj.eval()
        """
        #========================================#
        
        
        
        
        #============= Prepare data =============#
        if self.config.meta.problem_name.startswith('gaussian'):
            
            target_dataset =  get_rotated_gaussian_sampler("target",self.config.target.data.dim, with_density=False,
                                         batch_size=self.config.scones.training.batch_size, device="cpu") 
            
            input_dataset =   get_rotated_gaussian_sampler("input", self.config.source.data.dim, with_density=False,
                                         batch_size=self.config.scones.training.batch_size, device="cpu")
            
            target_loader = target_dataset.loader
            input_loader  = input_dataset.loader
            
        elif self.config.meta.problem_name.startswith('mixgauss'):
            
            source_sampler = get_guassian_mixture_benchmark_sampler ('input',  self.config.source.data.dim,  self.config.transport.coeff, self.config.scones.training.batch_size,  device="cpu", download=False)
            
            target_sampler = get_guassian_mixture_benchmark_sampler ('target',  self.config.target.data.dim,  self.config.transport.coeff, self.config.scones.training.batch_size,  device="cpu", download=False)
            
            #source_loader = source_sampler.loader
            #target_loader = target_sampler.loader
            
            input_loader = LoaderFromSampler(source_sampler,batch_size=self.config.scones.training.batch_size,
                                             num_batches=10)
            target_loader = LoaderFromSampler(target_sampler,batch_size=self.config.scones.training.batch_size,
                                             num_batches=10)
            
        else:
            raise NotImplementedError
                 
        print(util.green("data is ready!"))
        #=======================================#
        
        
        
        #========= SCORE function =========#
        score = get_scorenet(self.config, target_sampler).to(self.config.device)
        score = torch.nn.DataParallel(score, device_ids=[self.config.device])
        score.eval()
        print(util.yellow('score is ready!'))
        #==================================#
        
        
        
        #========= Initilization of dynamics =============#
        dataloader =  input_loader
        
        (Xs, _) = next(iter(dataloader))
        Xs_global = torch.cat([Xs] * self.config.ncsn.sampling.samples_per_source, dim=0).to(self.config.device).type(torch.float)
        Xs_global = Xs_global.view(self.config.ncsn.sampling.sources_per_batch *
                                 self.config.ncsn.sampling.samples_per_source,
                                 self.config.target.data.dim, 1, 1)
        
        """
        if baryproj_data_init:
            Xs_init = torch.clone(Xs_global).to(self.config.device).view((-1, self.config.source.data.dim))
            init_Xt = bproj(Xs_init).detach()


            xs = Xs_init.detach().cpu().numpy()
            xt = init_Xt.cpu().numpy()
            pair = np.concatenate([xs, xt], axis=1)
            print(np.cov(pair, rowvar=False))
            lines = [[(xs[i][0], xt[i][0]), (xs[i][1], xt[i][1])] for i in range(100)]
            lc = LineCollection(lines)
            fig, ax = plt.subplots()
            ax.add_collection(lc)
            plt.xlim(-10, 10)
            plt.ylim(-10, 10)
            plt.show()


            init_Xt = init_Xt.view((self.config.ncsn.sampling.sources_per_batch *
                                 self.config.ncsn.sampling.samples_per_source,
                                 self.config.target.data.dim, 1, 1))
            init_Xt.requires_grad = True
        """
        if self.config.ncsn.sampling.data_init:
            init_Xt = torch.clone(Xs_global)
            init_Xt.requires_grad = True
            init_Xt = init_Xt.to(self.config.device)
            
 
        else:
            init_Xt = torch.zeros(self.config.ncsn.sampling.sources_per_batch *
                                 self.config.ncsn.sampling.samples_per_source,
                                 self.config.target.data.dim, 1, 1,
                                 device=self.config.device)
            init_Xt.requires_grad = True
            init_Xt = init_Xt.to(self.config.device)
        #=============================================#
        
        
        
        
        #============== Langevin Dynamics ================#
        """
        if(self.config.compatibility.model.architecture == "comparison"):
            all_samples = compare_Langevin_dynamics(init_Xt, Xs_global, score, cpat,
                                            self.config.ncsn.sampling.n_steps_each,
                                            self.config.ncsn.sampling.step_lr,
                                            verbose=True,
                                            sample_every=self.config.ncsn.sampling.sample_every,
                                            final_only=self.config.ncsn.sampling.final_only)
            
        elif(self.config.compatibility.model.architecture == "joint_marginal"):
            all_samples = joint_marginal_Langevin_dynamics(init_Xt, Xs_global, score, cpat,
                                                    self.config.ncsn.sampling.n_steps_each,
                                                    self.config.ncsn.sampling.step_lr,
                                                    verbose=True,
                                                    sample_every=self.config.ncsn.sampling.sample_every,
                                                    final_only=self.config.ncsn.sampling.final_only)
            
        else:
            
            all_samples = Langevin_dynamics(init_Xt, Xs_global, score, cpat,
                                               self.config.ncsn.sampling.n_steps_each,
                                               self.config.ncsn.sampling.step_lr,
                                               verbose=False,
                                               sample_every=self.config.ncsn.sampling.sample_every,
                                               final_only=self.config.ncsn.sampling.final_only)
            
          
        all_samples = torch.stack(all_samples, dim=0)
        all_samples = all_samples.view((-1,
                                        self.config.ncsn.sampling.sources_per_batch,
                                        self.config.ncsn.sampling.samples_per_source,
                                        self.config.target.data.dim))
        print(util.green("Samples are obtained!"))
        """
        #============================================#
        
        
         
        #============ Metrics ============#
        """
        init_samples = init_Xt.reshape(-1,self.config.source.data.dim)
        target_sample = all_samples[-1].view(self.config.ncsn.sampling.sources_per_batch *
                                      self.config.ncsn.sampling.samples_per_source,
                                      self.config.target.data.dim)
        """
        
        
        stats = get_rotated_gaussian_benchmark_stats(self.config.target.data.dim, self.config.transport.coeff, device='cpu' )
        mu_X, mu_Y, covariance_X, covariance_Y, optimal_plan_mu, optimal_plan_covariance = stats
        
        init_samples, model_samples = [],[]
        for _ in tqdm(range(1)):
            
            dataloader =  input_loader
        
            (Xs, _) = next(iter(dataloader))
            Xs_global = torch.cat([Xs] * self.config.ncsn.sampling.samples_per_source, dim=0).to(self.config.device).type(torch.float)
            Xs_global = Xs_global.view(self.config.ncsn.sampling.sources_per_batch *
                                 self.config.ncsn.sampling.samples_per_source,
                                 self.config.target.data.dim, 1, 1)
            
            """
            if baryproj_data_init:
            Xs_init = torch.clone(Xs_global).to(self.config.device).view((-1, self.config.source.data.dim))
            init_Xt = bproj(Xs_init).detach()


            xs = Xs_init.detach().cpu().numpy()
            xt = init_Xt.cpu().numpy()
            pair = np.concatenate([xs, xt], axis=1)
            print(np.cov(pair, rowvar=False))
            lines = [[(xs[i][0], xt[i][0]), (xs[i][1], xt[i][1])] for i in range(100)]
            lc = LineCollection(lines)
            fig, ax = plt.subplots()
            ax.add_collection(lc)
            plt.xlim(-10, 10)
            plt.ylim(-10, 10)
            plt.show()


            init_Xt = init_Xt.view((self.config.ncsn.sampling.sources_per_batch *
                                 self.config.ncsn.sampling.samples_per_source,
                                 self.config.target.data.dim, 1, 1))
            init_Xt.requires_grad = True
            """
            if self.config.ncsn.sampling.data_init:
                init_Xt = torch.clone(Xs_global)
                init_Xt.requires_grad = True
                init_Xt = init_Xt.to(self.config.device)
                
             
            
            else:
                init_Xt = torch.zeros(self.config.ncsn.sampling.sources_per_batch *
                                 self.config.ncsn.sampling.samples_per_source,
                                 self.config.target.data.dim, 1, 1,
                                 device=self.config.device)
                init_Xt.requires_grad = True
                init_Xt = init_Xt.to(self.config.device)
            
            
             
            
            all_samples = Langevin_dynamics(self.config,init_Xt, Xs_global, score, cpat,
                                               self.config.ncsn.sampling.n_steps_each,
                                               self.config.ncsn.sampling.step_lr,
                                               verbose=False,
                                               sample_every=self.config.ncsn.sampling.sample_every,
                                               final_only=self.config.ncsn.sampling.final_only)
            
            all_samples = torch.stack(all_samples, dim=0)
            all_samples = all_samples.view((-1,
                                        self.config.ncsn.sampling.sources_per_batch,
                                        self.config.ncsn.sampling.samples_per_source,
                                        self.config.target.data.dim))
            
            x = init_Xt.reshape(-1,self.config.source.data.dim)
            t_x = all_samples[-1].view(self.config.ncsn.sampling.sources_per_batch *
                                      self.config.ncsn.sampling.samples_per_source,
                                      self.config.target.data.dim)
           
            init_samples.append(x.detach().cpu().numpy())
            model_samples.append(t_x.detach().cpu().numpy())
         
         

        init_samples =  np.stack(init_samples,axis=1).reshape(-1,self.config.target.data.dim)   
        model_samples = np.stack(model_samples,axis=1).reshape(-1,self.config.target.data.dim)
        #trajectory = np.stack(trajectory,axis=1).reshape(opt.samp_metrics,-1,opt.data_dim[0])
        concat_samples = np.concatenate([init_samples, model_samples],axis=1)
        
        bw_uvp_terminal = compute_BW_UVP_with_gt_stats(model_samples,
                                         true_samples_mu=mu_Y, true_samples_covariance=covariance_Y)

        bw_uvp_plan =  compute_BW_UVP_with_gt_stats(concat_samples,
                                             true_samples_mu=optimal_plan_mu, true_samples_covariance=optimal_plan_covariance)
        
        print(f"BW-UVP", bw_uvp_terminal)
        print(f"BW-UVP plan", bw_uvp_plan)
        

        #=================================#
        
    

                         
                          
                         
