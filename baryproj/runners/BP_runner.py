import numpy as np
import glob
from tqdm import tqdm
import torch.nn.functional as F
import logging
import torch
import torchvision
import os
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
#from datasets import get_dataset, data_transform, inverse_data_transform
from compatibility.models import get_compatibility as _get_compatibility
from baryproj.models import get_bary
import util
import torch.optim as optim 
import wandb
import shutil
import yaml
import copy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


"""
import sys
sys.path.append('..')
from eot_benchmark.rotated_gaussian_benchmark import get_rotated_gaussian_sampler
from EntropicOTBenchmark.benchmark.metrics import compute_BW_UVP_with_gt_stats
from EntropicOTBenchmark.benchmark.rotated_gaussian_benchmark import get_rotated_gaussian_benchmark_stats
"""
__all__ = ['BPRunner']


#==================================#
def get_optimizer(config, parameters):
    if config.baryproj.optim.optimizer == 'Adam':
        if(hasattr(config.baryproj.optim, "beta2")):
            beta2 = config.baryproj.optim.beta2
        else:
            beta2 = 0.999

        return optim.Adam(parameters, lr=config.baryproj.optim.lr, weight_decay=config.baryproj.optim.weight_decay,
                          betas=(config.baryproj.optim.beta1, beta2), amsgrad=config.baryproj.optim.amsgrad,
                          eps=config.baryproj.optim.eps)
    
    elif config.baryproj.optim.optimizer == "LBFGS":
        return optim.LBFGS(parameters, lr=config.baryproj.optim.lr)
    elif config.baryproj.optim.optimizer == 'RMSProp':
        return optim.RMSprop(parameters, lr=config.baryproj.optim.lr, weight_decay=config.baryproj.optim.weight_decay)
    elif config.baryproj.optim.optimizer == 'SGD':
        return optim.SGD(parameters, lr=config.optim.lr, momentum=0.9)
    else:
        raise NotImplementedError('Optimizer {} not understood.'.format(config.baryproj.optim.optimizer))
#==================================#



#============ get CPAT ==========#
def get_compatibility(config):
    cnf_for_cpat = copy.deepcopy(config.compatibility)
    cnf_for_cpat.source = config.source
    cnf_for_cpat.target = config.target
    cnf_for_cpat.transport = config.transport
    cnf_for_cpat.device = config.device
    cnf_for_cpat.compatibility = config.compatibility
    return _get_compatibility(cnf_for_cpat)
#=================================#


#=============== Barycentirc Runner ===========#
class BPRunner():
    def __init__(self, config):
        
        
        #=============RESUME_BLOCK==========#
        print(util.yellow(" BP_runner.py : resume for baryproj... "))
        if not config.baryproj.logging.resume_training:
            if os.path.exists(config.baryproj.logging.log_path):
                overwrite = False
                response = input("Folder already exists. Overwrite? (Y/N) ")
                if response.upper() == 'Y':
                    overwrite = True

                if overwrite:
                    shutil.rmtree(config.baryproj.logging.log_path)
                    os.makedirs(config.baryproj.logging.log_path)

                    """
                    if os.path.exists(tb_path):
                        shutil.rmtree(tb_path)
                    """
                else:
                    print("Folder exists. Program halted.")
                    sys.exit(0)
            else:
                os.makedirs(config.baryproj.logging.log_path)

        with open(os.path.join(config.baryproj.logging.log_path, 'config.yml'), 'w+') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(util.yellow(" BP_runner.py : resume for baryptoj is ready "))
       #=============RESUME_BLOCK==========#
    
        self.config = config
        self.config.baryproj.logging.log_sample_path = os.path.join(self.config.baryproj.logging.log_path, 'samples')
        os.makedirs(self.config.baryproj.logging.log_sample_path, exist_ok=True)

    def train(self):
        
        #=========    Data    ============#
        if self.config.meta.problem_name.startswith("gaussian"):
            source_sampler = get_rotated_gaussian_sampler("input", self.config.source.data.dim, with_density=False,
                                         batch_size=self.config.baryproj.training.batch_size, device="cpu")
            
            target_sampler = get_rotated_gaussian_sampler("target",self.config.target.data.dim, with_density=False,
                                         batch_size=self.config.baryproj.training.batch_size, device="cpu")
            target_loader = target_sampler.loader
            source_loader = source_sampler.loader
            
        else:
            
            raise NotImplementedError
            
        source_batches = iter(source_loader)
        target_batches = iter(target_loader)
   
        print(util.magenta("Data is ready!"))
        #=====================================#
        
        
        
        #============ cpat is set  ==============#
        if self.config.baryproj.compatibility.ckpt_id is None:
            states = torch.load(os.path.join(self.config.compatibility.logging.log_path,
                         f'checkpoint_dim_{self.config.source.data.dim}_eps_{self.config.transport.coeff}.pth'),                                                                       map_location=self.config.device)
        else:
            states = torch.load( os.path.join(self.config.compatibility.logging.log_path,
                                                  f'checkpoint_{self.config.baryproj.compatibility.ckpt_id}.pth'),
                                             map_location=self.config.device)

        print(util.green("states is loaded"))
   
        cpat = get_compatibility(self.config)
        cpat.load_state_dict(states[0])
        cpat.eval()
        print(util.red("cpat is ready"))
        #========================================#
        
        
        #========== bary is set ==============#
        print(util.green("bary is being set ..."))
        baryproj = get_bary(self.config)
        bp_opt = get_optimizer(self.config, baryproj.parameters())
        print(util.green("bary is ready!"))
        #=====================================#
        
        
        
        #========== resume =======#
        if(self.config.baryproj.logging.resume_training):
            states = torch.load(os.path.join(self.config.baryproj.logging.log_path, 'checkpoint.pt'))
            baryproj.load_state_dict(states[0])
            bp_opt.load_state_dict(states[1])
            logging.info(f"Resuming training after {states[2]} steps.")

        logging.info("Optimizing the barycentric projection of the OT map.")
        #=========================#
        
        
        print(util.magenta("Bproj begins..."))
        
        with tqdm(total=self.config.baryproj.training.n_iters) as progress:
            for d_step in range(self.config.baryproj.training.n_iters):
                try:
                    (Xs, ys) = next(source_batches)
                    (Xt, yt) = next(target_batches)
                except StopIteration:
                    # Refresh after one epoch
                    source_batches = iter(source_loader)
                    target_batches = iter(target_loader)
                    (Xs, ys) = next(source_batches)
                    (Xt, yt) = next(target_batches)

                #Xs = data_transform(self.config.source, Xs)
                Xs = Xs.to(self.config.device)

                #Xt = data_transform(self.config.target, Xt)
                Xt = Xt.to(self.config.device)

                obj = bp_opt.step(lambda: self._bp_closure(Xs, Xt, cpat, baryproj, bp_opt))

                progress.update(1)
                progress.set_description_str("L2 Error: {:.4e}".format(obj.item()))
                
                """
                self.config.tb_logger.add_scalars('Optimization', {
                    'Objective': obj.item()
                }, d_step)
                """
                
                if(d_step  % self.config.baryproj.training.sample_freq == 0):
                    with torch.no_grad():
                        samples = baryproj(Xs).detach().cpu().numpy()
                        
                        fig,axes = plt.subplots(1,4,figsize=(12,4),squeeze=True, sharex=True, sharey=True)
                        axes[1].scatter(samples[:,0],samples[:,1], color='purple',edgecolor='black',s=30,label='mapped')
                        Xs = Xs.detach().cpu().numpy()
                        axes[0].scatter(Xs[:,0], Xs[:,1], color='yellowgreen',edgecolor='black',s=50,label='source')
                        Xt = Xt.detach().cpu().numpy()
                        axes[2].scatter(Xt[:,0], Xt[:,1], color='salmon',edgecolor='black',s=50,label='target')
                        axes[3].scatter(Xt[:,0], Xt[:,1], color='salmon',edgecolor='black',s=50,label='target')
                        axes[3].scatter(samples[:,0],samples[:,1], color='purple',edgecolor='black',s=30,label='mapped')
                        fig.tight_layout(pad=0.5)
                        axes[1].legend()
                        axes[2].legend()
                        axes[0].legend()
                        axes[3].legend()
                        
                        wandb.log({'Plot mapped samples' : [wandb.Image(util.fig2img(fig))]}, step=d_step + self.config.compatibility.training.n_iters) 
                      
                      
                        
                    """   
                    img_grid1 = torchvision.utils.make_grid(torch.clamp(samples, 0, 1))
                    img_grid2 = torchvision.utils.make_grid(torch.clamp(Xs, 0, 1))
                    self.config.tb_logger.add_image('Samples', img_grid1, d_step)
                    self.config.tb_logger.add_image('Sources', img_grid2, d_step)
                    """
                    
                wandb.log({"Loss of Baryproj": obj.item()},step=d_step + self.config.compatibility.training.n_iters)   
                if(d_step % self.config.baryproj.training.snapshot_freq == 0):
                    states = [
                        baryproj.state_dict(),
                        bp_opt.state_dict(),
                        d_step
                    ]
                    
                     
                     
                    
                    torch.save(states, os.path.join(self.config.baryproj.logging.log_path,
                    f'checkpoint_dim_{self.config.source.data.dim}_eps_{self.config.transport.coeff}_step_{d_step}.pth'))
                    torch.save(states, os.path.join(self.config.baryproj.logging.log_path,  
                    f'checkpoint_dim_{self.config.source.data.dim}_eps_{self.config.transport.coeff}.pth'))
                    
                if (d_step % self.config.baryproj.training.metrics_freq == 0):
                    
                    with torch.no_grad():
                        stats = get_rotated_gaussian_benchmark_stats(self.config.source.data.dim, self.config.transport.coeff, 'cpu',True)
                        mu_X, mu_Y, covariance_X, covariance_Y, optimal_plan_mu, optimal_plan_covariance = stats
                        init_samples, model_samples = [],[]
                        for _ in tqdm(range(self.config.metrics.samp_metrics//self.config.baryproj.training.batch_size)):
                        
                            try:
                                (Xs, ys) = next(source_batches)
                                (Xt, yt) = next(target_batches)
                            except StopIteration:
                                source_batches = iter(source_loader)
                                target_batches = iter(target_loader)
                                (Xs, ys) = next(source_batches)
                                (Xt, yt) = next(target_batches)
                            
                            mapped = baryproj(Xs.to(self.config.device))
                            init_samples.append(Xs.detach().cpu().numpy())
                            model_samples.append(mapped.detach().cpu().numpy())
                        
                        init_samples = np.stack(init_samples,axis=1).reshape(-1,self.config.source.data.dim)
                        model_samples = np.stack(model_samples,axis=1).reshape(-1,self.config.source.data.dim)
                         
                        concat_samples = np.concatenate([init_samples, model_samples],axis=1)
                        
                        
                        bw_uvp_terminal = compute_BW_UVP_with_gt_stats(model_samples,
                                         true_samples_mu=mu_Y, true_samples_covariance=covariance_Y)

                        bw_uvp_plan =  compute_BW_UVP_with_gt_stats(concat_samples,
                                             true_samples_mu=optimal_plan_mu, true_samples_covariance=optimal_plan_covariance)
                        
                        wandb.log({" EOT :BW UVP terminal":bw_uvp_terminal},step=d_step + self.config.compatibility.training.n_iters)
                        wandb.log({" EOT :BW UVP plan":bw_uvp_plan},step=d_step + self.config.compatibility.training.n_iters)

 

                    
    #=============================================#
    def _bp_closure(self, Xs, Xt, cpat, bp, bp_opt):
        """
        Xs - torch.Size([B,dim])
        Xt -torch.Size([B,dim])
        """
        """
        bp_opt.zero_grad()
        mapped = bp(Xs)
        distance = 0.5*torch.cdist(mapped,Xt,p=2)**2
        with torch.no_grad():
            K = 0.5*torch.cdist(Xs,Xt,p=2)**2
            u = cpat.inp_density_param_net.eval()
            v = cpat.outp_density_param_net.eval()
            plan = torch.exp((u(Xs)[:, None] + v(Xt)[None, :] - K) /  self.config.transport.coeff)
        obj = torch.mean(plan.detach()*distance)
        obj.backward()
        return obj
        
        """
        dx = cpat(Xs, Xt)
        nnz = (dx > 1e-20).flatten()
        Xt_hat = bp(Xs)
        transport_cost = ((Xt[nnz] - Xt_hat[nnz])**2).flatten(start_dim=1)
        cost = torch.mean(transport_cost, dim=1, keepdim=True) * dx[nnz]
        obj = torch.mean(cost)
        obj.backward()
        return obj
        
    #==============================================#
    
     

    
    #===============================================#
    """
    def sample(self):
        
        if self.config.meta.problem_name.startswith('gaussian'):
            source_sampler = get_rotated_gaussian_sampler("input", self.config.source.data.dim, with_density=False,
                                         batch_size=self.config.training.batch_size, device="cpu")
            source_loader = source_sampler.loader

        baryproj = get_bary(self.config)
        baryproj.eval()
         

        if self.config.bproj.sampling.ckpt_id is None:
            bp_states = torch.load(os.path.join(self.config.bproj.logging.log_path, 'checkpoint.pth'),                                                             map_location=self.config.device)
        else:
            bp_states = torch.load(os.path.join(self.config.bproj.logging.log_path, f'checkpoint_{self.config.bproj.sampling.ckpt_id}.pth'),
                                     map_location=self.config.device)

        baryproj.load_state_dict(bp_states[0])
        baryproj = baryproj.to(self.config.device)
        print(util.green(f"device is {self.config.device}"))
        print(util.magenta("baryproj is ready"))
        
        
        
        #===============================#
        if(not self.config.sampling.fid):
            
            if not self.config.meta.problem_name.startswith('gaussian'):
            
                pass
                
                dataloader = DataLoader(source_dataset,
                                    batch_size=self.config.sampling.batch_size,
                                    shuffle=True,
                                    num_workers=self.config.source.data.num_workers)
                 
                  
            else:
                source_loader = get_rotated_gaussian_sampler("input", self.config.source.data.dim, with_density=False,
                                         batch_size=self.config.baryproj.training.batch_size, device="cpu")
                dataloader = source_loader.loader
                 
            batch_samples = []
            start_samples = []
            for i in range(self.config.bproj.
                           sampling.n_batches):
                (Xs, _) = next(iter(dataloader))
                
                #Xs = data_transform(self.config.source, Xs)
                transport = baryproj(Xs.to(self.config.device))
                #batch_samples.append(inverse_data_transform(self.config, transport))
                batch_samples.append(transport)
                start_samples.append(Xs)

            sample = torch.cat(batch_samples, dim=0)
            start_sample = torch.cat(start_samples,dim=0)
             
            
            if not self.config.meta.problem_name.startswith('gaussian'):
                image_grid = make_grid(sample[:min(64, len(sample))], nrow=8)
                save_image(image_grid, os.path.join(self.config.image_folder, 'sample_grid.png'))

                source_grid = make_grid(Xs[:min(64, len(Xs))], nrow=8)
                save_image(source_grid, os.path.join(self.config.image_folder, 'source_grid.png'))

                np.save(os.path.join(self.config.image_folder, 'sample.npy'), sample.detach().cpu().numpy())
                np.save(os.path.join(self.config.image_folder, 'sources.npy'), Xs.detach().cpu().numpy())
            else:
                
                np.save(os.path.join(self.config.log_path,'transport.npy'), sample.detach().cpu().numpy())
                np.save(os.path.join(self.config.log_path, 'init.npy'), start_sample.detach().cpu().numpy())

        else:
            pass
            
            
            batch_size = self.config.sampling.samples_per_batch
            total_n_samples = self.config.sampling.num_samples4fid
            n_rounds = total_n_samples // batch_size
            
            dataloader = DataLoader(source_dataset,
                                    batch_size=self.config.sampling.samples_per_batch,
                                    shuffle=True,
                                    num_workers=self.config.source.data.num_workers)
            data_iter = iter(dataloader)

            img_id = 0
            for _ in tqdm(range(n_rounds), desc='Generating image samples for FID/inception score evaluation.'):
                with torch.no_grad():
                    (Xs, _) = next(data_iter)
                    Xs = data_transform(self.config.source, Xs).to(self.config.device)
                    transport = baryproj(Xs)
                for img in transport:
                    img = inverse_data_transform(self.config.target, img)
                    save_image(img, os.path.join(self.args.image_folder, 'image_{}.png'.format(img_id)))
                    img_id += 1
                del Xs
                del transport
    """
   