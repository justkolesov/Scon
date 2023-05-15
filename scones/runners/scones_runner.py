import os
import copy

import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from ncsn.models.ncsnv2 import NCSNv2Deeper, NCSNv2, NCSNv2Deepest
from ncsn.models import get_sigmas
from ncsn.models.ema import EMAHelper
from datasets import get_dataset
from torch.utils.data import DataLoader

from scones.models.langevin_dynamics import  anneal_Langevin_dynamics
from compatibility.models import get_compatibility as _get_compatibility
import util
from tqdm import tqdm
from datasets import inverse_data_transform
"""
import sys
sys.path.append('..')
from eot_benchmark.rotated_gaussian_benchmark import get_rotated_gaussian_sampler
from eot_benchmark.gaussian_mixture_benchmark import get_guassian_mixture_benchmark_sampler 
from eot_benchmark.gaussian_mixture_benchmark import LoaderFromSampler
#from eot_benchmark.image_benchmark import get_image_benchmark
"""

class Sampler:
    def __init__(
        self, device='cuda',
    ):
        self.device = device
    
    def sample(self, size=5):
        pass
    
class LoaderSampler(Sampler):
    def __init__(self, loader, device='cuda'):
        super(LoaderSampler, self).__init__(device)
        self.loader = loader
        self.it = iter(self.loader)
        
    def sample(self, size=5):
        assert size <= self.loader.batch_size
        try:
            batch, _ = next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            return self.sample(size)
        if len(batch) < size:
            return self.sample(size)
            
        return batch[:size].to(self.device)

#=====================================#
def get_scorenet(config):
    
    cnf_for_ncsn = copy.deepcopy(config.ncsn)
    cnf_for_ncsn.data = config.target.data
    cnf_for_ncsn.device = config.device
    return NCSNv2(cnf_for_ncsn) 
#=====================================#    
    

    
    
#====================================#
def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256. * 255. + torch.rand_like(X) / 256.
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:
        X = 2 * X - 1.
    elif config.data.logit_transform:
        X = logit_transform(X)

    if hasattr(config, 'image_mean'):
        return X - config.image_mean.to(X.device)[None, ...]

    return X.float()
#====================================#


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


class SCONESRunner():
    
    def __init__(self,  config):
        
        self.config = config
        config.scones.logging.log_sample_path = os.path.join(config.scones.logging.log_path, 'samples')
        os.makedirs(config.scones.logging.log_sample_path, exist_ok=True)
        
    def sample(self):
        
        
        #=================== SCORE  ===================#
        score = get_scorenet(self.config).to(self.config.device)
        score = torch.nn.DataParallel(score, device_ids=self.config.device_gpus.ngpu)
        
        sigmas_th = get_sigmas(self.config.ncsn)
        sigmas = sigmas_th.cpu().numpy()
        
    
        if self.config.ncsn.sampling.ckpt_id is None:
                ncsn_states = torch.load(os.path.join( self.config.ncsn.ncsn_path,
                                                      'checkpoint.pth'),
                                         map_location=self.config.device)
        else:
                ncsn_states = torch.load(os.path.join( self.config.ncsn.ncsn_path,
                                                      f'checkpoint_{self.config.ncsn.sampling.ckpt_id}.pth'),
                                    map_location=self.config.device)
       
    
        if("module.sigmas" in ncsn_states[0].keys()):
            ncsn_states[0]["module.sigmas"] = sigmas_th

        score.load_state_dict(ncsn_states[0], strict=True)
        score.eval()
        print(util.magenta("Score is ready!"))
        #=================== SCORE  ===================#
        
        
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
        
        
        if self.config.ncsn.model.ema:
            ema_helper = EMAHelper(mu=self.config.ncsn.model.ema_rate)
            ema_helper.register(score)
            ema_helper.load_state_dict(ncsn_states[-1])
            ema_helper.ema(score)
        
        
        #=============== DATA ==================#
        """
        benchmark = get_image_benchmark(batch_size=self.config.scones.training.batch_size,
                                             eps=self.config.transport.coeff,glow_device=self.config.device_gpus.glow_device,
                                             samples_device=self.config.device_gpus.samples_device,download=False
                                            )
            
        source_sampler = benchmark.X_sampler
        source_test_sampler = benchmark.X_test_sampler

        source_loader = LoaderFromSampler(source_sampler,batch_size=self.config.scones.training.batch_size,
                                         num_batches=self.config.scones.training.num_batches)

        data_iter = iter(source_loader)
        """
        #===============================#
        target_dataset, _ = get_dataset(self.config.target)
        tgt_dataloader = DataLoader(target_dataset,
                                batch_size=self.config.ncsn.sampling.sources_per_batch,
                                shuffle=True,
                                num_workers=self.config.source.data.num_workers)
        tgt_data_iter = iter(tgt_dataloader)
        (Xt, labels) = next(tgt_data_iter)
         #===============================#
        
        source_dataset, test_source_dataset = get_dataset(self.config.source)
        dataloader = DataLoader(test_source_dataset,
                                batch_size=self.config.ncsn.sampling.sources_per_batch,
                                shuffle=True,
                                num_workers=self.config.source.data.num_workers)
        data_iter = iter(dataloader)
        (Xs, labels) = next(data_iter) # Xs - [B,3,32,32]
        
        #===============================#
        
        test_sampler = LoaderSampler(DataLoader(test_source_dataset, shuffle=False, num_workers=0, batch_size=64), 'cpu')
        valid_images = test_sampler.sample(32)[[22,19,2,13]]
        
        """
        valid_images = test_source_dataset[[10,24,20,25]][0] #[4,3,32,32]
        valid_images = valid_images 
        """
        #===============================#
        print(valid_images.shape,Xs.shape)
        Xs = torch.cat([Xs, valid_images], dim=0) 
        
        
        
        Xs_global = torch.cat([Xs] * self.config.ncsn.sampling.samples_per_source, dim=0).to(self.config.device)
        Xs_global = data_transform(self.config.source, Xs_global) #[B+4,3,32,32]
        #================ DATA ================#
        
         
        
        
        if(hasattr(self.config.ncsn.sampling, "n_sigmas_skip")):
            n_sigmas_skip = self.config.ncsn.sampling.n_sigmas_skip
        else:
            n_sigmas_skip = 0
            
        
        batch_size = self.config.ncsn.sampling.sources_per_batch * self.config.ncsn.sampling.samples_per_source
        total_n_samples = self.config.ncsn.sampling.num_samples4fid
        n_rounds = total_n_samples // batch_size
        result = []
        
        
        if self.config.ncsn.sampling.data_init:
            
            
            dataloader = DataLoader(test_source_dataset,
                                        batch_size=self.config.ncsn.sampling.sources_per_batch,
                                        shuffle=True,
                                        num_workers=self.config.source.data.num_workers)
            data_iter = iter(dataloader)
            
            
           
            
            #============== DATA ==========
            """
            benchmark = get_image_benchmark(batch_size=self.config.scones.training.batch_size,
                                             eps=self.config.transport.coeff,glow_device=self.config.device_gpus.glow_device,
                                             samples_device=self.config.device_gpus.samples_device,download=False
                                            )
            
            source_sampler = benchmark.X_sampler
 

            source_loader = LoaderFromSampler(source_sampler,batch_size=self.config.scones.training.batch_size,
                                         num_batches=self.config.scones.training.num_batches)

            data_iter = iter(source_loader)
            """
            #============== DATA ==========#
            
            
                
            img_id = 0
            iters=0
            
            for r in tqdm(range(n_rounds)):
                
                
                if self.config.ncsn.sampling.data_init:
                    try:
                        init_samples, labels = next(data_iter)
                        
                        #=========================#
                        init_samples = torch.cat([init_samples, valid_images],dim=0)
                        #=========================#
                        
                        init_samples = torch.cat([init_samples] * self.config.ncsn.sampling.samples_per_source, dim=0)
                        
                        labels = torch.cat([labels] * self.config.ncsn.sampling.samples_per_source, dim=0)
                    except StopIteration:
                        data_iter = iter(dataloader)
                        init_samples, labels = next(data_iter)
                        #=========================#
                        init_samples = torch.cat([init_samples, valid_images],dim=0)
                        #=========================#
                        init_samples = torch.cat([init_samples] * self.config.ncsn.sampling.samples_per_source, dim=0)
                        
                        labels = torch.cat([labels] * self.config.ncsn.sampling.samples_per_source, dim=0)

                    init_samples = init_samples.to(self.config.device)
                    init_samples = data_transform(self.config.target, init_samples)
                    
                    baryproj_data_init = False
                    if(baryproj_data_init):
                        with torch.no_grad():
                            bproj_samples = torch.clone(bproj(init_samples)).detach()
                    else:
                        bproj_samples = torch.clone(init_samples).detach()

                    samples = bproj_samples + sigmas_th[n_sigmas_skip] * torch.randn_like(bproj_samples)
                    samples.requires_grad = True
                    samples = samples.to(self.config.device)
                else:
                    
                    samples = torch.rand(batch_size,
                                         self.config.target.data.channels,
                                         self.config.target.data.image_size,
                                         self.config.target.data.image_size, device=self.config.device)
                    init_samples = torch.clone(samples)
                    samples = data_transform(self.config.target, samples)
                    samples.requires_grad = True
                    samples = samples.to(self.config.device)

                
                 
                iters+=1000
                all_samples = anneal_Langevin_dynamics(self.config, samples, Xs_global, Xt, score, cpat, sigmas, iters, 
                                                       self.config.ncsn.sampling.n_steps_each,
                                                       self.config.ncsn.sampling.step_lr,
                                                       verbose=True,
                                                       final_only=self.config.ncsn.sampling.final_only,
                                                       denoise=self.config.ncsn.sampling.denoise,
                                                       n_sigmas_skip=n_sigmas_skip)

                
                
                samples = all_samples[-1]
                torch.save(samples,f"/gpfs/data/gpfs0/n.gushchin/Scon/results/celeba_eps_100/celeba_{r}.pth")
                
                 
            
            
    """       
    def fast_fid(self):
        ### Test the fids of ensembled checkpoints.
        ### Shouldn't be used for pretrained with ema

        if self.config.ncsn.fast_fid.ensemble:
            if self.config.ncsn.model.ema:
                raise RuntimeError("Cannot apply ensembling to pretrained with EMA.")
            self.fast_ensemble_fid()
            return

        from ncsn.evaluation.fid_score import get_fid, get_fid_stats_path
        import pickle

        source_dataset, _ = get_dataset(self.config.source)
        source_dataloader = DataLoader(source_dataset,
                                batch_size=self.config.ncsn.sampling.sources_per_batch,
                                shuffle=True,
                                num_workers=self.config.source.data.num_workers)
        source_iter = iter(source_dataloader)

        score = get_scorenet(self.config.ncsn)
        score = torch.nn.DataParallel(score)

        if self.config.scones.compatibility.ckpt_id is None:    
            cpat_states = torch.load(os.path.join(self.config.compatibility.logging.log_path,
               f'checkpoint_dim_{self.config.source.data.dim}_eps_{self.config.transport.coeff}.pth'),
                                     map_location=self.config.device)
        else:
            cpat_states = torch.load(os.path.join(self.config.compatibility.logging.log_path,
              f'checkpoint_dim_{self.config.source.data.dim}_eps_{self.config.transport.coeff}_step_{self.config.scones.compatibility.ckpt_id}.pth'),
                                     map_location=self.config.device)

            
        cpat = get_compatibility(self.config)
        cpat.load_state_dict(cpat_states[0])

        sigmas_th = get_sigmas(self.config.ncsn)
        sigmas = sigmas_th.cpu().numpy()

        fids = {}
        for ckpt in tqdm.tqdm(range(self.config.ncsn.fast_fid.begin_ckpt, self.config.ncsn.fast_fid.end_ckpt + 1, 5000),
                              desc="processing ckpt"):
            states = torch.load(os.path.join(self.args.log_path, f'checkpoint_{ckpt}.pth'),
                                map_location=self.config.device)

            if self.config.ncsn.model.ema:
                ema_helper = EMAHelper(mu=self.config.ncsn.model.ema_rate)
                ema_helper.register(score)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(score)
            else:
                score.load_state_dict(states[0])

            score.eval()

            num_iters = self.config.ncsn.fast_fid.num_samples // self.config.ncsn.fast_fid.batch_size
            output_path = os.path.join(self.args.image_folder, 'ckpt_{}'.format(ckpt))
            os.makedirs(output_path, exist_ok=True)
            
            iters = 0
            for i in range(num_iters):
                try:
                    (Xs, _) = next(source_iter)
                    Xs_global = torch.cat([Xs] * self.config.ncsn.sampling.samples_per_source, dim=0).to(self.config.device)
                except StopIteration:
                    source_iter = iter(source_dataloader)
                    (Xs, _) = next(source_iter)
                    Xs_global = torch.cat([Xs] * self.config.ncsn.sampling.samples_per_source, dim=0).to(self.config.device)

                init_samples = torch.rand(self.config.ncsn.fast_fid.batch_size, self.config.target.data.channels,
                                          self.config.target.data.image_size, self.config.target.data.image_size,
                                          device=self.config.device)
                init_samples = data_transform(self.config.target, init_samples)
                init_samples.requires_grad = True
                init_samples = init_samples.to(self.config.device)
                
                iters+=1000
                all_samples = anneal_Langevin_dynamics(init_samples, Xs_global, score, cpat, sigmas, iters,
                                                       self.config.ncsn.fast_fid.n_steps_each,
                                                       self.config.ncsn.fast_fid.step_lr,
                                                       verbose=self.config.ncsn.fast_fid.verbose,
                                                       final_only=self.config.ncsn.sampling.final_only,
                                                       denoise=self.config.ncsn.sampling.denoise)

                final_samples = all_samples[-1]
                
                for id, sample in enumerate(final_samples):
                    sample = sample.view(self.config.target.data.channels,
                                         self.config.target.data.image_size,
                                         self.config.target.data.image_size)

                    sample = inverse_data_transform(self.config.target, sample)

                    save_image(sample, os.path.join(output_path, 'sample_{}.png'.format(id)))

            stat_path = get_fid_stats_path(self.args, self.config.ncsn, download=True)
            fid = get_fid(stat_path, output_path)
            fids[ckpt] = fid
            print("ckpt: {}, fid: {}".format(ckpt, fid))

        with open(os.path.join(self.args.image_folder, 'fids.pickle'), 'wb') as handle:
            pickle.dump(fids, handle, protocol=pickle.HIGHEST_PROTOCOL)

    """