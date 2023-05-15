import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm
import util
"""
import sys
sys.path.append('..')
from eot_benchmark.rotated_gaussian_benchmark import get_rotated_gaussian_sampler
from eot_benchmark.rotated_gaussian_benchmark import get_rotated_gaussian_benchmark_stats
from EntropicOTBenchmark.benchmark.metrics import compute_BW_UVP_with_gt_stats
from eot_benchmark.gaussian_mixture_benchmark import (
    get_guassian_mixture_benchmark_sampler,
    get_guassian_mixture_benchmark_ground_truth_sampler, 
)
from eot_benchmark.metrics import (
    compute_BW_UVP_by_gt_samples, calculate_gm_mmd
)
"""


def anneal_Langevin_dynamics(config, tgt, source, Xt, scorenet, cpat, sigmas, iters,  n_steps_each=200, step_lr=0.000008,
                             final_only=False, verbose=False, denoise=True, n_sigmas_skip=0):
   
    images = []
    verbose=False
    
    init_samples=  tgt 

    for c, sigma in tqdm(enumerate(sigmas)):
        
        
        if(c < n_sigmas_skip):
            continue
            
        labels = torch.ones(tgt.shape[0], device=tgt.device) * c
        labels = labels.long()
         
        step_size = step_lr * (sigma / sigmas[-1]) ** 2
        
        for s in range(n_steps_each):
            with torch.no_grad():
                grad = scorenet(tgt, labels)
                noise = torch.randn_like(tgt)
            cpat_grad = cpat.score(source, tgt)

            grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
            cpat_grad_norm = torch.norm(cpat_grad.view(cpat_grad.shape[0], -1), dim=-1).mean()

            tgt = tgt + step_size * (grad + 0.65*cpat_grad) + noise * np.sqrt(step_size * 2) 
            
            image_norm = torch.norm(tgt.view(tgt.shape[0], -1), dim=-1).mean()
            snr = np.sqrt(step_size / 2.) * grad_norm / noise_norm
            grad_mean_norm = torch.norm(grad.mean(dim=0).view(-1)) ** 2 * sigma ** 2
            
            
            if not final_only:
                images.append(tgt.to('cpu'))
            
            """
            if verbose:
                print("level: {}, noise level:{}, step_size: {}, grad_norm: {}, cpat_grad_norm: {}, image_norm: {}, snr: {}, grad_mean_norm: {}".format(
                    c, round(sigma, 4), step_size, grad_norm.item(), cpat_grad_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()))
            """    
            
            """
            if c%config.scones.training.snapshot_iter == 0 or sigma==sigmas[-1] :
              
                fig, ax = plt.subplots(4,4,figsize=(25,25),dpi=150)
                
                for i in range(1,5):
                    ax[0,i-1].imshow(source[-i].permute(1,2,0).detach().cpu().numpy())
                    ax[1,i-1].imshow(init_samples[-i].permute(1,2,0).detach().cpu().numpy())
                    ax[2,i-1].imshow(tgt[-i].permute(1,2,0).detach().cpu().numpy())
                    ax[3,i-1].imshow(Xt[-i].permute(1,2,0).detach().cpu().numpy())
                   
                    ax[0,i-1].set_xticks([]);ax[0,i-1].set_yticks([]);
                    ax[1,i-1].set_xticks([]);ax[1,i-1].set_yticks([]);
                    ax[2,i-1].set_xticks([]);ax[2,i-1].set_yticks([]);
                    ax[3,i-1].set_xticks([]);ax[3,i-1].set_yticks([]);
                    
                    
                    ax[0,0].set_ylabel(r"$x \sim \mathbb{P}(x)$",fontsize=45)
                    ax[1,0].set_ylabel(r"$x_{0}$", fontsize=45)
                    ax[2,0].set_ylabel(r"$T(x_{0})$",fontsize=45)
                    ax[3,0].set_ylabel(r"$y \sim \mathbb{Q}(y)$",fontsize=45)
                    
                fig.tight_layout()
                wandb.log({" Scones: samples ": [wandb.Image(util.fig2img(fig))]},step=iters+c+config.compatibility.training.n_iters + config.baryproj.training.n_iters+1)
            """

                
    if denoise:
        cpat_score = cpat.score(source, tgt).detach()
        with torch.no_grad():
            last_noise = (len(sigmas) - 1) * torch.ones(tgt.shape[0], device=tgt.device)
            last_noise = last_noise.long()
            tgt = tgt + sigmas[-1] ** 2 * (scorenet(tgt, last_noise)  + 0.65*cpat_score ) 
            images.append(tgt[:-4,:,:,:].to('cpu'))
            
    
    torch.save(tgt[-4:,:,:,:].to('cpu'),
               f"/gpfs/data/gpfs0/n.gushchin/Scon/results/paper_pics_celeba_eps_100/pics_{int(iters)}.pth")
    
    if final_only:
        return [tgt[:-4,:,:,:].to('cpu')]
    else:
        return images


@torch.no_grad()
def anneal_Langevin_dynamics_inpainting(x_mod, refer_image, scorenet, sigmas, image_size,
                                        n_steps_each=100, step_lr=0.000008):
    """
    Currently only good for 32x32 images. Assuming the right half is missing.
    """

    images = []

    refer_image = refer_image.unsqueeze(1).expand(-1, x_mod.shape[1], -1, -1, -1)
    refer_image = refer_image.contiguous().view(-1, 3, image_size, image_size)
    x_mod = x_mod.view(-1, 3, image_size, image_size)
    cols = image_size // 2
    half_refer_image = refer_image[..., :cols]
    with torch.no_grad():
        for c, sigma in enumerate(sigmas):
            if(c < n_sigma_skip):
                continue
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2

            for s in range(n_steps_each):
                images.append(x_mod.to('cpu'))
                corrupted_half_image = half_refer_image + torch.randn_like(half_refer_image) * sigma
                x_mod[:, :, :, :cols] = corrupted_half_image
                noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)
                grad = scorenet(x_mod, labels)
                x_mod = x_mod + step_size * grad + noise
                print("class: {}, step_size: {}, mean {}, max {}".format(c, step_size, grad.abs().mean(),
                                                                         grad.abs().max()))

        return images

@torch.no_grad()
def anneal_Langevin_dynamics_interpolation(x_mod, scorenet, sigmas, n_interpolations, n_steps_each=200, step_lr=0.000008,
                             final_only=False, verbose=False):
    images = []

    n_rows = x_mod.shape[0]

    x_mod = x_mod[:, None, ...].repeat(1, n_interpolations, 1, 1, 1)
    x_mod = x_mod.reshape(-1, *x_mod.shape[2:])

    for c, sigma in enumerate(sigmas):
        labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
        labels = labels.long()
        step_size = step_lr * (sigma / sigmas[-1]) ** 2
        for s in range(n_steps_each):
            grad = scorenet(x_mod, labels)

            noise_p = torch.randn(n_rows, x_mod.shape[1], x_mod.shape[2], x_mod.shape[3],
                                  device=x_mod.device)
            noise_q = torch.randn(n_rows, x_mod.shape[1], x_mod.shape[2], x_mod.shape[3],
                                  device=x_mod.device)
            angles = torch.linspace(0, np.pi / 2., n_interpolations, device=x_mod.device)

            noise = noise_p[:, None, ...] * torch.cos(angles)[None, :, None, None, None] + \
                        noise_q[:, None, ...] * torch.sin(angles)[None, :, None, None, None]

            noise = noise.reshape(-1, *noise.shape[2:])
            grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
            image_norm = torch.norm(x_mod.view(x_mod.shape[0], -1), dim=-1).mean()

            x_mod = x_mod + step_size * grad + noise * np.sqrt(step_size * 2)

            snr = np.sqrt(step_size / 2.) * grad_norm / noise_norm

            if not final_only:
                images.append(x_mod.to('cpu'))
            if verbose:
                print(
                    "level: {}, step_size: {}, image_norm: {}, grad_norm: {}, snr: {}".format(
                        c, step_size, image_norm.item(), grad_norm.item(), snr.item()))


    if final_only:
        return [x_mod.to('cpu')]
    else:
        return images


def Langevin_dynamics(config,tgt, source, scorenet, cpat, n_steps, step_lr=0.000008, final_only=False, sample_every=1, verbose=False):
    images = []
    
    step_lr = np.linspace(step_lr,0.000015,n_steps)
    
    
    if config.meta.problem_name.startswith("gaussian"):
        target_sampler =  get_rotated_gaussian_sampler("target",config.target.data.dim, with_density=False,
                                         batch_size= config.scones.training.batch_size, device="cpu") 
        y_samples = target_sampler.sample()
        
    elif config.meta.problem_name.startswith("mixgauss"):
        target_sampler = get_guassian_mixture_benchmark_sampler ('target',   config.target.data.dim,  config.transport.coeff,  config.scones.training.batch_size,  device="cpu", download=False)
        y_samples = target_sampler.sample(config.scones.training.batch_size)
    else:
        raise NotImplementedError
    
    
    stats = get_rotated_gaussian_benchmark_stats(config.target.data.dim, config.transport.coeff, device='cpu' )
    mu_X, mu_Y, covariance_X, covariance_Y, optimal_plan_mu, optimal_plan_covariance = stats
    
    for s, lr in tqdm(zip(range(n_steps),step_lr)):
        #with.torch_no_grad
        
        tgt_ = tgt.reshape(-1,config.target.data.dim)
        #tgt_.requires_grad=True
        log_prob = scorenet.forward(tgt_) 
        grad = torch.autograd.grad(log_prob.sum(),tgt_)[0].detach() 
        grad = grad.reshape(-1,config.target.data.dim,1,1)
        noise = 1.3*torch.randn_like(tgt)
        #with.torch no grad
        cpat_grad = cpat.score(source, tgt)

        grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
        cpat_grad_norm = torch.norm(cpat_grad.view(cpat_grad.shape[0], -1), dim=-1).mean()
        
        
        tgt = tgt + lr * (grad  + cpat_grad  ) + noise * np.sqrt( lr * 2)#cpat_Grad  
        
        if not final_only and (s % sample_every == 0):
            images.append(tgt.to('cpu'))
        """
        if verbose:
            cov = np.cov(tgt.detach().cpu().numpy().reshape((-1, 2)), rowvar=False)
            print("step: {}, grad_norm: {}, cpat_grad_norm: {}, cov: {}".format(s, grad_norm.item(), cpat_grad_norm.item(), cov))
        """
        
        
        
        #==================================================#
        
        if s%config.scones.sampling.snapshot_freq==0 and config.meta.problem_name.startswith("gaussian"):
            model_samples = tgt.reshape(-1,config.target.data.dim).detach().cpu().numpy()
            init_samples = source.reshape(-1,config.target.data.dim).detach().cpu().numpy()
            concat_samples = np.concatenate([init_samples,model_samples],axis=1)
            
            bw_uvp_terminal = compute_BW_UVP_with_gt_stats(model_samples,
                                         true_samples_mu=mu_Y, true_samples_covariance=covariance_Y)

            bw_uvp_plan =  compute_BW_UVP_with_gt_stats(concat_samples,
                                             true_samples_mu=optimal_plan_mu, true_samples_covariance=optimal_plan_covariance)
            
            wandb.log({"scones: BW UVP":bw_uvp_terminal },step=s+config.compatibility.training.n_iters + config.baryproj.training.n_iters + 1)
            wandb.log({"scones: BW UVP plan":bw_uvp_plan},step=s+config.compatibility.training.n_iters + config.baryproj.training.n_iters + 1)
            
            
            fig,axes = plt.subplots(1,4,figsize=(12,4),dpi=150,sharex=True,sharey=True)
            axes[0].scatter(init_samples[:,0],init_samples[:,1],c='yellowgreen',edgecolor='black',
                            s=30,label='init samples')
            axes[1].scatter(model_samples[:,0],model_samples[:,1],c='purple',edgecolor='black',
                            s=30,label='model samples')
            axes[2].scatter(y_samples[:,0],y_samples[:,1],c='salmon',edgecolor='black',
                            s=30,label='target samples')
            axes[3].scatter(y_samples[:,0],y_samples[:,1],c='salmon',edgecolor='black',
                            s=30,label='target')
            axes[3].scatter(model_samples[:,0],model_samples[:,1],c='purple',edgecolor='black',
                            s=30,label='model samples')
            
            axes[0].legend()
            axes[1].legend()
            axes[2].legend()
            axes[3].legend()
            fig.tight_layout(pad=0.5)
            wandb.log({" Scones: plot samples ": [wandb.Image(util.fig2img(fig))]},step=s+config.compatibility.training.n_iters + config.baryproj.training.n_iters+1)
        
        elif s%config.scones.sampling.snapshot_freq==0 and config.meta.problem_name.startswith("mixgauss"):
            
            xn = tgt.reshape(-1,config.target.data.dim).detach().cpu() 
            x = source.reshape(-1,config.target.data.dim).detach().cpu() 
            #concat_samples = np.concatenate([init_samples,model_samples],axis=1)
            y = y_samples
            
            ground_truth_plan_sampler = get_guassian_mixture_benchmark_ground_truth_sampler(config.source.data.dim,eps=config.transport.coeff,
                                                          batch_size=config.scones.training.batch_size , device="cpu",download=False)
            x0_gt, xn_gt = ground_truth_plan_sampler.sample(config.metrics.samp_metrics)
            x0_xn_gt = torch.cat((x0_gt, xn_gt), dim=1)
            x0_xn = torch.cat((x, xn), dim=1)
            
            bw_uvp_target = compute_BW_UVP_by_gt_samples(xn.cpu().numpy(), y.cpu().numpy())
            bw_uvp_plan = compute_BW_UVP_by_gt_samples(x0_xn.cpu().numpy(), x0_xn_gt.cpu().numpy())
            
            
            
            
            x0_gt, xn_gt = ground_truth_plan_sampler.sample(config.metrics.samp_metrics)
            x0_xn_gt = torch.cat((x0_gt, xn_gt), dim=1).to('cpu')
            x0_xn = torch.cat(( x, xn), dim=1).to('cpu')
            
            
            mmd_plan = calculate_gm_mmd(x0_xn, x0_xn_gt, dim=config.source.data.dim, eps=config.transport.coeff,
                                        normalization_type="indep_plan_rbf_kernel")
            
            mmd_plan_dim_normalized = calculate_gm_mmd(x0_xn, x0_xn_gt, dim=config.source.data.dim, eps=config.transport.coeff,
                                                       normalization_type="indep_plan_rbf_kernel_dim_norm")
            
            mmd_plan_distance_kernel = calculate_gm_mmd(x0_xn, x0_xn_gt, dim=config.source.data.dim, eps=config.transport.coeff,
                                                        normalization_type="indep_plan_rbf_distance_kernel")
            
            mmd_target = calculate_gm_mmd(xn, y, dim=config.source.data.dim, eps=config.transport.coeff, normalization_type="identity_rbf_kernel")
            
            mmd_target_dim_normalized = calculate_gm_mmd(xn, y, dim=config.source.data.dim, eps=config.transport.coeff,
                                                         normalization_type="indentity_rbf_kernel_norm")
            mmd_target_distance_kernel = calculate_gm_mmd(xn, y, dim=config.source.data.dim, eps=config.transport.coeff, 
                                                          normalization_type="identity_distance_kernel")
                    
             
                    
            #logging to wandb
                     
            wandb.log({f'BW_UVP_Target_forward': bw_uvp_target},step=s+config.compatibility.training.n_iters + config.baryproj.training.n_iters + 1)
            wandb.log({f'BW_UVP_plan_forward': bw_uvp_plan},step=s+config.compatibility.training.n_iters + config.baryproj.training.n_iters + 1) 


            wandb.log({f'MMD_plan_forward': mmd_plan},step=s+config.compatibility.training.n_iters + config.baryproj.training.n_iters + 1)
            wandb.log({f'MMD_plan_dim_norm_forward': mmd_plan_dim_normalized},step=s+config.compatibility.training.n_iters + config.baryproj.training.n_iters + 1)
            wandb.log({f'MMD_plan_distance_kernel_forward': mmd_plan_distance_kernel},step=s+config.compatibility.training.n_iters + config.baryproj.training.n_iters + 1)

            wandb.log({f'MMD_forward': mmd_target},step= s+config.compatibility.training.n_iters + config.baryproj.training.n_iters+1)
            wandb.log({f'MMD_dim_normalized_forward': mmd_target_dim_normalized},step=s+config.compatibility.training.n_iters + config.baryproj.training.n_iters + 1)
            wandb.log({f'MMD_distance_kernel_forward': mmd_target_distance_kernel},step=s+config.compatibility.training.n_iters + config.baryproj.training.n_iters + 1)
            
            #plotting to wandb
            
            if config.source.data.dim == 2:
                fig,axes = plt.subplots(1,4,figsize=(12,4),dpi=150,sharex=True,sharey=True)
                axes[0].scatter(x[:,0],x[:,1],c='yellowgreen',edgecolor='black',
                                s=30,label='init samples')
                axes[1].scatter(xn[:,0],xn[:,1],c='purple',edgecolor='black',
                                s=30,label='model samples')
                axes[2].scatter(y[:,0],y[:,1],c='salmon',edgecolor='black',
                                s=30,label='target samples')
                axes[3].scatter(y[:,0],y[:,1],c='salmon',edgecolor='black',
                                s=30,label='target')
                axes[3].scatter(xn[:,0],xn[:,1],c='purple',edgecolor='black',
                                s=30,label='model samples')

                axes[0].legend()
                axes[1].legend()
                axes[2].legend()
                axes[3].legend()
                fig.tight_layout(pad=0.5)
                wandb.log({" Scones: plot samples ": [wandb.Image(util.fig2img(fig))]},step=s+config.compatibility.training.n_iters + config.baryproj.training.n_iters+1)
                
            elif not config.meta.problem_name.startswith("images"):
                
                util.pca_plot(y_samples, xn, s+config.compatibility.training.n_iters + config.baryproj.training.n_iters+1)
                    
            else:
                raise NotImplementedError
                

         
            
    if final_only:
        return [tgt.to('cpu')]
    else:
        return images

def joint_marginal_Langevin_dynamics(tgt, source, scorenet, cpat, n_steps, step_lr=0.000008, final_only=False, sample_every=1, verbose=False):
    print("SAMPLING JOINT MARGINAL LANGEVIN")
    images = []

    for s in range(n_steps):
        with torch.no_grad():
            noise = torch.randn_like(tgt)

        cpat_grad = cpat.score(source, tgt)

        cpat_grad_norm = torch.norm(cpat_grad.view(cpat_grad.shape[0], -1), dim=-1).mean()

        tgt = tgt + step_lr * (cpat_grad) + noise * np.sqrt(step_lr * 2)

        if not final_only and (s % sample_every == 0):
            images.append(tgt.to('cpu'))
        if verbose:
            cov = np.cov(tgt.detach().cpu().numpy().reshape((-1, 2)), rowvar=False)
            print("step: {}, cpat_grad_norm: {}, cov: {}".format(s, cpat_grad_norm.item(), cov))

    if final_only:
        return [tgt.to('cpu')]
    else:
        return images

def compare_Langevin_dynamics(tgt, source, scorenet, cpat, n_steps, step_lr=0.000008, final_only=False,
                                  sample_every=1, verbose=False):
        images = []
        cpat_trained, cpat_true = cpat

        for s in range(n_steps):
            with torch.no_grad():
                grad = scorenet(tgt)
                noise = torch.randn_like(tgt)
            cpat_grad_r, cpat_grad_s = cpat_true.score(source, tgt)
            cpat_grad = cpat_grad_r + cpat_grad_s
            est_cpat_grad = cpat_trained.score(source, tgt)

            grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
            cpat_grad_norm = torch.norm(cpat_grad.view(cpat_grad.shape[0], -1), dim=-1).mean()
            est_vs_true_norm = torch.norm((cpat_grad - est_cpat_grad).view(cpat_grad.shape[0], -1), dim=-1).mean()
            tgt = tgt + step_lr * (grad + est_cpat_grad) + noise * np.sqrt(step_lr * 2)

            if not final_only and (s % sample_every == 0):
                images.append(tgt.to('cpu'))
            if verbose:
                cov = np.cov(tgt.detach().cpu().numpy().reshape((-1, 2)), rowvar=False)
                print("step: {}, grad_norm: {}, cpat_grad_norm: {}, est vs true error: {}, cov: {}".format(s,
                                            grad_norm.item(), cpat_grad_norm.item(), est_vs_true_norm.item(), cov))

        if final_only:
            return [tgt.to('cpu')]
        else:
            return images

