import torch
import torch.nn as nn
import torch.distributions as D
import numpy as np
import sklearn.mixture

class MixtureGaussianScore(torch.nn.Module):
    
    def __init__(self, config, target_sampler):
        super(MixtureGaussianScore,self).__init__()
        
        self.config = config
        self.target_sampler = target_sampler
        self.gmm_sklearn = sklearn.mixture.GaussianMixture(n_components=config.eval_gmm.n_components)
        
        self.target_samples = target_sampler.sample(config.eval_gmm.batch_size)
        self.gmm_sklearn.fit(self.target_samples.detach().cpu().numpy())
        self.means = self.gmm_sklearn.means_
        self.covs = self.gmm_sklearn.covariances_
        self.weights = self.gmm_sklearn.weights_
        
        mix = D.Categorical(torch.tensor(self.weights).to(config.device))
        comp = D.MultivariateNormal(torch.tensor(self.means).to(config.device),  torch.tensor(self.covs).to(config.device))
        self.gmm_prob = D.MixtureSameFamily(mix, comp)
        
    def forward(self, x):
        x = x.reshape(-1,self.config.target.data.dim)
        log_prob = self.gmm_prob.log_prob(x)
        return log_prob