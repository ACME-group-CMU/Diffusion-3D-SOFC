#!/usr/bin/env python
# coding: utf-8

# In[154]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


# In[155]:

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DDIMSampler(nn.Module):
    
    def __init__(self,diff_timesteps,inf_timesteps,eta=0.0,beta_start=0.0001,beta_end=0.02):
        super().__init__()
        
        # For diffusing step (We use the DDPM diffusing process with linear scheduling)
        self.betas = torch.linspace(beta_start,beta_end,diff_timesteps)
        self.alphas = 1- self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas,-1)
        self.one_minus_alphas_cumprod = 1 - self.alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(self.one_minus_alphas_cumprod)
        
        # For denoising step
        
        if inf_timesteps > diff_timesteps:
            raise ValueError('''Number of Inference Steps > Number of Diffusion Steps \U0001F928''')
            
        ## We assume linear distribution of inf timesteps
        self.timesteps = torch.arange(0,diff_timesteps,diff_timesteps//inf_timesteps) + 1
        
        self.ddim_alphas_cumprod =  self.alphas_cumprod[self.timesteps].clone()
        self.ddim_sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas_cumprod)
        
        self.ddim_alphas_cumprod_prev = torch.cat([self.alphas_cumprod[0:1],
                                                   self.alphas_cumprod[self.timesteps[:-1]]])
        
        self.ddim_sigma = eta*torch.sqrt(((1 - self.ddim_alphas_cumprod_prev) / (1 - self.ddim_alphas_cumprod) *
                                (1 - self.ddim_alphas_cumprod / self.ddim_alphas_cumprod_prev)))
        
        self.ddim_sqrt_one_minus_alphas_cumprod = torch.sqrt(1-self.ddim_alphas_cumprod)
        
    def add_noise(self,x,timestep,noise=None):
        
        noise = noise if noise is not None else torch.randn_like(x,device=x.device)
        
        assert noise.shape == x.shape , "Noise is not the same shape as input "
        if timestep.device != 'cpu':
            timestep = timestep.to('cpu')
        
        alpha_t_1 = self.sqrt_alphas_cumprod[timestep].flatten()
        alpha_t_2 = self.sqrt_one_minus_alphas_cumprod[timestep].flatten()
        
        while len(alpha_t_1.shape)<len(x.shape):
            alpha_t_1=alpha_t_1.unsqueeze(-1)
        while len(alpha_t_2.shape)<len(noise.shape):
            alpha_t_2=alpha_t_2.unsqueeze(-1)
        
        alpha_t_1= alpha_t_1.to(x.device)
        alpha_t_2= alpha_t_2.to(x.device)
        
        return alpha_t_1*x + alpha_t_2*noise
    
    @torch.no_grad()
    def sample(self,model,shape):
        
        try:
            device = model.device
        except:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        bs = shape[0]
        
        x = torch.randn(shape,device = device)
        
        timesteps = torch.flip(self.timesteps,[-1])
        
        for i,step in enumerate(tqdm(timesteps)):
            
            index = len(timesteps) - i - 1
            
            ts = x.new_full((bs,),step,dtype=torch.long)
            
            x, pred_x0, e_t = self.p_sample(model,x,ts,index)
        
        return x
    
    @torch.no_grad()
    def p_sample(self,model,x,timestep,index):
        
        e_t = model(x,timestep)
        
        x_prev , pred_x0 = self.get_x_prev_and_predx0(e_t,index,x)
        
        return x_prev,pred_x0,e_t
    
    @torch.no_grad()
    def get_x_prev_and_predx0(self,e_t,index,x):
        
        alpha = self.ddim_alphas_cumprod[index]
        alpha_prev = self.ddim_alphas_cumprod_prev[index]
        sigma = self.ddim_sigma[index]
        sqrt_one_minus_alpha = self.ddim_sqrt_one_minus_alphas_cumprod[index]
        
        
        pred_x0 = (x-sqrt_one_minus_alpha*e_t)/(alpha**0.5)
        
        dir_xt = (1-alpha_prev- sigma**2).sqrt() * e_t
        
        if sigma == 0.:
            noise = 0.
        else:
            noise = torch.randn_like(x,device=x.device)
        
        x_prev = (alpha_prev**0.5)*pred_x0 + dir_xt + sigma*noise
        
        return x_prev,pred_x0


# In[ ]:




