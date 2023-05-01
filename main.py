import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader,random_split
from torch.optim import AdamW

import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks.progress import TQDMProgressBar


from data import Microstructures
from model import Unet, UNet_New
#from new_model import SimpleUnet
from diffusers import DDIMScheduler
from pipeline import DDIMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup

#import warnings
#warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

#Data Saving Parameters
parser.add_argument("--dir", type=str, default='./results', help="directory that saves all the logs")
parser.add_argument("--data_path", type=str, default='greyscale.npz', help="file name where the data belongs")
# Model training 
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
parser.add_argument("--warmup_steps", type=int, default=500, help="scheduler warmup steps")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--n_gpu", type=int, default=1, help="number of gpu to use during training")
parser.add_argument("--n_nodes", type=int, default=1, help="number of nodes")
parser.add_argument("--clip_value", type=float, default=0.1, help="lower and upper clip value")
parser.add_argument("--divide_batch", type=bool, default=True, help="if batch_size needs to be divided for distributed training")

#Model and Data
parser.add_argument("--data_length", type=int, default=20000, help="number of random data points")
parser.add_argument("--img_size", type=int, default=64, help="generated image size cubic dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--den_timesteps", type=int, default=1000, help="number of noising timesteps")
parser.add_argument("--inf_timesteps", type=int, default=50, help="number of denoising timesteps")
parser.add_argument("--time_dim", type=int, default=128, help="time embedding dimension in the UNet")
parser.add_argument("--base_dim", type=int, default=32, help="base dimension in the UNet")
parser.add_argument("--sample_interval", type=int, default=100, help="interval betwen image samples")
parser.add_argument("--sample_size", type=int, default=36, help="number of samples that are generated")
parser.add_argument("--apply_sym", type=bool, default=True, help="if symmetry operations need to be applied during sampling from data")


args = parser.parse_args()

def main():
    
    global args
    
    if args.n_epochs < 5*args.sample_interval:
        args.sample_interval = args.n_epochs//6
    else:
        pass
    
    if (args.n_gpu*args.n_nodes>1) and args.divide_batch:
        args.batch_size = args.batch_size//(args.n_gpu*args.n_nodes)
    else:
        pass
    
    print(args)
    
    seed_everything(42, workers=True)
    image_size = [args.img_size]*3
    
    dm = MicroData(
                   data_path=args.data_path,
                   img_size=image_size,
                   data_length=args.data_length,
                   apply_sym = args.apply_sym,
                   batch_size=args.batch_size,
                   num_workers=args.n_cpu
                  )
        
    model = GAN(
                args.channels,
                *image_size,
                args.base_dim,
                args.time_dim,
                args.lr,
                args.sample_interval,
                args.clip_value,
                args.den_timesteps,
                args.inf_timesteps,
                args.sample_size,
                args.warmup_steps
                )
    
    if args.n_nodes>1:
        strategy = 'ddp'
    else:
        strategy = None
    
    trainer = Trainer(
        default_root_dir=args.dir,
        accelerator="auto",
        devices=args.n_gpu,
        num_nodes = args.n_nodes,
        max_epochs=args.n_epochs,
        strategy = strategy,
        deterministic = True,
        callbacks=[TQDMProgressBar(refresh_rate=(args.data_length//(20*args.batch_size)))],
        gradient_clip_val=args.clip_value
    )
    
    trainer.fit(model, dm)


class MicroData(LightningDataModule):
    def __init__(
        self,
        data_path: str = '.',
        img_size = (64,64,64),
        data_length = 10000,
        apply_sym = True,
        batch_size: int = 256,
        num_workers: int = 1,
    ):
        super().__init__()
        
        self.data_dir = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.length = data_length
        self.apply_symmetry = apply_sym
        
    def setup(self,stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            
            data_full = Microstructures(self.data_dir,
                                        self.img_size,
                                        self.length,
                                        apply_symmetry=self.apply_symmetry)
            
            
            self.data_train, self.data_val = random_split(data_full, 
                                                          [int(self.length*(8/10)),int(self.length*(2/10))])

    def train_dataloader(self):
        
        return DataLoader(self.data_train,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        
        return DataLoader(self.data_val, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers)
    
    
class GAN(LightningModule):
    def __init__(
        self,
        channels,
        width,
        height,
        depth,
        base_dim,
        time_dim,
        lr: float = 0.0001,
        save_freq : int = 20,
        clip_value : int = 0.01,
        den_timesteps : int = 1000,
        inf_timesteps : int = 50,
        sample_amt : int = 36,
        warmup_steps : int = 500,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.clip_value = clip_value
        self.save_freq = save_freq
        
        # network
        self.unet = Unet(size=height,timesteps=den_timesteps,time_embedding_dim=time_dim,base_dim=base_dim,dim_mults=[2,4])
        
        #self.unet = UNet_New(height,1,1,den_timesteps)
        
        self.noise_scheduler = DDIMScheduler(num_train_timesteps=den_timesteps)
        self.pipeline = DDIMPipeline(unet = self.unet, scheduler = self.noise_scheduler)
        self.sample_amt = sample_amt

    def mse_loss(self, y_hat, y):
        return F.mse_loss(y_hat.flatten(), y.flatten())

    def training_step(self, batch, batch_idx):
        imgs = batch

        # sample noise
        noise = torch.randn(imgs.shape).to(imgs.device)
        bs = imgs.shape[0]
        
        timesteps = torch.randint(0,self.hparams.den_timesteps,(bs,),device=imgs.device).long()
        
        noisy_imgs = self.noise_scheduler.add_noise(imgs,noise,timesteps)
        noise_pred = self.unet(noisy_imgs,timesteps)
        
        loss = self.mse_loss(noise_pred,noise)        
        self.log("loss", loss, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        
        lr = self.hparams.lr
        warmup = self.hparams.warmup_steps
        
        optimizer = AdamW(self.unet.parameters(), lr=lr)
        scheduler_lr = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=warmup,
                                                      num_training_steps = self.trainer.estimated_stepping_batches)
        
        return [[optimizer],[scheduler_lr]]
    
    def lr_scheduler_step(self, *args):
        if self.trainer.global_step > self.hparams.warmup_steps:
            super().lr_scheduler_step(*args)
    
    def training_epoch_end(self,training_step_outputs):
        
        if (self.current_epoch)%self.save_freq==0:
            sample_imgs = self.pipeline(batch_size=self.sample_amt,
                               generator = torch.manual_seed(42),output_type='np.array')
            np.save(f'{self.logger.log_dir}/{self.current_epoch}.npy',sample_imgs)
            

if __name__ == '__main__':
    main()