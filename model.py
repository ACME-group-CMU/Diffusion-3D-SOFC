import torch
import math
import torch.nn as nn

from diffusers import ModelMixin,UNet2DModel


class ChannelShuffle(nn.Module):
    def __init__(self,groups):
        super().__init__()
        self.groups=groups
    def forward(self,x):
        n,c,h,w,d=x.shape
        x=x.view(n,self.groups,c//self.groups,h,w,d) # group
        x=x.transpose(1,2).contiguous().view(n,-1,h,w,d) #shuffle
        
        return x

class ConvBnSiLu(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0):
        super().__init__()
        self.module=nn.Sequential(nn.Conv3d(in_channels,out_channels,kernel_size,stride=stride,padding=padding),
                                  nn.BatchNorm3d(out_channels),
                                  nn.SiLU())
    def forward(self,x):
        return self.module(x)

class ResidualBottleneck(nn.Module):
    '''
    shufflenet_v2 basic unit(https://arxiv.org/pdf/1807.11164.pdf)
    '''
    def __init__(self,in_channels,out_channels):
        super().__init__()

        self.branch1=nn.Sequential(nn.Conv3d(in_channels//2,in_channels//2,3,1,1,groups=in_channels//2),
                                    nn.BatchNorm3d(in_channels//2),
                                    ConvBnSiLu(in_channels//2,out_channels//2,1,1,0))
        self.branch2=nn.Sequential(ConvBnSiLu(in_channels//2,in_channels//2,1,1,0),
                                    nn.Conv3d(in_channels//2,in_channels//2,3,1,1,groups=in_channels//2),
                                    nn.BatchNorm3d(in_channels//2),
                                    ConvBnSiLu(in_channels//2,out_channels//2,1,1,0))
        self.channel_shuffle=ChannelShuffle(2)

    def forward(self,x):
        x1,x2=x.chunk(2,dim=1)
        x=torch.cat([self.branch1(x1),self.branch2(x2)],dim=1)
        x=self.channel_shuffle(x) #shuffle two branches

        return x

class ResidualDownsample(nn.Module):
    '''
    shufflenet_v2 unit for spatial down sampling(https://arxiv.org/pdf/1807.11164.pdf)
    '''
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.branch1=nn.Sequential(nn.Conv3d(in_channels,in_channels,3,2,1,groups=in_channels),
                                    nn.BatchNorm3d(in_channels),
                                    ConvBnSiLu(in_channels,out_channels//2,1,1,0))
        self.branch2=nn.Sequential(ConvBnSiLu(in_channels,out_channels//2,1,1,0),
                                    nn.Conv3d(out_channels//2,out_channels//2,3,2,1,groups=out_channels//2),
                                    nn.BatchNorm3d(out_channels//2),
                                    ConvBnSiLu(out_channels//2,out_channels//2,1,1,0))
        self.channel_shuffle=ChannelShuffle(2)

    def forward(self,x):
        x=torch.cat([self.branch1(x),self.branch2(x)],dim=1)
        x=self.channel_shuffle(x) #shuffle two branches

        return x

class TimeMLP(nn.Module):
    '''
    naive introduce timestep information to feature maps with mlp and add shortcut
    '''
    def __init__(self,embedding_dim,hidden_dim,out_dim):
        super().__init__()
        self.mlp=nn.Sequential(nn.Linear(embedding_dim,hidden_dim),
                                nn.SiLU(),
                               nn.Linear(hidden_dim,out_dim))
        self.act=nn.SiLU()
    def forward(self,x,t):
        t_emb=self.mlp(t).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x=x+t_emb
  
        return self.act(x)
    
class EncoderBlock(nn.Module):
    def __init__(self,in_channels,out_channels,time_embedding_dim):
        super().__init__()
        self.conv0=nn.Sequential(*[ResidualBottleneck(in_channels,in_channels) for i in range(3)],
                                    ResidualBottleneck(in_channels,out_channels//2))

        self.time_mlp=TimeMLP(embedding_dim=time_embedding_dim,hidden_dim=out_channels,out_dim=out_channels//2)
        self.conv1=ResidualDownsample(out_channels//2,out_channels)
    
    def forward(self,x,t=None):
        x_shortcut=self.conv0(x)
        if t is not None:
            x=self.time_mlp(x_shortcut,t)
            
        x=self.conv1(x)

        return [x,x_shortcut]
        
class DecoderBlock(nn.Module):
    def __init__(self,in_channels,out_channels,time_embedding_dim):
        super().__init__()
        self.upsample=nn.Upsample(scale_factor=2)
        self.conv0=nn.Sequential(*[ResidualBottleneck(in_channels,in_channels) for i in range(3)],
                                    ResidualBottleneck(in_channels,in_channels//2))

        self.time_mlp=TimeMLP(embedding_dim=time_embedding_dim,hidden_dim=in_channels,out_dim=in_channels//2)
        self.conv1=ResidualBottleneck(in_channels//2,out_channels//2)

    def forward(self,x,x_shortcut,t=None):
        x=self.upsample(x)
        x=torch.cat([x,x_shortcut],dim=1)
        x=self.conv0(x)
        if t is not None:
            x=self.time_mlp(x,t)
        x=self.conv1(x)

        return x        

class Unet(UNet2DModel):
    '''
    simple unet design without attention
    '''
    def __init__(self,size,timesteps,time_embedding_dim,in_channels=1,out_channels=1,base_dim=32,dim_mults=[2,4,8,16]):
        super().__init__()
        assert isinstance(dim_mults,(list,tuple))
        assert base_dim%2==0 
        
        self.sample_size = size
        self.in_channels= in_channels
        self.out_channels=out_channels
        channels=self._cal_channels(base_dim,dim_mults)

        self.init_conv=ConvBnSiLu(in_channels,base_dim,3,1,1)
        self.time_embedding=nn.Embedding(timesteps,time_embedding_dim)

        self.encoder_blocks=nn.ModuleList([EncoderBlock(c[0],c[1],time_embedding_dim) for c in channels])
        self.decoder_blocks=nn.ModuleList([DecoderBlock(c[1],c[0],time_embedding_dim) for c in channels[::-1]])
    
        self.mid_block=nn.Sequential(*[ResidualBottleneck(channels[-1][1],channels[-1][1]) for i in range(2)],
                                        ResidualBottleneck(channels[-1][1],channels[-1][1]//2))

        self.final_conv=nn.Conv3d(in_channels=channels[0][0]//2,out_channels=out_channels,kernel_size=1)
        self.act = nn.Tanh()

    def forward(self,x,timestep):
        x=self.init_conv(x)
        
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=x.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(x.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(x.shape[0], dtype=timesteps.dtype, device=timesteps.device)
        
        t=self.time_embedding(timesteps)
        
        
        encoder_shortcuts=[]
        for encoder_block in self.encoder_blocks:
            
            x,x_shortcut=encoder_block(x,t)
            encoder_shortcuts.append(x_shortcut)
        x=self.mid_block(x)
        
        encoder_shortcuts.reverse()
        for decoder_block,shortcut in zip(self.decoder_blocks,encoder_shortcuts):
            x=decoder_block(x,shortcut,t)
        #x=self.act(x)
        x=self.final_conv(x)
        #x = self.act(x)

        return x

    def _cal_channels(self,base_dim,dim_mults):
        dims=[base_dim*x for x in dim_mults]
        dims.insert(0,base_dim)
        channels=[]
        for i in range(len(dims)-1):
            channels.append((dims[i],dims[i+1])) # in_channel, out_channel

        return channels 

class UNet_New(nn.Module):
    def __init__(self, sample_size,in_channels, out_channels, num_timesteps):
        super().__init__()
        
        self.sample_size = sample_size
        self.in_channels = in_channels

        self.down1 = nn.Conv3d(in_channels, 64, kernel_size=3, stride=2, padding=1)
        self.down2 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
        self.down3 = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1)
        self.down4 = nn.Conv3d(256, 512, kernel_size=3, stride=2, padding=1)

        self.up1 = nn.ConvTranspose3d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose3d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up3 = nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up4 = nn.ConvTranspose3d(64, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.relu = nn.LeakyReLU()

        # Embedding layer for timesteps
        self.timestep_embedding = nn.Embedding(num_timesteps, sample_size)

    def forward(self, x, t):
        
        timesteps = t
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=x.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(x.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = self.timestep_embedding(t)
        t = timesteps * torch.ones(x.shape[0], dtype=timesteps.dtype, device=timesteps.device)
        
        x = x + t
        
        x = self.relu(self.down1(x))
        x = self.relu(self.down2(x))
        x = self.relu(self.down3(x))
        x = self.relu(self.down4(x))
        
        
        x = self.up1(x)
        x = self.relu(self.up2(x))
        x = self.relu(self.up3(x))
        x = self.relu(self.up4(x))

        return x