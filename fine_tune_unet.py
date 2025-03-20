import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import torch
import torchvision
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from diffusers import DDPMScheduler
from transformers import CLIPTokenizer
import torch
checkpoint = '../diffsion_from_scratch.params/'

scheduler = DDPMScheduler.from_pretrained(checkpoint, subfolder='scheduler')
tokenizer = CLIPTokenizer.from_pretrained(checkpoint, subfolder='tokenizer')

scheduler, tokenizer

info=pd.read_csv("./output.csv",sep=',')

compose = torchvision.transforms.Compose([
    torchvision.transforms.Resize(
        512, interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
    torchvision.transforms.CenterCrop(512),
    #torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5], [0.5]),
])
class MyDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform

    def __getitem__(self, index):
        img_path = "./saved_frames2/"+str(self.df['img'].iloc[index])+".png"
  
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        text = self.df['text'].iloc[index]
        input_ids = tokenizer.encode_plus(text,
                                        padding='max_length',
                                        truncation=True,
                                        max_length=77).input_ids
        return {'pixel_values': img, 'input_ids': torch.tensor(input_ids,dtype=torch.long)}

    def __len__(self):
        return len(self.df)
dataset = MyDataset(info,compose)
loader = torch.utils.data.DataLoader(dataset,
                                     shuffle=True,
                                     batch_size=1)


from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel


encoder = CLIPTextModel.from_pretrained(checkpoint, subfolder='text_encoder')
vae = AutoencoderKL.from_pretrained(checkpoint, subfolder='vae')
unet = UNet2DConditionModel.from_pretrained(checkpoint, subfolder='unet')

vae.requires_grad_(False)
encoder.requires_grad_(False)
unet.enable_gradient_checkpointing()
from diffusers.optimization import get_cosine_schedule_with_warmup
optimizer = torch.optim.AdamW(unet.parameters(),
                              lr=1e-5,
                              betas=(0.9, 0.999),
                              weight_decay=0.01,
                              eps=1e-8)
# lr_scheduler = get_cosine_schedule_with_warmup(
#     optimizer=optimizer,
#     num_warmup_steps=30,
#     num_training_steps=(len(loader ) * 150),
# )
criterion = torch.nn.MSELoss()
def get_loss(data):
    device = data['input_ids'].device

  
    #[1, 77] -> [1, 77, 768]
    out_encoder = encoder(data['input_ids'])[0]

 
    #[1, 3, 512, 512] -> [1, 4, 64, 64]
    out_vae = vae.encode(data['pixel_values']).latent_dist.sample()
    #0.18215 = vae.config.scaling_factor
    out_vae = out_vae * 0.18215

 
    noise = torch.randn_like(out_vae)


    #1000 = scheduler.num_train_timesteps
    #1 = out_vae.shape[0]
    noise_step = torch.randint(0, 1000, (1, )).long()
    noise_step = noise_step.to(device)
    out_vae_noise = scheduler.add_noise(out_vae, noise, noise_step)


    out_unet = unet(out_vae_noise, noise_step, out_encoder).sample

    #计算mse loss
    #[1, 4, 64, 64],[1, 4, 64, 64]
    return criterion(out_unet, noise)

from diffusers import StableDiffusionPipeline
def get_cur_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    unet.to(device)
    encoder.to(device)
    vae.to(device)
    unet.train()
    best=100
    loss_sum = 0
    for epoch in range(150):
        for i, data in enumerate(loader):
            for k in data.keys():
                data[k] = data[k].to(device)

            loss = get_loss(data) 
            optimizer.zero_grad()
            loss.backward()
            loss_sum += loss.item()
            if i % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tlr={}'.format(epoch+1,i ,len(loader.dataset),100. * i / len(loader),loss.item(),get_cur_lr(optimizer)))
    #        if (epoch * len(loader) + i) % 4 == 0:
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
                #lr_scheduler.step()

        if epoch % 1 == 0:
            print(epoch, loss_sum)
            if loss_sum<=best:
                best=loss_sum
                StableDiffusionPipeline.from_pretrained(checkpoint, text_encoder=encoder, vae=vae,unet=unet).save_pretrained(f'./save_shixu')
            loss_sum = 0





train()
