"""
Inverse an image to the latent space of StyleGAN3
"""
import os
from PIL import Image
import os.path as osp
import numpy as np
import pickle as pkl
from tqdm import tqdm
import dnnlib
from dnnlib.util import Logger
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import click
sys.path.append('/home/user/code/stylegan2-ada-pytorch')
import legacy
import pyspng
import torchvision.transforms as T
import pytorch_msssim

from crtiterions import LPIPS
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

class Inversor():
    def __init__(self, stylegan3_path, logpath):
        self.device = torch.device('cuda')
        self.load_G(stylegan3_path)
        self.set_criterions()
        self.logger = Logger()


    def process_image(self,img, img2tensor=True):
        """
        @img2tensor
            - False: inverse tensor to array, which can be saved directly by PIL
            - True: convert array to tensor
        """
        if img2tensor:
            with open(img, 'rb') as f:
                if os.path.splitext(img)[1].lower() == '.png':
                    image = pyspng.load(f.read())
                else:
                    image = np.array(Image.open(f))
            image = image/255*2 -1
            image = torch.from_numpy(image).permute(2,0,1).unsqueeze(0).float().to(self.device)
            return image
        else:
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            image = Image.fromarray(img[0].cpu().numpy(), 'RGB')
            return image

    def set_criterions(self,):
        self.MSE_C = nn.MSELoss().to(self.device)
        self.LPIPS_C = LPIPS(net_type='vgg').to(self.device).eval()
        self.SSIM = pytorch_msssim.ms_ssim


    def load_G(self, stylegan3_path):
        # with open(stylegan3_path, 'rb') as f:
        #     self.G = pkl.load(f)['G_ema']
        #     self.G.to(self.device)
            
        ckpt_paths = {
            'cat': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqcat.pkl',
            'dog': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqdog.pkl',
            'wild': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqwild.pkl',
            'brecahad': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/brecahad.pkl',
            'cifar10': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl',
            'ffhq': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl',
            'metfaces': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl'
        }
        network_pkl = ckpt_paths['ffhq']
        print('Loading networks from "%s"...' % network_pkl)
        with dnnlib.util.open_url(network_pkl) as f:
            self.G = legacy.load_network_pkl(f)['G_ema'].to(self.device) # type: ignore

    def resize_image(self, image, pixel=256):
        # resized_image = F.adaptive_avg_pool2d(image, (pixel, pixel))
        # resized_image = F.interpolate(image, (pixel, pixel), mode='area')
        resized_image = F.interpolate(image, (pixel, pixel), mode='bicubic')
        return resized_image


    def inverse(self, imgname, total_step, lr, mse_w, lpip_w, truncation_psi, noise_mode, space='z'):
        gt_image = self.process_image(imgname)
        # gt_image = T.Resize((512, 512))(gt_image) # cat
        gt_image = T.Resize((1024, 1024))(gt_image)
        label = torch.zeros([1, self.G.c_dim]).to(self.device)
        assert space in ['z', 'wp']
        code = None
        if space == 'z':
            code = torch.from_numpy(np.random.randn(1, self.G.z_dim)).to(self.device)
        elif space == 'wp':
            code = self.G.mapping.w_avg.clone().detach().unsqueeze(0).unsqueeze(0).repeat([1, self.G.num_ws, 1])
        code.requires_grad = True
        optimizer = torch.optim.AdamW([code], lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.99)
        # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5, total_iters=1000)
        img_str = imgname.split('/')[-1].split('.')[0] + '_L2_high'
        os.makedirs(f'out/{img_str}', exist_ok=True)
        os.makedirs(f'out/{img_str}/recon_imgs', exist_ok=True)
        os.makedirs(f'out/{img_str}/latents', exist_ok=True)
        for step_idx in tqdm(range(total_step), total=total_step):
            if space == 'z':
                inv_image = self.G(code, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
                # print(self.G.synthesis.extra_repr())
            elif space == 'wp':
                # print(code.shape)
                inv_image = self.G.synthesis(code, noise_mode=noise_mode)
            
            # print(inv_image.shape, gt_image.shape)
            # print('\n\n\n\n\n')
            # losses
            mseloss = self.MSE_C(inv_image, gt_image)*mse_w
            lpipsloss = self.LPIPS_C(self.resize_image(inv_image), self.resize_image(gt_image))*lpip_w   # the input to LPIPS should be 256x256 pixels
            # ssimloss = self.SSIM(inv_image, gt_image)
            loss = mseloss * 10 + lpipsloss # + ssimloss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step_idx %100 == 0:
                loss_info = 'Step:{} \t MSE:{:0.3f} \t LPIPS:{:0.3f} \n'.format(step_idx, mseloss.item(), lpipsloss.item())
                self.logger.write(loss_info)
                if space == 'wp':
                    inv_image = self.G.synthesis(code, noise_mode=noise_mode)
                elif space == 'z':
                    inv_image = self.G(code, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
                inv_image = self.process_image(inv_image, img2tensor=False)
                inv_image.save(osp.join(f'out/{img_str}/recon_imgs', f"{step_idx}_{img_str}.jpg"))
                np.save(osp.join(f'out/{img_str}/latents',f'{step_idx}_{img_str}.npy'), code.detach().cpu().numpy())
                
        self.logger.close()
        if space == 'z':
            inv_image = self.G(code, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        elif space == 'wp':
            inv_image = self.G.synthesis(code, noise_mode=noise_mode)
        inv_image = self.process_image(inv_image, img2tensor=False)
        return inv_image

@click.command()
@click.option('--imgname', type=str, help='image name')
@click.option('--out', 'out_path', type=str, default='out', help='save path of inversed image')
@click.option('--stylegan3', 'stylegan3_path', type=str, default='stylegan3-r-afhqv2-512x512.pkl', help='path of stylegan3 pkl file')
@click.option('--steps', 'total_step', type=int, default=4000, help='total optimization step')
@click.option('--lr', type=float, default=3e-1, help='learning rate')
@click.option('--mse_w', type=float, default=1, help='weight of mse loss')
@click.option('--lpip_w', type=float, default=10, help='weight of lpips')
@click.option('--trunc', 'truncation_psi', type=float, default=0.7, help='Truncation psi')
@click.option('--noise_mode', type=click.Choice(['const', 'random', 'none']), default='random',help='noise mode')
@click.option('--space', type=click.Choice(['z', 'wp']), default='wp',help='noise mode') # w space is to be implemented.
def main(imgname, out_path, stylegan3_path, total_step, lr, mse_w, lpip_w, truncation_psi, noise_mode, space):
    logpath = osp.join(out_path, 'log.txt')
    inversor = Inversor(stylegan3_path, logpath)
    inv_image = inversor.inverse(imgname, total_step, lr,  mse_w, lpip_w, truncation_psi, noise_mode, space)
    # inv_image.save(osp.join(out_path, imgname.split('/')[-1]))


if __name__ == '__main__':
    main()
