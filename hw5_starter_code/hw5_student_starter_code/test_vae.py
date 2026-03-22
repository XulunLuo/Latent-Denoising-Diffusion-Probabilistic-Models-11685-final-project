import sys
sys.path.insert(0, r'C:\Users\xulunl\Desktop\Intro to DeepLearning (11-785)\HW5\hw5_starter_code')

import torch
from models.vae import VAE

vae = VAE()
vae.init_from_ckpt(r'C:\Users\xulunl\Desktop\Intro to DeepLearning (11-785)\HW5\Model.ckpt')
vae.eval()
vae = vae.to('cuda')

# Test with 2 random 128*128 image shapes
x = torch.randn(2, 3, 128, 128).to('cuda')

latent = vae.encode(x) * 0.1845
print("latent shape:", latent.shape)

recon = vae.decode(latent / 0.1845)
print("recon shape:", recon.shape)

print("VAE test passed")