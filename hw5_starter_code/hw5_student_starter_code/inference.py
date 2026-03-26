import os 
import sys 
import argparse
import numpy as np
import ruamel.yaml as yaml
import torch
import wandb 
import logging 
from logging import getLogger as get_logger
from tqdm import tqdm 
from PIL import Image
import torch.nn.functional as F

from torchvision.utils  import make_grid

from models import UNet, VAE, ClassEmbedder
from schedulers import DDPMScheduler, DDIMScheduler
from pipelines import DDPMPipeline
from utils import seed_everything, load_checkpoint

from train import parse_args

logger = get_logger(__name__)


def main():
    # parse arguments
    args = parse_args()

    # seed everything
    seed_everything(args.seed)

    # setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # device (must be defined before generator)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    
    # setup model
    logger.info("Creating model")
    # unet
    unet = UNet(input_size=args.unet_in_size, input_ch=args.unet_in_ch, T=args.num_train_timesteps, ch=args.unet_ch, ch_mult=args.unet_ch_mult, attn=args.unet_attn, num_res_blocks=args.unet_num_res_blocks, dropout=args.unet_dropout, conditional=args.use_cfg, c_dim=args.unet_ch)
    # preint number of parameters
    num_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logger.info(f"Number of parameters: {num_params / 10 ** 6:.2f}M")
    
    # TODO: ddpm shceduler
    scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        variance_type=args.variance_type,
        prediction_type=args.prediction_type,
        clip_sample=args.clip_sample,
        clip_sample_range=args.clip_sample_range,
    )
    # vae
    vae = None
    if args.latent_ddpm:
        vae = VAE()
        vae.init_from_ckpt('pretrained/model.ckpt')
        vae.eval()
    # cfg
    class_embedder = None
    if args.use_cfg:
        # TODO: class embeder
        class_embedder = ClassEmbedder(embed_dim=args.unet_ch, n_classes=args.num_classes)
        
    # send to device
    unet = unet.to(device)
    scheduler = scheduler.to(device)
    if vae:
        vae = vae.to(device)
    if class_embedder:
        class_embedder = class_embedder.to(device)
        
    # scheduler — rebuild with proper args (DDIM or DDPM) for inference
    if args.use_ddim:
        shceduler_class = DDIMScheduler
    else:
        shceduler_class = DDPMScheduler
    # TOOD: scheduler
    scheduler = shceduler_class(
        num_train_timesteps=args.num_train_timesteps,
        num_inference_steps=args.num_inference_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        variance_type=args.variance_type,
        prediction_type=args.prediction_type,
        clip_sample=args.clip_sample,
        clip_sample_range=args.clip_sample_range,
    ).to(device)  # move to device after rebuild

    # load checkpoint
    load_checkpoint(unet, scheduler, vae=vae, class_embedder=class_embedder, checkpoint_path=args.ckpt)
    
    # TODO: pipeline
    pipeline = DDPMPipeline(unet, scheduler, vae=vae, class_embedder=class_embedder)

    
    logger.info("***** Running Infrence *****")
    
    # TODO: we run inference to generation 5000 images
    # TODO: with cfg, we generate 50 images per class
    all_images = []
    if args.use_cfg:
        # generate 50 images per class
        for i in tqdm(range(args.num_classes)):
            logger.info(f"Generating 50 images for class {i}")
            batch_size = 50
            classes = torch.full((batch_size,), i, dtype=torch.long, device=device)
            gen_images = pipeline(
                batch_size=batch_size,
                num_inference_steps=args.num_inference_steps,
                classes=classes,
                guidance_scale=args.cfg_guidance_scale,
                generator=generator,
                device=device,
            )
            all_images.extend(gen_images)
    else:
        # generate 5000 images
        batch_size = 50
        for _ in tqdm(range(0, 5000, batch_size)):
            gen_images = pipeline(
                batch_size=batch_size,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                device=device,
            )
            all_images.extend(gen_images)

    # TODO: load validation images as reference batch
    from torchvision import datasets, transforms
    val_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),   # [0, 1]
    ])
    val_dir = args.data_dir.replace('train', 'validation')
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=50, shuffle=False, num_workers=0)

    # TODO: using torchmetrics for evaluation, check the documents of torchmetrics
    import torchmetrics

    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore

    # TODO: compute FID and IS
    fid = FrechetInceptionDistance(feature=2048).to(device)
    inception_score = InceptionScore().to(device)

    # update FID with real validation images
    logger.info("Computing reference statistics from validation set...")
    for real_imgs, _ in tqdm(val_loader):
        real_imgs_uint8 = (real_imgs * 255).byte().to(device)
        fid.update(real_imgs_uint8, real=True)

    # convert generated PIL images to uint8 tensors and update metrics
    logger.info("Computing metrics on generated images...")
    to_tensor = transforms.ToTensor()
    gen_batch_size = 50
    for i in range(0, len(all_images), gen_batch_size):
        batch_pil = all_images[i:i + gen_batch_size]
        batch_tensor = torch.stack([to_tensor(img) for img in batch_pil])  # (N, C, H, W) float [0,1]
        batch_uint8 = (batch_tensor * 255).byte().to(device)
        fid.update(batch_uint8, real=False)
        inception_score.update(batch_uint8)

    fid_score = fid.compute().item()
    is_mean, is_std = inception_score.compute()
    logger.info(f"FID: {fid_score:.4f}")
    logger.info(f"IS: {is_mean.item():.4f} ± {is_std.item():.4f}")

    # Save results to a text file next to the checkpoint
    results_path = os.path.join(os.path.dirname(args.ckpt), 'eval_results.txt')
    with open(results_path, 'w') as f:
        f.write(f"Checkpoint: {args.ckpt}\n")
        f.write(f"DDIM: {args.use_ddim}, Steps: {args.num_inference_steps}\n")
        f.write(f"Latent DDPM: {args.latent_ddpm}, CFG: {args.use_cfg}\n")
        f.write(f"FID: {fid_score:.4f}\n")
        f.write(f"IS: {is_mean.item():.4f} +/- {is_std.item():.4f}\n")
    logger.info(f"Results saved to {results_path}")
    
        
if __name__ == '__main__':
    main()