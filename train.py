import os
import random
import logging
import argparse
import numpy as np
import torch
from torch import amp
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets import LoDoPaBSinogramDataset
import models
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import peak_signal_noise_ratio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchvision.models import vgg16, VGG16_Weights


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-dir', type=str, default='./checkpoints')
    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    # logging
    logging.basicConfig(level=logging.INFO)
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'logs'))

    # datasets & loaders
    train_ds = LoDoPaBSinogramDataset(
        os.path.join(args.data_root, 'observation_train'),
        os.path.join(args.data_root, 'ground_truth_train')
    )
    val_ds = LoDoPaBSinogramDataset(
        os.path.join(args.data_root, 'observation_validation'),
        os.path.join(args.data_root, 'ground_truth_validation')
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    # models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = models.SinogramEncoder().to(device)
    generator = models.Generator().to(device)
    discriminator = models.Discriminator().to(device)

    # optimizers & scheduler
    optG = Adam(list(encoder.parameters()) + list(generator.parameters()),
                lr=args.lr, weight_decay=args.weight_decay)
    optD = Adam(discriminator.parameters(),
                lr=args.lr, weight_decay=args.weight_decay)
    scheduler = OneCycleLR(optG, max_lr=args.lr,
                           total_steps=args.epochs * len(train_loader),
                           pct_start=0.1)

    # AMP & metrics
    scaler = amp.GradScaler(enabled=args.use_amp)
    ssim_fn  = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr_fn  = peak_signal_noise_ratio
    lpips_fn = LPIPS().to(device)
    vgg_feats= vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    vgg_feats = vgg_feats.features[:16].to(device).eval()
    for p in vgg_feats.parameters(): p.requires_grad = False

    best_ssim, patience = 0.0, 0
    for epoch in range(1, args.epochs + 1):
        encoder.train(); generator.train(); discriminator.train()
        sumD = sumG = 0.0
        for i, (sino, gt) in enumerate(train_loader, 1):
            sino, gt = sino.to(device), gt.to(device)

            # Discriminator step
            with amp.autocast(enabled=args.use_amp):
                fake = generator(encoder(sino)).detach()
                rD, fD = discriminator(gt), discriminator(fake)
                lossD = 0.5 * (
                    F.binary_cross_entropy_with_logits(rD, torch.ones_like(rD)) +
                    F.binary_cross_entropy_with_logits(fD, torch.zeros_like(fD))
                )
            optD.zero_grad()
            scaler.scale(lossD).backward()
            scaler.step(optD)

            # Generator step
            with amp.autocast(enabled=args.use_amp):
                out = generator(encoder(sino))
                adv    = F.binary_cross_entropy_with_logits(discriminator(out),
                                                           torch.ones_like(rD))
                perp   = F.l1_loss(
                    vgg_feats(out.repeat(1,3,1,1)),
                    vgg_feats(gt.repeat(1,3,1,1))
                )
                ssim_l = (1 - ssim_fn(out, gt))
                cycle  = F.l1_loss(generator(encoder(out)), gt)
                edge_l = models.edge_loss(out, gt)
                ffl    = models.focal_frequency_loss(out, gt)
                lossG  = adv + perp + ssim_l + cycle + edge_l + ffl
            optG.zero_grad()
            scaler.scale(lossG).backward()
            scaler.step(optG)
            scaler.update()
            scheduler.step()

            sumD += lossD.item()
            sumG += lossG.item()

            if i % 100 == 0:
                logging.info(f"Epoch {epoch} Batch {i}/{len(train_loader)} "
                             f"D: {sumD/i:.4f}  G: {sumG/i:.4f}")

        # validation & logging
        encoder.eval(); generator.eval()
        val_ssim = 0.0
        for sino, gt in val_loader:
            sino, gt = sino.to(device), gt.to(device)
            with torch.no_grad():
                out = generator(encoder(sino))
            val_ssim += ssim_fn(out, gt).item()
        val_ssim /= len(val_loader)

        writer.add_scalar('Loss/Discriminator', sumD/len(train_loader), epoch)
        writer.add_scalar('Loss/Generator',    sumG/len(train_loader), epoch)
        writer.add_scalar('Metric/SSIM',        val_ssim,               epoch)

        logging.info(f"Epoch {epoch} Complete — Val SSIM: {val_ssim:.4f}")
        if val_ssim > best_ssim:
            best_ssim = val_ssim
            torch.save({
                'encoder': encoder.state_dict(),
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
            }, os.path.join(args.save_dir, f"best_ssim_{best_ssim:.4f}.pth"))
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                logging.info("Early stopping")
                break

    writer.close()

if __name__ == '__main__':
    main()