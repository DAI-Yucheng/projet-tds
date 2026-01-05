"""
Entraînement du modèle U-Net pour la séparation de sources

Méthode simplifiée selon notebook :
- Cible d'entraînement : vocals magnitude (direct)
- Loss : MaskedL1Loss(vocals_pred, vocals_true), où vocals_pred = mask * mix
- Pas de log normalization, utilisation directe des magnitude spectrograms
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from data_generator import SpectrogramGenerator
from unet_model import UNet

class DataLoader:
    def __init__(self, generator, batches_per_epoch):
        self.generator = generator
        self.batches_per_epoch = batches_per_epoch
    
    def __len__(self):
        return self.batches_per_epoch
    
    def __iter__(self):
        gen = self.generator.generate_batch()
        for _ in range(self.batches_per_epoch):
            yield next(gen)

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss(reduction="mean")

    def forward(self, X, mask, Y):
        """
        Args:
            X (Tensor): spectrogramme du mix (batch, freq_bins, time_frames)
            mask (Tensor): masque prédit par le réseau (batch, freq_bins, time_frames)
            Y (Tensor): spectrogramme cible (batch, freq_bins, time_frames)

        Returns:
            loss (Tensor): valeur scalaire
        """
        Y_hat = mask * X      # ⊙ produit élément par élément
        loss = self.l1(Y_hat, Y)
        return loss


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, n_epochs):
    """Entraîner un epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    iterator = tqdm(dataloader, desc=f"Epoch {epoch}/{n_epochs}") if HAS_TQDM else dataloader
    
    for mix_spec, vocals_spec in iterator:
        mix_spec = torch.FloatTensor(mix_spec).to(device)
        vocals_spec = torch.FloatTensor(vocals_spec).to(device)
        
        optimizer.zero_grad()
        mask = model(mix_spec)
        loss = criterion(mask, mix_spec, vocals_spec)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if HAS_TQDM:
            iterator.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches # la moyenne de la L1 distance sur tous les batches 


def validate(model, dataloader, criterion, device):
    """Valider le modèle"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for mix_spec, vocals_spec in dataloader:
            mix_spec = torch.FloatTensor(mix_spec).to(device)
            vocals_spec = torch.FloatTensor(vocals_spec).to(device)
            
            mask = model(mix_spec)
            loss = criterion(mask, mix_spec, vocals_spec)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def load_checkpoint(checkpoint_path, model, optimizer, device):
    """Charger un checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_loss = checkpoint.get('loss', checkpoint.get('best_loss', float('inf')))
    return start_epoch, best_loss


def train(
    n_epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 5e-4,
    n_songs: int = 10,
    save_dir: str = "checkpoints",
    use_gpu: bool = True,
    resume_from: str = None,
    batches_per_epoch: int = 50
):
    """Entraîner le modèle U-Net"""
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Générateur de données
    musdb_path = "MUSDB18/musdb18"  # Chemin relatif au répertoire du projet
    generator = SpectrogramGenerator(
        batch_size=batch_size,
        chunk_duration=12.0,
        musdb_path=musdb_path
    )
    
    if n_songs < len(generator.mus.tracks):
        generator.mus.tracks = generator.mus.tracks[:n_songs]
    
    data_loader = DataLoader(generator, batches_per_epoch)
    
    # Ensemble de validation
    val_generator = SpectrogramGenerator(
        batch_size=batch_size,
        chunk_duration=12.0,
        musdb_path=musdb_path
    )
    val_generator.mus.tracks = generator.mus.tracks[:]
    validation_batches = val_generator.generate_fixed_validation_set(n_batches=15, seed=42)
    
    # Modèle - utiliser 512 fréquence bins 
    model = UNet(n_freq_bins=512, n_time_frames=128, n_channels=16, n_layers=6).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Modèle: {total_params:,} paramètres")
    
    criterion = MaskedL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    
    # Charger le checkpoint
    start_epoch = 1
    best_loss = float('inf')
    
    if resume_from and os.path.exists(resume_from):
        print(f"Chargement checkpoint: {resume_from}")
        start_epoch, best_loss = load_checkpoint(resume_from, model, optimizer, device)
    elif os.path.exists(os.path.join(save_dir, 'best_model.pth')):
        checkpoint_path = os.path.join(save_dir, 'best_model.pth')
        print(f"Chargement checkpoint: {checkpoint_path}")
        start_epoch, best_loss = load_checkpoint(checkpoint_path, model, optimizer, device)
    
    print(f"Entraînement: Epoch {start_epoch}-{n_epochs}")
    
    train_losses = []
    val_losses = []
    
    for epoch in range(start_epoch, n_epochs + 1):
        # Entraînement
        train_loss = train_epoch(model, data_loader, criterion, optimizer, device, epoch, n_epochs)
        train_losses.append(train_loss)
        
        # Validation
        if len(validation_batches) > 0:
            val_loss = validate(model, validation_batches, criterion, device)
        else:
            val_loss = train_loss
        val_losses.append(val_loss)
        
        # Planification du taux d'apprentissage
        if len(val_losses) >= 3:
            smoothed_val_loss = sum(val_losses[-3:]) / 3
        else:
            smoothed_val_loss = val_loss
        scheduler.step(smoothed_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Afficher les informations
        print(f"Epoch {epoch}/{n_epochs}: train={train_loss:.6f}, val={val_loss:.6f}, lr={current_lr:.2e}", end="")
        
        # Sauvegarder le meilleur modèle
        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': val_loss,
                'best_loss': best_loss
            }
            torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
            print(f"  ✓ Meilleur modèle sauvegardé (loss : {val_loss:.6f})")
        
        print("-" * 70)
    
    # Sauvegarder le modèle final
    final_checkpoint = {
        'epoch': n_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': val_losses[-1],
        'best_loss': best_loss
    }
    torch.save(final_checkpoint, os.path.join(save_dir, 'final_model.pth'))
    
    print(f"\nTerminé. Meilleur loss: {best_loss:.6f} | Sauvegardé dans: {save_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Entraîner le modèle U-Net pour la séparation de sources')
    parser.add_argument('--epochs', type=int, default=20, help='Nombre d\'epochs d\'entraînement')
    parser.add_argument('--batch-size', type=int, default=16, help='Taille du batch')
    parser.add_argument('--lr', type=float, default=5e-4, help='Taux d\'apprentissage')
    parser.add_argument('--n-songs', type=int, default=10, help='Nombre de chansons utilisées')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='Répertoire de sauvegarde du modèle')
    parser.add_argument('--cpu', action='store_true', help='Forcer l\'utilisation du CPU')
    parser.add_argument('--resume', type=str, default=None, help='Reprendre l\'entraînement depuis un checkpoint')
    parser.add_argument('--batches-per-epoch', type=int, default=50, help='Nombre de batches par epoch')
    
    args = parser.parse_args()
    
    train(
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        n_songs=args.n_songs,
        save_dir=args.save_dir,
        use_gpu=not args.cpu,
        resume_from=args.resume,
        batches_per_epoch=args.batches_per_epoch
    )
