"""
Visualiser les patches de spectrogramme générés, pour vérifier la qualité des données
"""

import numpy as np
import matplotlib.pyplot as plt
from data_generator import SpectrogramGenerator

def visualize_patches():
    """Visualiser quelques patches de spectrogramme"""
    
    # Créer le générateur
    generator = SpectrogramGenerator(
        batch_size=4,
        chunk_duration=5.0
    )
    
    # Obtenir un batch
    gen = generator.generate_batch()
    x_batch, y_batch = next(gen)
    
    # Créer la figure
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Spectrogram Patches (Mix vs Vocals)', fontsize=16)
    
    # Afficher 4 échantillons
    for i in range(4):
        # Spectrogramme Mix
        ax1 = axes[0, i]
        im1 = ax1.imshow(
            x_batch[i],
            aspect='auto',
            origin='lower',
            cmap='viridis',
            interpolation='nearest'
        )
        ax1.set_title(f'Mix - Échantillon {i+1}')
        ax1.set_xlabel('Trames Temporelles')
        ax1.set_ylabel('Bins de Fréquence')
        plt.colorbar(im1, ax=ax1)
        
        # Spectrogramme Vocals
        ax2 = axes[1, i]
        im2 = ax2.imshow(
            y_batch[i],
            aspect='auto',
            origin='lower',
            cmap='viridis',
            interpolation='nearest'
        )
        ax2.set_title(f'Vocals - Échantillon {i+1}')
        ax2.set_xlabel('Trames Temporelles')
        ax2.set_ylabel('Bins de Fréquence')
        plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig('spectrogram_patches.png', dpi=150, bbox_inches='tight')
    print("Image sauvegardée sous : spectrogram_patches.png")
    plt.show()
    
    # Afficher les statistiques
    print("\nStatistiques des données :")
    print(f"Shape du batch : {x_batch.shape}")
    print(f"\nMix (x_batch) :")
    print(f"  Min : {x_batch.min():.4f}")
    print(f"  Max : {x_batch.max():.4f}")
    print(f"  Moyenne : {x_batch.mean():.4f}")
    print(f"  Écart-type : {x_batch.std():.4f}")
    
    print(f"\nVocals (y_batch) :")
    print(f"  Min : {y_batch.min():.4f}")
    print(f"  Max : {y_batch.max():.4f}")
    print(f"  Moyenne : {y_batch.mean():.4f}")
    print(f"  Écart-type : {y_batch.std():.4f}")
    
    # Vérifier l'effet de recouvrement
    print("\nVérification du recouvrement :")
    print(f"  Longueur de patch : {generator.patch_frames} trames")
    print(f"  Hop de patch : {generator.patch_hop} trames")
    print(f"  Taux de recouvrement : {(generator.patch_frames - generator.patch_hop) / generator.patch_frames * 100:.1f}%")
    print(f"  Environ {int(5.0 * generator.sample_rate / generator.hop_length / generator.patch_hop)} patches générés par chunk de 5 secondes")


if __name__ == "__main__":
    visualize_patches()

