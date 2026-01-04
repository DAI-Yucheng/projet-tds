"""
Utilisation du modèle U-Net entraîné pour l'inférence, génération d'audio vocal

Processus complet :
1. Charger le fichier audio (mix)
2. Convertir en spectrogramme (utiliser les mêmes paramètres que l'entraînement)
3. Normaliser (identique à l'entraînement)
4. Prédire le mask avec le modèle
5. Dénormaliser
6. Reconstruire l'audio (ISTFT)
7. Sauvegarder le fichier audio vocal
"""

import torch
import numpy as np
import librosa
import soundfile as sf
from unet_model import UNet
import matplotlib.pyplot as plt
import os


def load_model(checkpoint_path, device='cpu', n_channels=64):
    """
    Charger le modèle entraîné
    
    Args:
        checkpoint_path: Chemin du checkpoint du modèle
        device: Appareil ('cpu' ou 'cuda')
        n_channels: Nombre de canaux du modèle (doit correspondre à l'entraînement)
        
    Returns:
        model: Modèle chargé
    """
    # Créer le modèle (doit correspondre à la configuration d'entraînement)
    # Attention : maintenant on utilise 512 bins de fréquence (pas 513), cohérent avec la nouvelle méthode d'entraînement
    model = UNet(
        n_freq_bins=512,  # Modification : 512 au lieu de 513
        n_time_frames=128,
        n_channels=n_channels,  # Utiliser le nombre de canaux de l'entraînement
        n_layers=4
    )
    
    # Charger le checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Modèle chargé avec succès !")
    print(f"  Epoch : {checkpoint.get('epoch', 'N/A')}")
    print(f"  Loss : {checkpoint.get('loss', 'N/A'):.6f}")
    
    return model


def audio_to_spectrogram(audio, original_sr=None, sample_rate=8192, n_fft=1024, hop_length=768):
    """
    Convertir l'audio en spectrogramme de magnitude (identique à l'entraînement)
    
    Args:
        audio: Tableau audio
        original_sr: Taux d'échantillonnage original de l'audio (si None, assume déjà à sample_rate)
        sample_rate: Taux d'échantillonnage cible (utilisé 8192 pendant l'entraînement)
        n_fft: Taille de la fenêtre STFT (utilisé 1024 pendant l'entraînement)
        hop_length: Longueur du hop STFT (utilisé 768 pendant l'entraînement)
        
    Returns:
        magnitude: Spectrogramme de magnitude, shape (freq_bins, time_frames)
        phase: Spectrogramme de phase (utilisé pour reconstruire l'audio)
    """
    # Si stéréo, convertir en mono
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=0)
    
    # Rééchantillonnage au taux d'échantillonnage cible (si nécessaire)
    if len(audio) == 0:
        return np.zeros((512, 1)), np.zeros((512, 1))
    
    # Important : d'abord rééchantillonner à 8192Hz (si le taux d'échantillonnage original est différent)
    if original_sr is not None and original_sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=original_sr, target_sr=sample_rate)
    
    # Exécuter STFT
    stft = librosa.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        window='hann',
        center=True
    )
    
    # Séparer magnitude et phase
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    # Utiliser seulement les 512 premiers bins de fréquence (au lieu de 513)
    # Cohérent avec le data_generator utilisé lors de l'entraînement
    magnitude = magnitude[:512, :]
    phase = phase[:512, :]
    
    return magnitude, phase


def normalize_spectrogram(spec):
    """
    Normaliser le spectrogramme (identique à l'entraînement)
    
    Attention : selon la nouvelle méthode d'entraînement, on n'utilise plus la log normalization
    Retourne directement la magnitude originale (cohérent avec data_generator)
    
    Args:
        spec: Spectrogramme de magnitude
        
    Returns:
        spec: Spectrogramme de magnitude (pas de normalisation, utilisation directe)
    """
    # Nouvelle méthode : pas de log normalization, utilisation directe de la magnitude originale
    return spec


def denormalize_spectrogram(spec):
    """
    Dénormaliser le spectrogramme
    
    Attention : comme on n'utilise plus la normalisation, cette fonction retourne simplement la valeur originale
    
    Args:
        spec: Spectrogramme de magnitude
        
    Returns:
        spec: Spectrogramme de magnitude (inchangé)
    """
    # Nouvelle méthode : pas besoin de dénormalisation, retour direct
    return spec


def spectrogram_to_audio(magnitude, phase, hop_length=768, n_fft=1024):
    """
    Reconstruire l'audio à partir de magnitude et phase (ISTFT)
    
    Args:
        magnitude: Spectrogramme de magnitude (512 bins de fréquence)
        phase: Spectrogramme de phase (512 bins de fréquence)
        hop_length: Longueur du hop STFT
        n_fft: Taille de la fenêtre STFT (utilisé pour reconstruire le spectre complet)
        
    Returns:
        audio: Tableau audio reconstruit
    """
    # Important : nous devons reconstruire le STFT complet (513 bins de fréquence)
    # Car nous n'avons pris que les 512 premiers bins, il faut compléter à 513 (n_fft//2+1)
    full_freq_bins = n_fft // 2 + 1  # 513
    
    # Si magnitude n'a que 512 bins, il faut compléter
    if magnitude.shape[0] == 512:
        # Créer le STFT complet, les 512 premiers bins utilisent nos données, le dernier bin (Nyquist) est mis à 0
        full_magnitude = np.zeros((full_freq_bins, magnitude.shape[1]), dtype=magnitude.dtype)
        full_phase = np.zeros((full_freq_bins, phase.shape[1]), dtype=phase.dtype)
        
        full_magnitude[:512, :] = magnitude
        full_phase[:512, :] = phase
        # Phase de la fréquence Nyquist (dernier bin) mise à 0
        full_phase[-1, :] = 0
    else:
        full_magnitude = magnitude
        full_phase = phase
    
    # Reconstruire le spectrogramme complexe
    stft = full_magnitude * np.exp(1j * full_phase)
    
    # ISTFT pour reconstruire l'audio
    audio = librosa.istft(
        stft,
        hop_length=hop_length,
        window='hann',
        center=True,
        length=None  # Laisser librosa calculer automatiquement la longueur
    )
    
    return audio


def predict_mask(model, mix_spec_norm, device='cpu'):
    """
    Prédire le mask (l'entrée doit être un spectrogramme normalisé)
    
    Args:
        model: Modèle entraîné
        mix_spec_norm: Spectrogramme du mix normalisé, shape (freq_bins, time_frames)
        device: Appareil
        
    Returns:
        mask: Mask prédit, shape (freq_bins, time_frames)
    """
    # Convertir en tensor et ajouter la dimension batch
    if isinstance(mix_spec_norm, np.ndarray):
        mix_spec_norm = torch.FloatTensor(mix_spec_norm)
    
    if len(mix_spec_norm.shape) == 2:
        mix_spec_norm = mix_spec_norm.unsqueeze(0)
    
    mix_spec_norm = mix_spec_norm.to(device)
    
    # Inférence
    with torch.no_grad():
        mask = model(mix_spec_norm)
    
    # Reconvertir en numpy
    if mask.is_cuda:
        mask = mask.cpu()
    
    mask = mask.numpy()
    
    # Supprimer la dimension batch
    if len(mask.shape) == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    
    return mask


def separate_vocals(audio_path, model, device='cpu', output_path=None):
    """
    Séparer la voix d'un fichier audio
    
    Args:
        audio_path: Chemin du fichier audio d'entrée
        model: Modèle entraîné
        device: Appareil
        output_path: Chemin du fichier audio vocal de sortie (optionnel)
        
    Returns:
        vocals_audio: Tableau audio vocal séparé
        sample_rate: Taux d'échantillonnage
    """
    print(f"\nTraitement du fichier audio : {audio_path}")
    
    # 1. Charger l'audio (conserver le taux d'échantillonnage original)
    audio, original_sr = librosa.load(audio_path, sr=None, mono=False)
    print(f"  Taux d'échantillonnage original : {original_sr} Hz")
    original_length = len(audio[0] if len(audio.shape) > 1 else audio) / original_sr
    print(f"  Longueur de l'audio : {original_length:.2f} secondes")
    
    # 2. Convertir en spectrogramme (rééchantillonner à 8192Hz)
    print("  Conversion en spectrogramme...")
    magnitude, phase = audio_to_spectrogram(
        audio, 
        original_sr=original_sr,  # Passer le taux d'échantillonnage original
        sample_rate=8192, 
        n_fft=1024, 
        hop_length=768
    )
    print(f"  Shape du spectrogramme : {magnitude.shape}")
    
    # 3. Attention : la nouvelle méthode n'utilise pas de normalisation, utilise directement la magnitude originale
    # Cohérent avec le data_generator utilisé lors de l'entraînement
    mix_spec = magnitude  # Utilisation directe, pas de normalisation
    
    # 4. Traiter les longs audios : traitement par blocs (si plus de 128 frames)
    patch_frames = 128
    hop_frames = patch_frames // 2  # 50% de chevauchement (cohérent avec l'entraînement)
    
    if magnitude.shape[1] <= patch_frames:
        # Audio court, traitement direct
        mask = predict_mask(model, mix_spec, device)
        estimated_vocals_magnitude = mask * mix_spec  # mask * mix = vocals
    else:
        # Audio long, traitement par blocs
        print(f"  Audio long, traitement par blocs (patch_frames={patch_frames}, chevauchement=50%)...")
        estimated_vocals_magnitude = np.zeros_like(mix_spec)
        weight = np.zeros_like(mix_spec)
        
        for start in range(0, magnitude.shape[1] - patch_frames + 1, hop_frames):
            end = start + patch_frames
            patch = mix_spec[:, start:end]
            
            mask_patch = predict_mask(model, patch, device)
            estimated_patch = mask_patch * patch  # mask * mix = vocals
            
            estimated_vocals_magnitude[:, start:end] += estimated_patch
            weight[:, start:end] += 1.0
        
        # Traiter la partie restante
        if start + patch_frames < magnitude.shape[1]:
            patch = mix_spec[:, -patch_frames:]
            mask_patch = predict_mask(model, patch, device)
            estimated_patch = mask_patch * patch
            estimated_vocals_magnitude[:, -patch_frames:] += estimated_patch
            weight[:, -patch_frames:] += 1.0
        
        # Normaliser (gérer le chevauchement)
        estimated_vocals_magnitude /= (weight + 1e-8)
    
    # 5. Reconstruire l'audio (utiliser la phase du mix original)
    print("  Reconstruction de l'audio...")
    # L'audio reconstruit est à un taux d'échantillonnage de 8192Hz
    vocals_audio_8192 = spectrogram_to_audio(
        estimated_vocals_magnitude, 
        phase, 
        hop_length=768,
        n_fft=1024
    )
    
    # 6. Rééchantillonnage au taux d'échantillonnage original
    print(f"  Rééchantillonnage de 8192 Hz vers {original_sr} Hz...")
    if original_sr != 8192:
        vocals_audio = librosa.resample(vocals_audio_8192, orig_sr=8192, target_sr=original_sr)
    else:
        vocals_audio = vocals_audio_8192
    
    # Vérifier la longueur
    reconstructed_length = len(vocals_audio) / original_sr
    print(f"  Longueur de l'audio reconstruit : {reconstructed_length:.2f} secondes")
    if abs(reconstructed_length - original_length) > 1.0:
        print(f"  ⚠️  ATTENTION : Longueur différente de l'original ({original_length:.2f}s vs {reconstructed_length:.2f}s)")
    
    # 7. Sauvegarder le fichier audio
    if output_path:
        sf.write(output_path, vocals_audio, original_sr)
        print(f"  ✓ Audio vocal sauvegardé : {output_path}")
    
    return vocals_audio, original_sr


def visualize_prediction(mix_spec, mix_spec_input, mask, estimated_vocals, save_path=None):
    """
    Visualiser les résultats de prédiction
    
    Args:
        mix_spec: Spectrogramme du mix original
        mix_spec_input: Spectrogramme du mix utilisé comme entrée (maintenant identique à mix_spec)
        mask: Mask prédit
        estimated_vocals: Vocals estimés (mask * mix)
        save_path: Chemin de sauvegarde (optionnel)
    """
    # Nouvelle méthode : pas besoin de dénormalisation, utilisation directe
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Mix (original)
    im1 = axes[0, 0].imshow(
        np.log(mix_spec + 1e-8),  # Échelle log pour une meilleure visualisation
        aspect='auto',
        origin='lower',
        cmap='viridis'
    )
    axes[0, 0].set_title('Spectrogramme du Mix (Original)')
    axes[0, 0].set_xlabel('Frames Temporelles')
    axes[0, 0].set_ylabel('Bins de Fréquence')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Mask
    im2 = axes[0, 1].imshow(
        mask,
        aspect='auto',
        origin='lower',
        cmap='hot',
        vmin=0,
        vmax=1
    )
    axes[0, 1].set_title(f'Mask Prédit (mean={mask.mean():.3f}, std={mask.std():.3f})')
    axes[0, 1].set_xlabel('Frames Temporelles')
    axes[0, 1].set_ylabel('Bins de Fréquence')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Vocals Estimés (après dénormalisation)
    im3 = axes[1, 0].imshow(
        np.log(estimated_vocals + 1e-8),  # Échelle log pour une meilleure visualisation
        aspect='auto',
        origin='lower',
        cmap='viridis'
    )
    axes[1, 0].set_title('Spectrogramme des Vocals Estimés')
    axes[1, 0].set_xlabel('Frames Temporelles')
    axes[1, 0].set_ylabel('Bins de Fréquence')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Comparaison : Mix vs Vocals Estimés
    diff = np.log(mix_spec + 1e-8) - np.log(estimated_vocals + 1e-8)
    im4 = axes[1, 1].imshow(
        diff,
        aspect='auto',
        origin='lower',
        cmap='RdBu_r',
        vmin=-3,
        vmax=3
    )
    axes[1, 1].set_title('Différence (Mix - Vocals)')
    axes[1, 1].set_xlabel('Frames Temporelles')
    axes[1, 1].set_ylabel('Bins de Fréquence')
    plt.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Image sauvegardée : {save_path}")
    else:
        plt.show()


def find_latest_checkpoint(checkpoint_dir="checkpoints"):
    """
    Trouver automatiquement le fichier checkpoint le plus récent
    
    Args:
        checkpoint_dir: Répertoire contenant les checkpoints
        
    Returns:
        checkpoint_path: Chemin du checkpoint le plus récent, None s'il n'existe pas
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Ordre de priorité : best_model.pth > final_model.pth > autres par date de modification
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
    final_model_path = os.path.join(checkpoint_dir, "final_model.pth")
    
    if os.path.exists(best_model_path):
        return best_model_path
    elif os.path.exists(final_model_path):
        return final_model_path
    else:
        # Chercher tous les fichiers .pth, trier par date de modification
        checkpoint_files = []
        for file in os.listdir(checkpoint_dir):
            if file.endswith('.pth'):
                file_path = os.path.join(checkpoint_dir, file)
                mtime = os.path.getmtime(file_path)
                checkpoint_files.append((mtime, file_path))
        
        if checkpoint_files:
            # Trier par date de modification décroissante, retourner le plus récent
            checkpoint_files.sort(reverse=True)
            return checkpoint_files[0][1]
    
    return None


def test_inference(audio_path=None, checkpoint_path=None, n_channels=64):
    """
    Tester la fonction d'inférence : séparer la voix d'un fichier audio
    
    Args:
        audio_path: Chemin du fichier audio d'entrée (si None, essaiera de charger depuis MUSDB)
        checkpoint_path: Chemin du checkpoint du modèle (si None, recherche automatique du plus récent)
        n_channels: Nombre de canaux du modèle (doit correspondre à l'entraînement)
    """
    print("=" * 70)
    print("Inférence de Séparation Vocale U-Net")
    print("=" * 70)
    
    # Si aucun checkpoint n'est spécifié, rechercher automatiquement le plus récent
    if checkpoint_path is None:
        print("\nRecherche du checkpoint le plus récent...")
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path:
            print(f"  ✓ Checkpoint trouvé : {checkpoint_path}")
        else:
            print("  ✗ Aucun checkpoint trouvé dans le dossier 'checkpoints'")
            print("  Veuillez d'abord entraîner le modèle : python train.py")
            return
    else:
        # Si un checkpoint est spécifié, vérifier s'il existe
        if not os.path.exists(checkpoint_path):
            print(f"\n⚠️  Checkpoint spécifié introuvable : {checkpoint_path}")
            print("Tentative de recherche automatique...")
            checkpoint_path = find_latest_checkpoint()
            if checkpoint_path:
                print(f"  ✓ Checkpoint trouvé : {checkpoint_path}")
            else:
                print("  ✗ Aucun checkpoint trouvé")
                print("  Veuillez d'abord entraîner le modèle : python train.py")
                return
    
    # Vérifier le fichier du modèle
    if not os.path.exists(checkpoint_path):
        print(f"\nErreur : fichier modèle introuvable {checkpoint_path}")
        print("Veuillez d'abord entraîner le modèle : python train.py")
        return
    
    # Charger le modèle
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nAppareil utilisé : {device}")
    model = load_model(checkpoint_path, device, n_channels=n_channels)
    
    # Traiter l'audio
    if audio_path and os.path.exists(audio_path):
        # Utiliser le fichier audio spécifié
        output_path = audio_path.replace('.wav', '_vocals.wav').replace('.mp3', '_vocals.wav')
        if output_path == audio_path:
            output_path = audio_path + '_vocals.wav'
        
        vocals_audio, sr = separate_vocals(audio_path, model, device, output_path)
        
        # Visualisation (traiter un petit segment pour la visualisation)
        print("\nGénération de la visualisation...")
        magnitude, phase = audio_to_spectrogram(vocals_audio[:sr*5], sample_rate=8192)  # 5 premières secondes
        mix_magnitude, _ = audio_to_spectrogram(
            librosa.load(audio_path, sr=8192, duration=5)[0],
            sample_rate=8192
        )
        mix_spec = mix_magnitude  # Nouvelle méthode : pas de normalisation
        mask = predict_mask(model, mix_spec, device)
        estimated_vocals = mask * mix_spec  # mask * mix = vocals
        
        visualize_prediction(
            mix_magnitude, mix_spec, mask, estimated_vocals,
            "inference_result.png"
        )
        
    else:
        # Essayer de charger l'audio de test depuis MUSDB
        try:
            import musdb
            musdb_path = "MUSDB18/musdb18"
            if os.path.exists(musdb_path):
                print(f"\nChargement de l'audio de test depuis MUSDB...")
                mus = musdb.DB(root=musdb_path, download=False)
                if len(mus.tracks) > 0:
                    track = mus.tracks[0]
                    print(f"  Utilisation du track : {track.name}")
                    
                    # Charger l'audio du mix
                    mix_audio = track.audio.T
                    if len(mix_audio.shape) > 1:
                        mix_audio = np.mean(mix_audio, axis=0)
                    
                    # Sauvegarder le fichier temporaire
                    temp_path = "temp_mix.wav"
                    sf.write(temp_path, mix_audio, 44100)
                    
                    # Séparer la voix
                    output_path = "output_vocals.wav"
                    vocals_audio, sr = separate_vocals(temp_path, model, device, output_path)
                    
                    # Nettoyer le fichier temporaire
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    
                    print(f"\n✓ Inférence terminée !")
                    print(f"  Entrée : {track.name}")
                    print(f"  Sortie : {output_path}")
                else:
                    print("  Aucun track disponible dans MUSDB")
            else:
                print(f"\nErreur : dataset MUSDB introuvable : {musdb_path}")
                print("Veuillez spécifier le chemin du fichier audio : python inference.py --audio <path>")
        except Exception as e:
            print(f"\nErreur : {e}")
            print("Veuillez spécifier le chemin du fichier audio : python inference.py --audio <path>")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Inférence de séparation vocale U-Net')
    parser.add_argument('--audio', type=str, default=None, help='Chemin du fichier audio d\'entrée')
    parser.add_argument('--checkpoint', type=str, default=None, help='Chemin du checkpoint du modèle (si None, recherche automatique du plus récent)')
    parser.add_argument('--n-channels', type=int, default=64, help='Nombre de canaux du modèle (doit correspondre à l\'entraînement)')
    
    args = parser.parse_args()
    
    test_inference(
        audio_path=args.audio,
        checkpoint_path=args.checkpoint,  # None = recherche automatique
        n_channels=args.n_channels
    )
