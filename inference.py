"""
Utilisation du modèle U-Net entraîné pour l'inférence, génération d'audio vocal

Processus complet :
1. Charger le fichier audio (mix)
2. Convertir en spectrogramme (utiliser les mêmes paramètres que l'entraînement)
3. Prédire le mask avec le modèle
4. Reconstruire l'audio (ISTFT)
5. Sauvegarder le fichier audio vocal
"""

import torch
import numpy as np
import librosa
import soundfile as sf
from unet_model import UNet
import matplotlib.pyplot as plt
import os


def load_model(checkpoint_path, device='cpu', n_channels=16):
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
        n_freq_bins=512,  
        n_time_frames=128,
        n_channels=n_channels,  # Utiliser le nombre de canaux de l'entraînement
        n_layers=6
    )
    
    # Charger le checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    epoch = checkpoint.get('epoch', 'N/A')
    loss = checkpoint.get('loss', 'N/A')
    print(f"Modèle chargé: epoch={epoch}, loss={loss:.6f}" if isinstance(loss, (int, float)) else f"Modèle chargé: epoch={epoch}")
    
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
    # Important : nous devons reconstruire le STFT complet (librosa exige 513 bins de fréquence)
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
    print(f"Traitement: {audio_path}")
    
    
    audio, original_sr = librosa.load(audio_path, sr=None, mono=False)
    original_length = len(audio[0] if len(audio.shape) > 1 else audio) / original_sr
    
    magnitude, phase = audio_to_spectrogram(
        audio, 
        original_sr=original_sr,
        sample_rate=8192, 
        n_fft=1024, 
        hop_length=768
    )
    
    # Cohérent avec le data_generator utilisé lors de l'entraînement
    mix_spec = magnitude  
    
    patch_frames = 128
    hop_frames = patch_frames // 2  # 50% de chevauchement (cohérent avec l'entraînement)
    
    if magnitude.shape[1] <= patch_frames:
        # Audio court, traitement direct
        mask = predict_mask(model, mix_spec, device)
        estimated_vocals_magnitude = mask * mix_spec  # mask * mix = vocals
    else:
        # Audio long, traitement par blocs
        estimated_vocals_magnitude = np.zeros_like(mix_spec)
        weight = np.zeros_like(mix_spec)
        
        for start in range(0, magnitude.shape[1] - patch_frames + 1, hop_frames):
            end = start + patch_frames
            patch = mix_spec[:, start:end]
            
            mask_patch = predict_mask(model, patch, device)
            estimated_patch = mask_patch * patch  
            
            estimated_vocals_magnitude[:, start:end] += estimated_patch
            weight[:, start:end] += 1.0
        
        if start + patch_frames < magnitude.shape[1]:
            patch = mix_spec[:, -patch_frames:]
            mask_patch = predict_mask(model, patch, device)
            estimated_patch = mask_patch * patch
            estimated_vocals_magnitude[:, -patch_frames:] += estimated_patch
            weight[:, -patch_frames:] += 1.0
        
        # Normaliser (gérer le chevauchement)
        estimated_vocals_magnitude /= (weight + 1e-8)
    
    # Reconstruire l'audio
    vocals_audio_8192 = spectrogram_to_audio(
        estimated_vocals_magnitude, 
        phase, 
        hop_length=768,
        n_fft=1024
    )
    
    if original_sr != 8192:
        vocals_audio = librosa.resample(vocals_audio_8192, orig_sr=8192, target_sr=original_sr)
    else:
        vocals_audio = vocals_audio_8192
    
    if output_path:
        sf.write(output_path, vocals_audio, original_sr)
        print(f"✓ Sauvegardé: {output_path}")
    
    return vocals_audio, original_sr


def visualize_prediction(mix_spec, mask, estimated_vocals, save_path=None):
    """
    Visualiser les résultats de prédiction
    
    Args:
        mix_spec: Spectrogramme du mix original
        mask: Mask prédit
        estimated_vocals: Vocals estimés (mask * mix)
        save_path: Chemin de sauvegarde (optionnel)
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
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
    
    im3 = axes[1, 0].imshow(
        np.log(estimated_vocals + 1e-8),  
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
        print(f"Image sauvegardée: {save_path}")
    else:
        plt.show()


def find_latest_checkpoint(checkpoint_dir="vocal_checkpoints"):
    """
    Trouver automatiquement le fichier checkpoint le plus récent
    
    Args:
        checkpoint_dir: Répertoire contenant les checkpoints
        
    Returns:
        checkpoint_path: Chemin du checkpoint le plus récent, None s'il n'existe pas
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
    final_model_path = os.path.join(checkpoint_dir, "final_model.pth")
    
    if os.path.exists(best_model_path):
        return best_model_path
    elif os.path.exists(final_model_path):
        return final_model_path
    else:
        checkpoint_files = []
        for file in os.listdir(checkpoint_dir):
            if file.endswith('.pth'):
                file_path = os.path.join(checkpoint_dir, file)
                mtime = os.path.getmtime(file_path)
                checkpoint_files.append((mtime, file_path))
        
        if checkpoint_files:
            checkpoint_files.sort(reverse=True)
            return checkpoint_files[0][1]
    
    return None


def separate_musdb_tracks(checkpoint_path=None, n_channels=16, n_songs=None, musdb_path="MUSDB18/musdb18/test", output_dir="vocal_separation"):
    """
    Séparer les vocals de chansons du test set MUSDB
    
    Args:
        checkpoint_path: Chemin du checkpoint du modèle (si None, recherche automatique)
        n_channels: Nombre de canaux du modèle
        n_songs: Nombre de chansons à séparer (None = première chanson uniquement)
        musdb_path: Chemin du dataset MUSDB
        output_dir: Répertoire pour sauvegarder les vocals séparés
        
    Returns:
        Liste des chemins des fichiers créés
    """
    print("=" * 70)
    print("SÉPARATION VOCALE - TEST SET MUSDB")
    print("=" * 70)
    
    # Rechercher le checkpoint si non spécifié
    if checkpoint_path is None:
        print("\nRecherche du checkpoint...")
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path:
            print(f"  ✓ Checkpoint trouvé : {checkpoint_path}")
        else:
            print("  ✗ Aucun checkpoint trouvé")
            print("  Veuillez d'abord entraîner le modèle : python train.py")
            return []
    
    if not os.path.exists(checkpoint_path):
        print(f"\n✗ Checkpoint introuvable : {checkpoint_path}")
        return []
    
    # Charger le modèle
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nAppareil utilisé : {device}")
    model = load_model(checkpoint_path, device, n_channels=n_channels)
    
    # Charger le dataset MUSDB
    print(f"\nChargement du dataset MUSDB depuis : {musdb_path}")
    try:
        import musdb
        mus = musdb.DB(root=musdb_path, download=False)
    except Exception as e:
        print(f"✗ Erreur lors du chargement : {e}")
        return []
    
    # Obtenir les tracks de test
    test_tracks = [t for t in mus.tracks if t.subset == 'test']
    
    # Déterminer le nombre de chansons à traiter
    if n_songs is None:
        # Par défaut : première chanson uniquement
        test_tracks = test_tracks[:1] if len(test_tracks) > 0 else []
        print(f"\nTraitement de la première chanson du test set")
    elif n_songs == 9999:
        # 9999 = tous les fichiers
        print(f"\nTraitement de TOUS les fichiers du test set ({len(test_tracks)} tracks)")
    else:
        test_tracks = test_tracks[:n_songs]
        print(f"\nTraitement de {len(test_tracks)} chanson(s) du test set")
    
    if len(test_tracks) == 0:
        print("✗ Aucun track disponible")
        return []
    
    # Créer le répertoire de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # Séparer chaque track
    output_files = []
    print("\n" + "=" * 70)
    print("SÉPARATION EN COURS")
    print("=" * 70)
    
    for i, track in enumerate(test_tracks):
        try:
            print(f"\n[{i+1}/{len(test_tracks)}] Traitement : {track.name}")
            
            # Obtenir l'audio du mix
            mix_audio = track.audio  # stéréo, 44100Hz
            if len(mix_audio.shape) == 2:
                mix_audio_mono = np.mean(mix_audio, axis=1)
            else:
                mix_audio_mono = mix_audio
            
            # Créer un fichier temporaire
            temp_path = "temp_mix.wav"
            sf.write(temp_path, mix_audio_mono, track.rate)
            
            # Nom du fichier de sortie
            safe_name = track.name.replace('/', '_').replace('\\', '_')
            output_path = os.path.join(output_dir, f"{safe_name}_vocals.wav")
            
            # Séparer les vocals
            vocals_audio, sr = separate_vocals(temp_path, model, device, output_path)
            
            # Nettoyer le fichier temporaire
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            output_files.append(output_path)
            print(f"  ✓ Sauvegardé : {output_path}")
            
        except Exception as e:
            print(f"  ✗ Erreur : {e}")
            continue
    
    print("\n" + "=" * 70)
    print(f"SÉPARATION TERMINÉE : {len(output_files)}/{len(test_tracks)} fichiers créés")
    print("=" * 70)
    print(f"\nFichiers sauvegardés dans : {output_dir}/")
    print(f"\nPour évaluer ces fichiers, utilisez :")
    print(f"  python evaluate.py --separation-dir {output_dir} --n-tracks {len(output_files)}")
    
    return output_files


def test_inference(audio_path=None, checkpoint_path=None, n_channels=16):
    """
    Tester la fonction d'inférence : séparer la voix d'un fichier audio
    
    Args:
        audio_path: Chemin du fichier audio d'entrée (si None, essaiera de charger depuis MUSDB)
        checkpoint_path: Chemin du checkpoint du modèle (si None, recherche automatique du plus récent)
        n_channels: Nombre de canaux du modèle (doit correspondre à l'entraînement)
    """
    # Si aucun checkpoint n'est spécifié, rechercher automatiquement le plus récent
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint()
        if not checkpoint_path:
            print("✗ Aucun checkpoint trouvé. Entraînez d'abord le modèle: python train.py")
            return
    else:
        if not os.path.exists(checkpoint_path):
            checkpoint_path = find_latest_checkpoint()
            if not checkpoint_path:
                print("✗ Checkpoint introuvable. Entraînez d'abord le modèle: python train.py")
                return
    
    # Charger le modèle
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(checkpoint_path, device, n_channels=n_channels)
    
    # Traiter l'audio
    if audio_path and os.path.exists(audio_path):
        output_path = audio_path.replace('.wav', '_vocals.wav').replace('.mp3', '_vocals.wav')
        if output_path == audio_path:
            output_path = audio_path + '_vocals.wav'
        
        vocals_audio, sr = separate_vocals(audio_path, model, device, output_path)
        
        magnitude, phase = audio_to_spectrogram(vocals_audio[:sr*5], sample_rate=8192)  # 5 premières secondes
        mix_magnitude, _ = audio_to_spectrogram(
            librosa.load(audio_path, sr=8192, duration=5)[0],
            sample_rate=8192
        )
        mix_spec = mix_magnitude  
        mask = predict_mask(model, mix_spec, device)
        estimated_vocals = mask * mix_spec  
        
        visualize_prediction(
            mix_magnitude, mask, estimated_vocals,
            "vocal_inference_result.png"
        )
        
    else:
        # Essayer de charger l'audio de test depuis MUSDB
        try:
            import musdb
            musdb_path = "MUSDB18/musdb18/test"
            if os.path.exists(musdb_path):
                mus = musdb.DB(root=musdb_path, download=False)
                if len(mus.tracks) > 0:
                    track = mus.tracks[0]
                    print(f"Track: {track.name}")
                    
                    mix_audio = track.audio.T
                    if len(mix_audio.shape) > 1:
                        mix_audio = np.mean(mix_audio, axis=0)
                    
                    temp_path = "temp_mix.wav"
                    sf.write(temp_path, mix_audio, 44100)
                    
                    output_path = "output_vocals.wav"
                    vocals_audio, sr = separate_vocals(temp_path, model, device, output_path)
                    
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    
                    print(f"✓ Terminé: {output_path}")
                else:
                    print("✗ Aucun track disponible dans MUSDB")
            else:
                print(f"✗ Dataset MUSDB introuvable: {musdb_path}")
                print("Spécifiez un fichier audio: python inference.py --audio <path>")
        except Exception as e:
            print(f"✗ Erreur: {e}")
            print("Spécifiez un fichier audio: python inference.py --audio <path>")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Inférence de séparation vocale U-Net')
    parser.add_argument('--audio', type=str, default=None, help='Chemin du fichier audio d\'entrée (pour un seul fichier)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Chemin du checkpoint du modèle (si None, recherche automatique)')
    parser.add_argument('--n-channels', type=int, default=16, help='Nombre de canaux du modèle')
    parser.add_argument('--n-songs', type=int, default=None, help='Nombre de chansons MUSDB à séparer (None = mode fichier unique)')
    parser.add_argument('--musdb-path', type=str, default='MUSDB18/musdb18/test', help='Chemin du dataset MUSDB')
    parser.add_argument('--output-dir', type=str, default='vocal_separation', help='Répertoire de sortie pour MUSDB')
    
    args = parser.parse_args()
    
    # Si --n-songs est spécifié, utiliser le mode MUSDB
    if args.n_songs is not None:
        separate_musdb_tracks(
            checkpoint_path=args.checkpoint,
            n_channels=args.n_channels,
            n_songs=args.n_songs,
            musdb_path=args.musdb_path,
            output_dir=args.output_dir
        )
    else:
        # Sinon, mode fichier unique
        test_inference(
            audio_path=args.audio,
            checkpoint_path=args.checkpoint,
            n_channels=args.n_channels
        )
