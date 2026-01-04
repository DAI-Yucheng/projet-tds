"""
Évaluation du modèle U-Net pour la séparation de sources vocales

Utilise la bibliothèque museval pour calculer les métriques standard :
- SDR (Signal-to-Distortion Ratio) : Signal-to-Distortion Ratio
- SIR (Signal-to-Interference Ratio) : Signal-to-Interference Ratio  
- SAR (Signal-to-Artifacts Ratio) : Signal-to-Artifacts Ratio

Référence : https://github.com/sigsep/sigsep-mus-eval
"""

import torch
import numpy as np
import librosa
import musdb
import museval
import pandas as pd
import os
from tqdm import tqdm
from unet_model import UNet
from inference import (
    load_model,
    audio_to_spectrogram,
    spectrogram_to_audio,
    predict_mask
)


def separate_track(model, track, device='cpu', sample_rate=8192, n_fft=1024, hop_length=768):
    """
    Séparer une piste complète en vocals et accompaniment
    
    Args:
        model: Modèle entraîné
        track: Track MUSDB (contient track.audio et track.targets)
        device: Appareil ('cpu' ou 'cuda')
        sample_rate: Taux d'échantillonnage utilisé pour le traitement
        n_fft: Taille de la fenêtre STFT
        hop_length: Longueur du hop STFT
        
    Returns:
        vocals_audio: Audio vocal séparé (mono, 44100Hz)
        accompaniment_audio: Audio d'accompagnement séparé (mono, 44100Hz)
        original_sr: Taux d'échantillonnage original (44100Hz)
    """
    # Obtenir l'audio du mix (stéréo, 44100Hz)
    mix_audio = track.audio  # shape: (samples, channels)
    original_sr = track.rate  # 44100 Hz
    
    # Convertir en mono pour le traitement (moyenne des canaux)
    if len(mix_audio.shape) == 2:
        mix_audio_mono = np.mean(mix_audio, axis=1)
    else:
        mix_audio_mono = mix_audio
    
    # Convertir en spectrogramme (rééchantillonnage à 8192Hz pour le traitement)
    magnitude, phase = audio_to_spectrogram(
        mix_audio_mono,
        original_sr=original_sr,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    # Traitement par patches (pour les longs audios)
    patch_frames = 128
    hop_frames = patch_frames // 2  # 50% de chevauchement
    
    if magnitude.shape[1] <= patch_frames:
        # Audio court, traitement direct
        mask = predict_mask(model, magnitude, device)
        estimated_vocals_magnitude = mask * magnitude
    else:
        # Audio long, traitement par blocs avec chevauchement
        estimated_vocals_magnitude = np.zeros_like(magnitude)
        weight = np.zeros_like(magnitude)
        
        for start in range(0, magnitude.shape[1] - patch_frames + 1, hop_frames):
            end = start + patch_frames
            patch = magnitude[:, start:end]
            
            mask_patch = predict_mask(model, patch, device)
            estimated_patch = mask_patch * patch
            
            estimated_vocals_magnitude[:, start:end] += estimated_patch
            weight[:, start:end] += 1.0
        
        # Traiter la partie restante
        if magnitude.shape[1] > patch_frames:
            patch = magnitude[:, -patch_frames:]
            mask_patch = predict_mask(model, patch, device)
            estimated_patch = mask_patch * patch
            estimated_vocals_magnitude[:, -patch_frames:] += estimated_patch
            weight[:, -patch_frames:] += 1.0
        
        # Normaliser pour gérer le chevauchement
        estimated_vocals_magnitude /= (weight + 1e-8)
    
    # Reconstruire l'audio vocal (8192Hz)
    vocals_audio_8192 = spectrogram_to_audio(
        estimated_vocals_magnitude,
        phase,
        hop_length=hop_length,
        n_fft=n_fft
    )
    
    # Rééchantillonnage à 44100Hz
    vocals_audio = librosa.resample(
        vocals_audio_8192,
        orig_sr=sample_rate,
        target_sr=original_sr
    )
    
    # Calculer l'accompagnement : mix - vocals
    # Ajuster la longueur si nécessaire
    target_length = len(mix_audio_mono)
    vocals_audio = librosa.util.fix_length(vocals_audio, size=target_length)
    
    # Accompaniment = mix - vocals
    accompaniment_audio = mix_audio_mono - vocals_audio
    
    return vocals_audio, accompaniment_audio, original_sr


def evaluate_model(
    checkpoint_path,
    musdb_path="MUSDB18/musdb18",
    n_channels=16,
    n_tracks=None,
    output_dir="./eval",
    device=None
):
    """
    Évaluer le modèle sur le test set de MUSDB
    
    Args:
        checkpoint_path: Chemin du checkpoint du modèle
        musdb_path: Chemin du dataset MUSDB
        n_channels: Nombre de canaux du modèle
        n_tracks: Nombre de tracks à évaluer (None = tous)
        output_dir: Répertoire pour sauvegarder les résultats
        device: Appareil ('cpu' ou 'cuda', None = auto)
        
    Returns:
        results_df: DataFrame avec les résultats pour chaque track
        summary: Dictionnaire avec les moyennes globales
    """
    print("=" * 70)
    print("ÉVALUATION DU MODÈLE U-NET POUR LA SÉPARATION VOCALE")
    print("=" * 70)
    
    # Déterminer l'appareil
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nAppareil utilisé : {device}")
    
    # Charger le modèle
    print(f"\nChargement du modèle depuis : {checkpoint_path}")
    model = load_model(checkpoint_path, device, n_channels=n_channels)
    
    # Charger le dataset MUSDB
    print(f"\nChargement du dataset MUSDB depuis : {musdb_path}")
    if not os.path.exists(musdb_path):
        print(f"⚠️  Chemin introuvable : {musdb_path}")
        print("   Tentative avec le chemin par défaut...")
        musdb_path = "MUSDB18/musdb18"
    
    try:
        mus = musdb.DB(root=musdb_path, download=False)
    except Exception as e:
        print(f"❌ Erreur lors du chargement du dataset : {e}")
        return None, None
    
    # Obtenir les tracks de test
    test_tracks = [t for t in mus.tracks if t.subset == 'test']
    if n_tracks is not None:
        test_tracks = test_tracks[:n_tracks]
    
    print(f"\nNombre de tracks de test à évaluer : {len(test_tracks)}")
    
    # Créer le répertoire de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # Liste pour stocker les résultats
    results = []
    
    # Évaluer chaque track
    print("\n" + "=" * 70)
    print("ÉVALUATION DES TRACKS")
    print("=" * 70)
    
    for i, track in enumerate(tqdm(test_tracks, desc="Évaluation")):
        try:
            print(f"\n[{i+1}/{len(test_tracks)}] Évaluation de : {track.name}")
            
            # Séparer la piste
            vocals_audio, accompaniment_audio, original_sr = separate_track(
                model, track, device=device
            )
            
            # Préparer les estimates pour museval
            # museval nécessite un format stéréo (samples, channels)
            target_length = track.audio.shape[0]
            
            # Ajuster la longueur si nécessaire
            vocals_audio = librosa.util.fix_length(vocals_audio, size=target_length)
            accompaniment_audio = librosa.util.fix_length(accompaniment_audio, size=target_length)
            
            # Convertir en stéréo (dupliquer le canal mono)
            vocals_stereo = np.stack([vocals_audio, vocals_audio], axis=1)
            accompaniment_stereo = np.stack([accompaniment_audio, accompaniment_audio], axis=1)
            
            # Créer le dictionnaire d'estimates
            estimates = {
                'vocals': vocals_stereo,
                'accompaniment': accompaniment_stereo
            }
            
            # Évaluer avec museval
            scores = museval.eval_mus_track(
                track,
                estimates,
                output_dir=output_dir
            )
            
            # Extraire les scores depuis le DataFrame
            df = scores.df
            
            # Calculer les moyennes pour vocals
            vocals_sdr = df[(df.target == 'vocals') & (df.metric == 'SDR')]['score'].mean()
            vocals_sir = df[(df.target == 'vocals') & (df.metric == 'SIR')]['score'].mean()
            vocals_sar = df[(df.target == 'vocals') & (df.metric == 'SAR')]['score'].mean()
            
            # Stocker les résultats
            results.append({
                'track': track.name,
                'SDR': vocals_sdr,
                'SIR': vocals_sir,
                'SAR': vocals_sar
            })
            
            print(f"  ✓ SDR: {vocals_sdr:.2f} dB | SIR: {vocals_sir:.2f} dB | SAR: {vocals_sar:.2f} dB")
            
        except Exception as e:
            print(f"  ❌ Erreur lors de l'évaluation de {track.name}: {e}")
            continue
    
    # Créer le DataFrame des résultats
    results_df = pd.DataFrame(results)
    
    # Calculer les moyennes globales
    if len(results_df) > 0:
        summary = {
            'SDR_mean': results_df['SDR'].mean(),
            'SIR_mean': results_df['SIR'].mean(),
            'SAR_mean': results_df['SAR'].mean(),
            'SDR_std': results_df['SDR'].std(),
            'SIR_std': results_df['SIR'].std(),
            'SAR_std': results_df['SAR'].std()
        }
        
        # Afficher les résultats globaux
        print("\n" + "=" * 70)
        print("RÉSULTATS GLOBAUX")
        print("=" * 70)
        print(f"\nNombre de tracks évalués : {len(results_df)}")
        print(f"\nMétriques moyennes (vocals) :")
        print(f"  SDR : {summary['SDR_mean']:.2f} ± {summary['SDR_std']:.2f} dB")
        print(f"  SIR : {summary['SIR_mean']:.2f} ± {summary['SIR_std']:.2f} dB")
        print(f"  SAR : {summary['SAR_mean']:.2f} ± {summary['SAR_std']:.2f} dB")
        
        # Sauvegarder les résultats dans un fichier CSV
        csv_path = os.path.join(output_dir, 'evaluation_results.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"\n✓ Résultats sauvegardés dans : {csv_path}")
        
        # Sauvegarder le résumé
        summary_path = os.path.join(output_dir, 'summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("RÉSULTATS GLOBAUX D'ÉVALUATION\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Nombre de tracks évalués : {len(results_df)}\n\n")
            f.write("Métriques moyennes (vocals) :\n")
            f.write(f"  SDR : {summary['SDR_mean']:.2f} ± {summary['SDR_std']:.2f} dB\n")
            f.write(f"  SIR : {summary['SIR_mean']:.2f} ± {summary['SIR_std']:.2f} dB\n")
            f.write(f"  SAR : {summary['SAR_mean']:.2f} ± {summary['SAR_std']:.2f} dB\n")
        print(f"✓ Résumé sauvegardé dans : {summary_path}")
        
        return results_df, summary
    else:
        print("\n❌ Aucun résultat à afficher")
        return None, None


def main():
    """Fonction principale pour l'évaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Évaluer le modèle U-Net pour la séparation vocale'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Chemin du checkpoint du modèle (si None, cherche automatiquement)'
    )
    parser.add_argument(
        '--musdb-path',
        type=str,
        default='MUSDB18/musdb18',
        help='Chemin du dataset MUSDB'
    )
    parser.add_argument(
        '--n-channels',
        type=int,
        default=16,
        help='Nombre de canaux du modèle'
    )
    parser.add_argument(
        '--n-tracks',
        type=int,
        default=None,
        help='Nombre de tracks à évaluer (None = tous)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./eval',
        help='Répertoire pour sauvegarder les résultats'
    )
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Forcer l\'utilisation du CPU'
    )
    
    args = parser.parse_args()
    
    # Trouver le checkpoint si non spécifié
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        # Chercher dans vocal_checkpoints ou checkpoints
        possible_paths = [
            'vocal_checkpoints/best_model.pth',
            'checkpoints/best_model.pth',
            'vocal_checkpoints/final_model.pth',
            'checkpoints/final_model.pth'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                checkpoint_path = path
                print(f"✓ Checkpoint trouvé : {checkpoint_path}")
                break
        
        if checkpoint_path is None:
            print("❌ Aucun checkpoint trouvé. Veuillez spécifier --checkpoint")
            return
    
    # Déterminer l'appareil
    device = 'cpu' if args.cpu else None
    
    # Lancer l'évaluation
    results_df, summary = evaluate_model(
        checkpoint_path=checkpoint_path,
        musdb_path=args.musdb_path,
        n_channels=args.n_channels,
        n_tracks=args.n_tracks,
        output_dir=args.output_dir,
        device=device
    )
    
    if results_df is not None:
        print("\n" + "=" * 70)
        print("ÉVALUATION TERMINÉE")
        print("=" * 70)


if __name__ == "__main__":
    main()

