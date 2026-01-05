"""
Évaluation du modèle U-Net pour la séparation de sources vocales

Utilise la bibliothèque museval pour calculer les métriques standard :
- SDR : Signal-to-Distortion Ratio
- SIR : Signal-to-Interference Ratio  
- SAR : Signal-to-Artifacts Ratio

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


def evaluate_separated_vocals(
    musdb_path="MUSDB18/musdb18",
    separation_dir="vocal_separation",
    n_tracks=None,
    output_dir="./eval"
):
    """
    Évaluer les vocals déjà séparés avec museval
    
    Args:
        musdb_path: Chemin du dataset MUSDB (pour les références)
        separation_dir: Répertoire contenant les vocals séparés
        n_tracks: Nombre de tracks à évaluer (None = tous les fichiers disponibles)
        output_dir: Répertoire pour sauvegarder les résultats
        
    Returns:
        results_df: DataFrame avec les résultats pour chaque track
        summary: Dictionnaire avec les moyennes globales
    """
    print("=" * 70)
    print("ÉVALUATION DES VOCALS SÉPARÉS")
    print("=" * 70)
    
    # Vérifier que le répertoire de séparation existe
    if not os.path.exists(separation_dir):
        print(f"\n✗ Répertoire introuvable : {separation_dir}")
        print(f"   Veuillez d'abord séparer les vocals avec :")
        print(f"   python inference.py --n-songs N")
        return None, None
    
    # Charger le dataset MUSDB (pour les références)
    print(f"\nChargement du dataset MUSDB depuis : {musdb_path}")
    try:
        mus = musdb.DB(root=musdb_path, download=False)
    except Exception as e:
        print(f"✗ Erreur lors du chargement : {e}")
        return None, None
    
    # Obtenir les tracks de test
    test_tracks = [t for t in mus.tracks if t.subset == 'test']
    
    # Lister les fichiers séparés disponibles
    separated_files = []
    for f in os.listdir(separation_dir):
        if f.endswith('_vocals.wav'):
            separated_files.append(f)
    
    # Trier par ordre alphabétique pour un ordre cohérent
    separated_files.sort()
    
    print(f"\nFichiers séparés trouvés : {len(separated_files)}")
    
    if len(separated_files) == 0:
        print("✗ Aucun fichier séparé trouvé")
        print(f"   Veuillez d'abord séparer les vocals avec :")
        print(f"   python inference.py --n-songs N")
        return None, None
    
    # Limiter le nombre de tracks si spécifié
    if n_tracks is None:
        # Par défaut : 8 premières chansons
        separated_files = separated_files[:8]
        print(f"Évaluation des 8 premières chansons : {len(separated_files)} fichiers")
    elif n_tracks == 9999:
        # 9999 = tous les fichiers
        print(f"Évaluation de TOUS les fichiers : {len(separated_files)} fichiers")
    else:
        separated_files = separated_files[:n_tracks]
        print(f"Évaluation limitée à : {len(separated_files)} fichiers")
    
    # Créer le répertoire de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # Liste pour stocker les résultats
    results = []
    
    # Évaluer chaque fichier
    print("\n" + "=" * 70)
    print("ÉVALUATION EN COURS")
    print("=" * 70)
    
    for i, vocals_file in enumerate(separated_files):
        try:
            # Extraire le nom du track depuis le nom du fichier
            track_name = vocals_file.replace('_vocals.wav', '').replace('_', '/')
            
            # Trouver le track correspondant dans MUSDB
            matching_track = None
            for track in test_tracks:
                if track.name in track_name or track_name in track.name:
                    matching_track = track
                    break
            
            if matching_track is None:
                print(f"\n[{i+1}/{len(separated_files)}] ⚠️  Track non trouvé : {track_name}")
                continue
            
            print(f"\n[{i+1}/{len(separated_files)}] Évaluation : {matching_track.name}")
            
            # Charger les vocals séparés
            vocals_path = os.path.join(separation_dir, vocals_file)
            vocals_audio, sr = librosa.load(vocals_path, sr=matching_track.rate, mono=True)
            
            # Ajuster la longueur
            target_length = matching_track.audio.shape[0]
            vocals_audio = librosa.util.fix_length(vocals_audio, size=target_length)
            
            # Calculer l'accompagnement
            mix_mono = np.mean(matching_track.audio, axis=1)
            accompaniment_audio = mix_mono - vocals_audio
            
            # Convertir en stéréo pour museval
            vocals_stereo = np.stack([vocals_audio, vocals_audio], axis=1)
            accompaniment_stereo = np.stack([accompaniment_audio, accompaniment_audio], axis=1)
            
            # Créer le dictionnaire d'estimates
            estimates = {
                'vocals': vocals_stereo,
                'accompaniment': accompaniment_stereo
            }
            
            # Évaluer avec museval
            scores = museval.eval_mus_track(
                matching_track,
                estimates,
                output_dir=output_dir
            )
            
            # Extraire les scores
            df = scores.df
            vocals_sdr = df[(df.target == 'vocals') & (df.metric == 'SDR')]['score'].mean()
            vocals_sir = df[(df.target == 'vocals') & (df.metric == 'SIR')]['score'].mean()
            vocals_sar = df[(df.target == 'vocals') & (df.metric == 'SAR')]['score'].mean()
            
            # Stocker les résultats
            results.append({
                'track': matching_track.name,
                'SDR': vocals_sdr,
                'SIR': vocals_sir,
                'SAR': vocals_sar
            })
            
            print(f"  ✓ SDR: {vocals_sdr:.2f} dB | SIR: {vocals_sir:.2f} dB | SAR: {vocals_sar:.2f} dB")
            
        except Exception as e:
            print(f"  ✗ Erreur : {e}")
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
        print("\n Aucun résultat à afficher")
        return None, None


def main():
    """Fonction principale pour l'évaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Évaluer les vocals séparés avec museval'
    )
    parser.add_argument(
        '--musdb-path',
        type=str,
        default='MUSDB18/musdb18',
        help='Chemin du dataset MUSDB'
    )
    parser.add_argument(
        '--separation-dir',
        type=str,
        default='vocal_separation',
        help='Répertoire contenant les vocals séparés'
    )
    parser.add_argument(
        '--n-tracks',
        type=int,
        default=None,
        help='Nombre de tracks à évaluer (None = tous les fichiers disponibles)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./eval',
        help='Répertoire pour sauvegarder les résultats'
    )
    
    args = parser.parse_args()
    
    # Lancer l'évaluation
    results_df, summary = evaluate_separated_vocals(
        musdb_path=args.musdb_path,
        separation_dir=args.separation_dir,
        n_tracks=args.n_tracks,
        output_dir=args.output_dir
    )
    
    if results_df is not None:
        print("\n" + "=" * 70)
        print("ÉVALUATION TERMINÉE")
        print("=" * 70)


if __name__ == "__main__":
    main()

