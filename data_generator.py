"""
Générateur de données : génère des patches de spectrogramme pour l'entraînement U-Net

Selon les paramètres du papier :
- Taux d'échantillonnage : 8192 Hz
- STFT : window=1024, hop=768
- Longueur de patch : 128 frames
- Chevauchement : un patch tous les 64 frames (50% de chevauchement)
"""

import numpy as np
import librosa
import musdb
from typing import Tuple, Iterator
import random
import os
from tqdm import tqdm

class SpectrogramGenerator:
    """Générateur de patches de spectrogramme"""
    
    def __init__(
        self,
        musdb_path: str = None,
        sample_rate: int = 8192,
        n_fft: int = 1024,
        hop_length: int = 768,
        patch_frames: int = 128,
        patch_hop: int = 64,  # Un patch tous les 64 frames, réalise 50% de chevauchement
        chunk_duration: float = 12.0, 
        batch_size: int = 16
    ):
        """
        Initialisation des paramètres
        
        Args:
            musdb_path: Chemin du dataset MUSDB, None pour téléchargement automatique
            sample_rate: Taux d'échantillonnage (papier : 8192 Hz)
            n_fft: Taille de la fenêtre STFT (papier : 1024)
            hop_length: Longueur du hop STFT (papier : 768)
            patch_frames: Nombre de frames temporelles du patch (papier : 128)
            patch_hop: Longueur du hop entre patches, pour le chevauchement (suggéré : 32, réalise 75% de chevauchement)
            chunk_duration: Longueur du chunk extrait de l'audio (secondes)
            batch_size: Taille du batch
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.patch_frames = patch_frames
        self.patch_hop = patch_hop
        self.chunk_duration = chunk_duration
        self.batch_size = batch_size
        
        # Charger le dataset MUSDB
        if musdb_path:
            self.mus = musdb.DB(root=musdb_path, download=False)
        else:
            default_path = "MUSDB18/musdb18"
            if os.path.exists(default_path):
                self.mus = musdb.DB(root=default_path, download=False)
            else:
                print("⚠️  Dataset non trouvé, décompressez le fichier .zip dans le répertoire MUSDB18/musdb18")
                
        
        num_tracks = len(self.mus.tracks)
        if num_tracks == 0:
            print("⚠️  Aucun track trouvé dans le dataset")
        else:
            print(f"Dataset: {num_tracks} tracks")
        
        # Calculer le nombre de bins de fréquence
        # Utiliser 512 au lieu de 513 pour faciliter le traitement par le réseau (puissance de 2)
        self.n_freq_bins = 512  # Au lieu de n_fft // 2 + 1 (513)
        
    def audio_to_spectrogram(self, audio: np.ndarray, original_sr: int = 44100) -> np.ndarray:
        """
        Convertir l'audio en spectrogramme de magnitude
        
        Args:
            audio: Tableau audio, shape (channels, samples) ou (samples,)
            original_sr: Taux d'échantillonnage original (MUSDB est 44100Hz)
            
        Returns:
            spectrogramme de magnitude, shape (freq_bins, time_frames)
        """
        # Si stéréo, convertir en mono (moyenne)
        if len(audio.shape) == 2:
            audio = np.mean(audio, axis=0)
        
        if len(audio) == 0:
            return np.zeros((self.n_freq_bins, 1))
        
        if original_sr != self.sample_rate:
            audio = librosa.resample(
                audio,
                orig_sr=original_sr,
                target_sr=self.sample_rate
            )
        
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window='hann',
            center=True
        )
        
        magnitude = np.abs(stft)
        
        magnitude = magnitude[:512, :]
        
        return magnitude
    
    def extract_patches(self, spectrogram: np.ndarray) -> list:
        """
        Extraire des patches du spectrogramme (gérer le chevauchement)
        Méthode simplifiée selon notebook : 50% de chevauchement
        
        Args:
            spectrogram: spectrogramme de magnitude, shape (freq_bins, time_frames)
            
        Returns:
            Liste de patches, chaque patch shape (freq_bins, patch_frames)
        """
        _, time_frames = spectrogram.shape
        patches = []
        
        for i in range(0, time_frames - self.patch_frames + 1, self.patch_hop):
            patch = spectrogram[:, i:i+self.patch_frames]
            if patch.shape[1] == self.patch_frames:
                patches.append(patch)
        
        return patches
    
    def _process_track(self, track):
        """
        Traiter un track et extraire des patches
        
        Args:
            track: Track MUSDB
            
        Returns:
            (mix_patches, vocals_patches): Listes de patches ou None si échec
        """
        if track.duration < self.chunk_duration:
            return None, None
        
        # chunk aléatoire
        track.chunk_duration = self.chunk_duration
        track.chunk_start = random.uniform(0, track.duration - self.chunk_duration)
        
        mix_audio = track.audio.T
        vocals_audio = track.targets['vocals'].audio.T
        
        mix_spec = self.audio_to_spectrogram(mix_audio, original_sr=44100)
        vocals_spec = self.audio_to_spectrogram(vocals_audio, original_sr=44100)
        
        min_time = min(mix_spec.shape[1], vocals_spec.shape[1])
        if min_time == 0:
            return None, None
        
        mix_spec = mix_spec[:, :min_time]
        vocals_spec = vocals_spec[:, :min_time]
        
        mix_patches = self.extract_patches(mix_spec)
        vocals_patches = self.extract_patches(vocals_spec)
        
        if len(mix_patches) == 0 or len(vocals_patches) == 0:
            return None, None
        
        min_patches = min(len(mix_patches), len(vocals_patches))
        return mix_patches[:min_patches], vocals_patches[:min_patches]
    
    def generate_batch(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Générateur de batches
        """
        while True:
            x_batch, y_batch = [], []
            retry_count = 0
            max_retries = 100
            
            while len(x_batch) < self.batch_size and retry_count < max_retries:
                try:
                    track = random.choice(self.mus.tracks)
                    mix_patches, vocals_patches = self._process_track(track)
                    
                    if mix_patches is None:
                        retry_count += 1
                        continue
                    
                    # Ajouter les patches au batch
                    for mix_patch, vocal_patch in zip(mix_patches, vocals_patches):
                        x_batch.append(mix_patch)
                        y_batch.append(vocal_patch)
                        if len(x_batch) >= self.batch_size:
                            break
                    
                    retry_count = 0  # Réinitialiser en cas de succès
                    
                except Exception as e:
                    retry_count += 1
                    if retry_count % 50 == 0:
                        tqdm.write(f"Erreur génération (réessai {retry_count}/{max_retries})")
            
            # Padding si nécessaire
            if len(x_batch) < self.batch_size:
                if len(x_batch) > 0:
                    last_patch = (x_batch[-1], y_batch[-1])
                    while len(x_batch) < self.batch_size:
                        x_batch.append(last_patch[0])
                        y_batch.append(last_patch[1])
                else:
                    dummy_patch = np.zeros((self.n_freq_bins, self.patch_frames))
                    while len(x_batch) < self.batch_size:
                        x_batch.append(dummy_patch)
                        y_batch.append(dummy_patch)
            
            x_batch = np.array(x_batch[:self.batch_size])
            y_batch = np.array(y_batch[:self.batch_size])
            
            # Diagnostic (premier batch seulement)
            if not hasattr(self, '_first_batch_printed'):
                tqdm.write(f"Premier batch: mix mean={x_batch.mean():.4f}, vocals mean={y_batch.mean():.4f}")
                self._first_batch_printed = True
            
            yield x_batch, y_batch
    
    def generate_fixed_validation_set(self, n_batches: int = 15, seed: int = 42) -> list:
        """
        Générer un ensemble de validation fixe (même seed = mêmes données à chaque appel)
        
        Args:
            n_batches: Nombre de batches de validation
            seed: Graine aléatoire pour fixer les données
            
        Returns:
            validation_batches: Liste de batches de validation fixes
        """
        # Sauvegarder et fixer la graine aléatoire
        old_state = random.getstate()
        old_np_state = np.random.get_state()
        random.seed(seed)
        np.random.seed(seed)
        
        try:
            val_batches = []
            gen = self.generate_batch()
            
            for _ in range(n_batches):
                try:
                    val_batches.append(next(gen))
                except (StopIteration, Exception):
                    if len(val_batches) >= 5:  # Minimum 5 batches
                        break
                    continue
            
            return val_batches
        finally:
            # Restaurer l'état aléatoire
            random.setstate(old_state)
            np.random.set_state(old_np_state)


def test_generator():
    """Tester si le générateur fonctionne normalement"""
    print("=" * 50)
    print("Test du générateur de données")
    print("=" * 50)
    
    # Créer le générateur
    generator = SpectrogramGenerator(
        batch_size=4,
        chunk_duration=12.0  # Utiliser 12s comme dans l'entraînement
    )
    
    # Obtenir un batch
    gen = generator.generate_batch()
    mix_batch, vocals_batch = next(gen)
    
    print(f"\nShape du batch :")
    print(f"  mix_batch : {mix_batch.shape}")
    print(f"  vocals_batch : {vocals_batch.shape}")
    print(f"\nPlage de données :")
    print(f"  mix : min={mix_batch.min():.4f}, max={mix_batch.max():.4f}, mean={mix_batch.mean():.4f}")
    print(f"  vocals : min={vocals_batch.min():.4f}, max={vocals_batch.max():.4f}, mean={vocals_batch.mean():.4f}")
    
    # Tester plusieurs batches
    print(f"\nGénération de 5 batches pour test...")
    for i in range(5):
        mix, vocals = next(gen)
        print(f"  Batch {i+1} : mix.shape={mix.shape}, vocals.shape={vocals.shape}, vocals.mean={vocals.mean():.4f}")
    
    print("\n✓ Test du générateur de données réussi !")
    print("\nDescription :")
    print("1. Utilisation des paramètres du notebook : sr=8192, n_fft=1024, hop=768")
    print("2. Longueur de patch : 128 frames")
    print("3. Traitement du chevauchement : 50% de chevauchement (stride = patch_size // 2)")
    print("4. Cible d'entraînement : vocals magnitude (direct, pas de oracle_mask)")
    print("5. Pas de log normalization : utilisation directe des magnitude spectrograms")
    print("6. Shape de sortie : (batch_size, 512, patch_frames)")
    print("7. Fréquence bins : 512 (au lieu de 513)")


if __name__ == "__main__":
    # Exécuter le test
    test_generator()
