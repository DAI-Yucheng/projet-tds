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
        chunk_duration: float = 5.0,
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
        print("Chargement du dataset MUSDB...")
        if musdb_path:
            print(f"   Utilisation du chemin spécifié : {musdb_path}")
            self.mus = musdb.DB(root=musdb_path, download=False)
        else:
            # Vérifier d'abord le dataset complet téléchargé manuellement par l'utilisateur
            default_path = "MUSDB18/musdb18"
            if os.path.exists(default_path):
                print(f"   Utilisation du dataset complet par défaut : {default_path}")
                self.mus = musdb.DB(root=default_path, download=False)
            else:
                print("   ⚠️  ATTENTION : Dataset complet non trouvé !")
                print("   ⚠️  Le téléchargement automatique (musdb.DB(download=True)) télécharge la VERSION DEMO (7 secondes)")
                print("   ⚠️  Pour obtenir le dataset complet (chansons longues), vous devez :")
                print("       1. Télécharger manuellement depuis : https://zenodo.org/records/1117372")
                print("       2. Extraire le fichier musdb18.zip (4.7 GB)")
                print("       3. Placer le dossier 'musdb18' dans : /home/dyc/MUSDB18/")
                print("       4. Ou spécifier le chemin avec : musdb_path='/chemin/vers/musdb18'")
                print()
                print("   ⚠️  Téléchargement de la version demo (tracks courts ~7 secondes)...")
                print("   ⚠️  Cette version n'est PAS adaptée pour l'entraînement sérieux !")
                self.mus = musdb.DB(download=True)
                print("   ⚠️  Version demo téléchargée - recommandé : utiliser le dataset complet")
        print(f"Dataset chargé, {len(self.mus.tracks)} chansons au total")
        
        # Afficher les informations du dataset
        if len(self.mus.tracks) > 0:
            durations = [t.duration for t in self.mus.tracks]
            avg_duration = sum(durations) / len(durations)
            min_duration = min(durations)
            max_duration = max(durations)
            print(f"   Longueur des tracks : {min_duration:.1f}s - {max_duration:.1f}s (moyenne : {avg_duration:.1f}s)")
            
            # Vérifier si les exigences d'entraînement sont satisfaites
            long_enough = [d for d in durations if d >= 12.0]
            if len(long_enough) == len(durations):
                print(f"   ✓ Tous les tracks satisfont l'exigence de 12 secondes, entraînement normal possible")
            elif len(long_enough) > 0:
                print(f"   ⚠️  Seulement {len(long_enough)}/{len(durations)} tracks satisfont l'exigence de 12 secondes")
            else:
                print(f"   ✗ Aucun track ne satisfait l'exigence de 12 secondes, l'entraînement peut échouer")
        
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
        
        # S'assurer que le taux d'échantillonnage est correct
        if len(audio) == 0:
            return np.zeros((self.n_freq_bins, 1))
        
        # Rééchantillonnage au taux d'échantillonnage cible (si différent)
        if original_sr != self.sample_rate:
            audio = librosa.resample(
                audio,
                orig_sr=original_sr,
                target_sr=self.sample_rate
            )
        
        # Exécuter STFT
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window='hann',
            center=True
        )
        
        # Prendre seulement la magnitude (amplitude), pas la phase
        magnitude = np.abs(stft)
        
        # Utiliser seulement les 512 premiers bins de fréquence (au lieu de 513)
        # Cela facilite le traitement par le réseau (puissance de 2)
        magnitude = magnitude[:512, :] # Spectrogramme shape : (freq_bins = 512, time_frames)
        
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
        freq_bins, time_frames = spectrogram.shape
        
        patches = []
        # Utiliser 50% de chevauchement (stride = patch_size // 2) 
        stride = self.patch_hop 
        
        for i in range(0, time_frames - self.patch_frames + 1, stride):
            patch = spectrogram[:, i:i+self.patch_frames]
            if patch.shape[1] == self.patch_frames:  # S'assurer que le patch est complet
                patches.append(patch)
        
        return patches
    
    def generate_batch(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Générateur de batches
        
        Yields:
            (x_batch, y_batch): 
                x_batch: patches de spectrogramme du mix, shape (batch_size, freq_bins, patch_frames)
                y_batch: patches de spectrogramme des vocals, shape (batch_size, freq_bins, patch_frames)
        """
        while True:
            x_batch = []
            y_batch = []
            
            # Collecter suffisamment de patches pour former un batch
            max_retries = 100  # Prévenir la boucle infinie
            retry_count = 0
            
            while len(x_batch) < self.batch_size and retry_count < max_retries:
                try:
                    # Sélectionner aléatoirement une chanson
                    track = random.choice(self.mus.tracks)
                    track_duration = track.duration
                    
                    # Ignorer les tracks trop courts, éviter les problèmes de padding
                    # Nécessite au moins chunk_duration + une marge, pour s'assurer de pouvoir extraire suffisamment de patches
                    min_required_duration = self.chunk_duration
                    if track_duration < min_required_duration:
                        retry_count += 1
                        # Réduire la fréquence d'impression, améliorer la vitesse (imprimer seulement tous les 50)
                        if retry_count % 50 == 0:
                            tqdm.write(
                                f"Ignorer track court : duration={track_duration:.2f}s < "
                                f"required={min_required_duration:.2f}s (ignoré {retry_count} fois)"
                            )
                        continue
                    
                    # Définir les paramètres du chunk
                    track.chunk_duration = self.chunk_duration
                    
                    # Sélectionner aléatoirement la position de départ du chunk (le track est assez long, peut sélectionner aléatoirement)
                    track.chunk_start = random.uniform(
                        0, 
                        track_duration - self.chunk_duration
                    )
                    
                    # Obtenir les données audio
                    # track.audio shape : (samples, channels)
                    # Besoin de transposer en (channels, samples) ou traiter directement
                    mix_audio = track.audio.T  # shape : (channels, samples)
                    vocals_audio = track.targets['vocals'].audio.T  # shape : (channels, samples)
                    
                    # Convertir en spectrogramme (taux d'échantillonnage original de MUSDB est 44100Hz)
                    # Utiliser un traitement par batch plus efficace
                    mix_spec = self.audio_to_spectrogram(mix_audio, original_sr=44100)
                    vocals_spec = self.audio_to_spectrogram(vocals_audio, original_sr=44100)
                    
                    # S'assurer que les dimensions temporelles des deux spectrogrammes sont cohérentes
                    min_time = min(mix_spec.shape[1], vocals_spec.shape[1])
                    if min_time == 0:
                        retry_count += 1
                        continue
                    
                    mix_spec = mix_spec[:, :min_time]
                    vocals_spec = vocals_spec[:, :min_time]
                    
                    # Extraire les patches
                    mix_patches = self.extract_patches(mix_spec)
                    vocals_patches = self.extract_patches(vocals_spec)
                    
                    # S'assurer que le nombre de patches est cohérent
                    min_patches = min(len(mix_patches), len(vocals_patches))
                    if min_patches == 0:
                        retry_count += 1
                        continue
                    
                    mix_patches = mix_patches[:min_patches]
                    vocals_patches = vocals_patches[:min_patches]
                    
                    # Ajouter au batch
                    for mix_patch, vocal_patch in zip(mix_patches, vocals_patches):
                        x_batch.append(mix_patch)
                        y_batch.append(vocal_patch)
                        
                        # Si le batch est plein, sortir de la boucle interne
                        if len(x_batch) >= self.batch_size:
                            break
                    
                    retry_count = 0  # Succès dans l'obtention des données, réinitialiser le compteur de réessai
                    
                except Exception as e:
                    # En cas d'erreur, ignorer cette chanson, continuer à essayer
                    retry_count += 1
                    if retry_count % 10 == 0:
                        print(f"Avertissement : problème rencontré lors de la génération de données (réessai {retry_count}/{max_retries}) : {e}")
                    continue
            
            # Si impossible de collecter suffisamment de patches, utiliser du padding
            if len(x_batch) < self.batch_size:
                # Si au moins quelques données, répéter le dernier patch pour remplir
                if len(x_batch) > 0:
                    while len(x_batch) < self.batch_size:
                        x_batch.append(x_batch[-1])
                        y_batch.append(y_batch[-1])
                else:
                    # Si complètement aucune donnée, créer un remplissage zéro
                    print("Avertissement : impossible de générer des données, utilisation d'un remplissage zéro")
                    dummy_patch = np.zeros((self.n_freq_bins, self.patch_frames))
                    while len(x_batch) < self.batch_size:
                        x_batch.append(dummy_patch)
                        y_batch.append(dummy_patch)
            
            # Convertir en tableau numpy
            x_batch = np.array(x_batch[:self.batch_size])
            y_batch = np.array(y_batch[:self.batch_size])
            
            # ============================================================
            # Méthode simplifiée selon notebook : utiliser directement les magnitude spectrograms
            # ============================================================
            # Pas de log normalization, pas de oracle_mask
            # Entrée : mix magnitude (original)
            # Sortie : vocals magnitude (original)
            # Le modèle apprendra à prédire un mask, puis mask * mix = vocals
            # ============================================================
            
            # Imprimer les informations de diagnostic seulement pour le premier batch
            if not hasattr(self, '_first_batch_printed'):
                tqdm.write(f"\nInformations de diagnostic du premier batch :")
                tqdm.write(f"  mix magnitude - min : {x_batch.min():.4f}, max : {x_batch.max():.4f}, mean : {x_batch.mean():.4f}")
                tqdm.write(f"  vocals magnitude - min : {y_batch.min():.4f}, max : {y_batch.max():.4f}, mean : {y_batch.mean():.4f}")
                tqdm.write(f"  ✓ Utilisation directe des magnitude spectrograms")
                self._first_batch_printed = True
            
            # Retourner directement : mix magnitude et vocals magnitude (pas de normalisation)
            yield x_batch, y_batch
    
    def generate_fixed_validation_set(self, n_batches: int = 15, seed: int = 42) -> list:
        """
        Générer un ensemble de validation fixe (pour la validation de chaque epoch, assurer la cohérence des données)
        
        Args:
            n_batches: Nombre de batches de validation
            seed: Graine aléatoire, assurer que l'ensemble de validation généré est le même à chaque fois
            
        Returns:
            validation_batches: Liste de batches de validation fixes
        """
        # Sauvegarder et définir la graine aléatoire (fixer à la fois python random et numpy.random)
        # Cela permet de s'assurer que l'ensemble de validation est complètement fixe, même si le code suivant utilise np.random
        # Mémorise l'état actuel des générateurs aléatoires Python et NumPy
        old_state = random.getstate()
        old_np_state = np.random.get_state()

        # Fixation de la graine -> seed = 42 (séquence aléatoire) -> À chaque appel avec même seed → mêmes tracks, mêmes positions, mêmes patches
        random.seed(seed)
        np.random.seed(seed)
        
        try:
            val_batches = []
            gen = self.generate_batch()
            
            # Générer un nombre fixe de batches
            for i in range(n_batches):
                try:
                    batch = next(gen)   # Obtenir le batch suivant -> chaque batch contient mix et vocals
                    val_batches.append(batch)
                except StopIteration:
                    break
                except Exception as e:
                    if len(val_batches) >= 5:  # Au moins 5 batches suffisent
                        break
                    continue
            
            return val_batches
        finally:
            # Restaurer l'état aléatoire, ne pas affecter le caractère aléatoire des données d'entraînement
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
        chunk_duration=5.0
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
