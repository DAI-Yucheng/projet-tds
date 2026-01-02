# Réponses aux Questions

## Question 1 : Téléchargement du Dataset

### ❌ Problème Identifié

**`musdb.DB(download=True)` télécharge la VERSION DEMO (7 secondes), pas le dataset complet !**

Selon la [documentation Zenodo](https://zenodo.org/records/1117372), le dataset complet MUSDB18 :
- Contient **150 chansons complètes** (100 train + 50 test)
- Fait **4.7 GB**
- Les chansons sont **longues** (plusieurs minutes)

La version demo téléchargée automatiquement :
- Contient seulement **quelques tracks courts** (~7 secondes)
- **Inadaptée pour l'entraînement** (trop de padding)

### ✅ Solution Implémentée

J'ai modifié `data_generator.py` pour :

1. **Vérifier d'abord le dataset complet** dans `/home/dyc/MUSDB18/musdb18`
2. **Afficher un avertissement clair** si le dataset complet n'est pas trouvé
3. **Demander confirmation** avant de télécharger la version demo
4. **Donner des instructions** pour télécharger le dataset complet depuis Zenodo

**Code modifié** :
```python
if os.path.exists(default_path):
    # Utiliser le dataset complet
    self.mus = musdb.DB(root=default_path, download=False)
else:
    # Avertissement + demande de confirmation
    print("⚠️  ATTENTION : Dataset complet non trouvé !")
    print("⚠️  Le téléchargement automatique télécharge la VERSION DEMO (7 secondes)")
    print("⚠️  Pour obtenir le dataset complet, télécharger depuis :")
    print("   https://zenodo.org/records/1117372")
    response = input("Continuer avec la version demo ? (o/n) : ")
    if response in ['o', 'oui']:
        self.mus = musdb.DB(download=True)  # Version demo
    else:
        raise FileNotFoundError("Dataset complet requis")
```

### 📥 Instructions pour Télécharger le Dataset Complet

1. Aller sur : https://zenodo.org/records/1117372
2. Cliquer sur "Download" → `musdb18.zip` (4.7 GB)
3. Extraire le fichier zip
4. Placer le dossier `musdb18` dans `/home/dyc/MUSDB18/`
5. Structure finale : `/home/dyc/MUSDB18/musdb18/train/` et `test/`

---

## Question 2 : Utilisation du Code du TP

### ✅ Oui, Nous Avons Utilisé la Structure de Base du TP

Le TP fournit un générateur "naive" (naïf) comme point de départ :

```python
# Code du TP (image)
while True:
    track = random.choice(mus.tracks)
    track.chunk_duration = 5.0
    track.chunk_start = random.uniform(0, track.duration - track.chunk_duration)
    x = track.audio.T
    y = track.targets['vocals'].audio.T
    yield x, y
```

### Comparaison avec Notre Code

#### ✅ Parties Identiques (Structure de Base)

| Code TP | Notre Code | Statut |
|---------|------------|--------|
| `track = random.choice(mus.tracks)` | `track = random.choice(self.mus.tracks)` | ✅ Identique |
| `track.chunk_duration = 5.0` | `track.chunk_duration = self.chunk_duration` | ✅ Identique (nous: 12.0) |
| `track.chunk_start = random.uniform(...)` | `track.chunk_start = random.uniform(...)` | ✅ Identique |
| `x = track.audio.T` | `mix_audio = track.audio.T` | ✅ Identique |
| `y = track.targets['vocals'].audio.T` | `vocals_audio = track.targets['vocals'].audio.T` | ✅ Identique |
| `yield x, y` | `yield x_batch_norm, oracle_mask` | ✅ Structure identique |

#### ➕ Extensions Ajoutées (Selon les Exigences du TP)

1. **Conversion en spectrogramme** (requis par le TP) :
   ```python
   mix_spec = self.audio_to_spectrogram(mix_audio, original_sr=44100)
   vocals_spec = self.audio_to_spectrogram(vocals_audio, original_sr=44100)
   ```
   - Utilise les paramètres du papier : `n_fft=1024`, `hop_length=768`, `sr=8192`

2. **Extraction de patches avec overlap** (répond à la question du TP) :
   ```python
   mix_patches = self.extract_patches(mix_spec)  # Patches de 128 frames
   # Overlap de 75% : un patch tous les 32 frames
   ```
   - **Répond à la question** : "on observe un fort taux de recouvrement entre deux spectrogrammes de 128 trames"
   - **Solution** : Utiliser une fenêtre glissante avec chevauchement de 75%

3. **Organisation en batches** :
   ```python
   # Collecte plusieurs patches pour former un batch
   for mix_patch, vocal_patch in zip(mix_patches, vocals_patches):
       x_batch.append(mix_patch)
       y_batch.append(vocal_patch)
   ```

4. **Calcul de l'Oracle Mask** (pour l'entraînement) :
   ```python
   oracle_mask = y_batch / (x_batch + eps)  # Calcul dans le domaine linéaire
   ```

5. **Normalisation** :
   ```python
   x_batch_norm = normalize_spectrogram(x_batch)  # Log + clip + mapping [0,1]
   ```

### 📊 Résumé

**Oui, nous avons utilisé la structure de base du TP**, mais nous l'avons **adaptée et étendue** pour :

- ✅ Générer des **spectrogrammes** (pas seulement de l'audio brut)
- ✅ Gérer l'**overlap** (75%) entre patches (répond à la question du TP)
- ✅ Organiser les données en **batches**
- ✅ Calculer l'**Oracle Mask** pour l'entraînement
- ✅ Normaliser les données correctement

**Notre code est donc une extension complète du générateur naïf du TP**, qui répond à toutes les exigences du TP tout en conservant la structure de base fournie.

