# Première Étape : Implémentation du Pipeline de Données

## Vue d'ensemble

Cette étape implémente le pipeline de données pour générer des patches de spectrogrammes à partir du dataset MUSDB, en suivant strictement les paramètres de l'article.

## Paramètres de l'Article

- **Taux d'échantillonnage**: 8192 Hz
- **Fenêtre STFT**: 1024
- **Hop STFT**: 768
- **Longueur de patch**: 128 trames
- **Shape d'entrée**: (freq_bins, 128) ≈ (512,128)

## Points Clés d'Implémentation

### 1. Traitement du Recouvrement (Question requise par le TP)

**Question**: "on observe un fort taux de recouvrement entre deux spectrogrammes de 128 trames"

**Solution**: Utilisation d'une fenêtre glissante avec recouvrement

- **Méthode d'implémentation**: Prendre un patch tous les 32 trames (au lieu de tous les 128 trames)
- **Taux de recouvrement**: (128-32)/128 = 75%
- **Avantages**: 
  - Augmenter le nombre d'exemples d'entraînement (environ 4 fois)
  - Maintenir la continuité temporelle
  - Améliorer la capacité de prédiction du modèle dans les zones frontières

**Expression dans le rapport**:
> "Nous utilisons une fenêtre glissante avec recouvrement afin d'augmenter le nombre d'exemples d'apprentissage tout en conservant la continuité temporelle."

### 2. Prétraitement des Données

- **STFT**: Utilise la STFT de librosa, avec des paramètres strictement conformes à l'article
- **Magnitude uniquement**: Ignore la phase, qui sera celle du mix lors de la reconstruction
- **Normalisation**: 
  - D'abord échelle logarithmique (log(x + eps))
  - Puis normalisation min-max vers [0, 1]

### 3. Conception du Générateur

- **Boucle infinie**: `while True`, peut générer des données en continu
- **Échantillonnage aléatoire**: Sélection aléatoire de la chanson et de la position du chunk à chaque fois
- **Organisation en batches**: Collecte automatiquement des patches pour former un batch

## Méthode d'Utilisation

### Installation des Dépendances Système (Important !)

**musdb nécessite ffmpeg pour traiter les fichiers audio, il faut l'installer en premier :**

Dans Ubuntu/WSL :
```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

Vérification de l'installation :
```bash
ffmpeg -version
```

### Installation des Dépendances Python

```bash
pip install -r requirements.txt
```

### Exécution du Test

```bash
python data_generator.py
```

### Utilisation dans l'Entraînement

```python
from data_generator import SpectrogramGenerator

# Créer le générateur
generator = SpectrogramGenerator(
    batch_size=16,
    chunk_duration=5.0
)

# Obtenir les données
gen = generator.generate_batch()
for epoch in range(num_epochs):
    for batch_idx in range(batches_per_epoch):
        x_batch, y_batch = next(gen)
        # x_batch: (batch_size, 513, 128) - spectrogramme du mix
        # y_batch: (batch_size, 513, 128) - spectrogramme des vocals
        
        # Entraîner le modèle...
```

## Format de Sortie

- **x_batch**: patches de spectrogramme magnitude du mix
  - Shape: `(batch_size, freq_bins, patch_frames)`
  - Exemple: `(16, 513, 128)`
  - Plage de valeurs: [0, 1] (normalisé)

- **y_batch**: patches de spectrogramme magnitude des vocals
  - Shape: `(batch_size, freq_bins, patch_frames)`
  - Exemple: `(16, 513, 128)`
  - Plage de valeurs: [0, 1] (normalisé)

## Points d'Attention

1. **Dépendances système**: Il faut d'abord installer ffmpeg (voir les étapes d'installation ci-dessus)
2. **Première exécution**: Si MUSDB n'est pas téléchargé, il se téléchargera automatiquement (environ 4.5GB)
3. **Mémoire**: Si la mémoire est insuffisante, réduire `batch_size` ou `chunk_duration`
4. **Taux d'échantillonnage**: Le taux d'échantillonnage original de MUSDB est 44100Hz, le code rééchantillonne automatiquement à 8192Hz
5. **Traitement stéréo**: Conversion automatique en mono (moyenne)

## Problèmes Fréquents

### Erreur: "ffmpeg or ffprobe could not be found"

**Cause**: Le système manque de ffmpeg, qui est une dépendance nécessaire de musdb.

**Solution**:
```bash
# Ubuntu/WSL
sudo apt-get update
sudo apt-get install -y ffmpeg

# Vérification
ffmpeg -version
```

## Étape Suivante

Après avoir complété cette étape, vous pouvez :
1. Vérifier que la forme et la plage de valeurs des données sont correctes
2. Visualiser quelques patches de spectrogrammes pour vérifier la qualité des données
3. Passer à la deuxième étape : implémentation du modèle U-Net


