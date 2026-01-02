# Projet de Séparation de Sources Vocales avec U-Net

## 📋 Vue d'ensemble

Ce projet implémente un modèle U-Net pour séparer la voix (vocals) d'une chanson mixée. Le modèle apprend à prédire un "masque" (mask) qui indique quelle partie du spectrogramme (représentation fréquentielle de l'audio) correspond à la voix.

**En termes simples** : On donne au modèle une chanson complète (avec voix + instruments), et il nous dit "à chaque moment et à chaque fréquence, quelle proportion est de la voix". On peut ensuite extraire uniquement la voix.

---

## 🗂️ Structure du Projet et Relations entre les Fichiers

### Architecture générale

```
Audio (chanson mixée)
    ↓
data_generator.py  →  Génère des données d'entraînement
    ↓
train.py  →  Entraîne le modèle U-Net
    ↓
unet_model.py  →  Définit l'architecture du modèle
    ↓
Modèle entraîné (checkpoint)
    ↓
inference.py  →  Utilise le modèle pour séparer la voix
    ↓
Audio vocal extrait
```

### Fichiers principaux

#### 1. `data_generator.py` - **Générateur de données (Étape 1)**

**Rôle** : Transforme les fichiers audio en données que le modèle peut comprendre.

**Ce qu'il fait** :
- Charge les chansons du dataset MUSDB (mix + voix séparée)
- Convertit l'audio en **spectrogramme** (représentation visuelle du son : fréquence × temps)
- Découpe en petits morceaux (patches) de 128 frames
- Normalise les données pour que le modèle puisse les traiter

**Termes importants** :
- **Spectrogramme** : Représentation 2D du son (fréquence en vertical, temps en horizontal, intensité en couleur)
- **STFT** (Short-Time Fourier Transform) : Méthode pour convertir l'audio en spectrogramme
- **Patch** : Un petit morceau du spectrogramme (128 frames temporelles)
- **Overlap** (chevauchement) : Les patches se chevauchent à 75% pour avoir plus de données d'entraînement

**Sortie** : Des batches de spectrogrammes normalisés (mix et voix)

---

#### 2. `unet_model.py` - **Architecture du modèle (Étape 2)**

**Rôle** : Définit la structure du réseau de neurones U-Net.

**Ce qu'il fait** :
- **Encoder** (encodeur) : Réduit la taille du spectrogramme, extrait des caractéristiques
  - Utilise des **Conv2D** (convolutions 2D) avec stride=2 pour réduire la taille
  - Active avec **LeakyReLU** (fonction d'activation)
  
- **Decoder** (décodeur) : Reconstruit le spectrogramme à la taille originale
  - Utilise des **ConvTranspose2D** (convolutions transposées) pour agrandir
  - **Skip connections** : Connecte les couches de l'encoder au decoder (comme un pont)
  - Dernière couche : **Sigmoid** pour produire un masque entre 0 et 1

**Termes importants** :
- **U-Net** : Architecture de réseau en forme de "U" (réduction puis expansion)
- **Skip connections** : Connexions qui sautent des couches pour préserver les détails
- **Mask** (masque) : Matrice de valeurs entre 0 et 1, indiquant la proportion de voix à chaque point

**Entrée** : Spectrogramme du mix (513 fréquences × 128 frames)  
**Sortie** : Mask (513 fréquences × 128 frames, valeurs entre 0 et 1)

---

#### 3. `train.py` - **Script d'entraînement (Étape 2)**

**Rôle** : Entraîne le modèle U-Net sur les données.

**Ce qu'il fait** :
1. Charge les données via `data_generator.py`
2. Crée le modèle via `unet_model.py`
3. Définit la **fonction de perte** (loss) : Oracle Mask Loss
   - Compare le mask prédit avec le mask "oracle" (vérité terrain)
   - **Oracle mask** = voix / mix (dans le domaine linéaire)
4. Entraîne le modèle avec l'optimiseur Adam
5. Sauvegarde le meilleur modèle dans `checkpoints/`

**Termes importants** :
- **Loss** (perte) : Mesure de l'erreur du modèle (plus c'est bas, mieux c'est)
- **Epoch** : Un passage complet sur toutes les données
- **Batch** : Un groupe d'échantillons traités ensemble
- **Learning rate** (taux d'apprentissage) : Vitesse à laquelle le modèle apprend

**Utilisation** :
```bash
python train.py --epochs 20 --n-songs 10
```

---

#### 4. `inference.py` - **Script d'inférence (Utilisation)**

**Rôle** : Utilise un modèle entraîné pour séparer la voix d'une nouvelle chanson.

**Ce qu'il fait** :
1. Charge un modèle entraîné (checkpoint)
2. Charge un fichier audio (mix)
3. Convertit en spectrogramme (même méthode que l'entraînement)
4. Normalise (même méthode que l'entraînement)
5. Passe dans le modèle → obtient le mask
6. Applique le mask au spectrogramme → obtient le spectrogramme vocal
7. **Reconstruit l'audio** avec ISTFT (inverse de STFT)
8. Sauvegarde le fichier audio vocal

**Termes importants** :
- **Inference** (inférence) : Utiliser un modèle entraîné pour faire des prédictions
- **ISTFT** : Inverse de STFT, reconvertit le spectrogramme en audio

**Utilisation** :
```bash
python inference.py --audio ma_chanson.wav
```

---

## 🔄 Flux de Données Complet

### Phase 1 : Préparation des données

```
Fichiers audio MUSDB (mix.wav, vocals.wav)
    ↓
data_generator.py
    ├─ Charge l'audio
    ├─ Convertit en spectrogramme (STFT)
    ├─ Découpe en patches (128 frames)
    ├─ Calcule oracle_mask = vocals / mix
    └─ Normalise
    ↓
Batches de données (mix_norm, oracle_mask)
```

### Phase 2 : Entraînement

```
Batches de données
    ↓
train.py
    ├─ Crée le modèle (unet_model.py)
    ├─ Passe mix_norm dans le modèle
    ├─ Obtient mask_prédit
    ├─ Compare avec oracle_mask (loss)
    ├─ Ajuste les poids du modèle (backpropagation)
    └─ Répète pour plusieurs epochs
    ↓
Modèle entraîné (checkpoints/best_model.pth)
```

### Phase 3 : Utilisation

```
Nouvelle chanson (mix.wav)
    ↓
inference.py
    ├─ Charge le modèle entraîné
    ├─ Convertit l'audio en spectrogramme
    ├─ Normalise
    ├─ Passe dans le modèle → mask
    ├─ Applique mask au spectrogramme
    ├─ Reconstruit l'audio (ISTFT)
    └─ Sauvegarde vocals.wav
    ↓
Fichier audio vocal extrait
```

---

## 📦 Installation

### 1. Dépendances système

**Important** : `musdb` nécessite `ffmpeg` pour traiter les fichiers audio.

```bash
# Ubuntu/WSL
sudo apt-get update
sudo apt-get install -y ffmpeg

# Vérifier l'installation
ffmpeg -version
```

### 2. Dépendances Python

```bash
# Installer PyTorch (avec CUDA si vous avez un GPU)
pip install torch torchvision torchaudio

# Installer les autres dépendances
pip install -r requirements.txt
```

### 3. Dataset MUSDB

**⚠️ IMPORTANT : Téléchargement du Dataset**

Le dataset MUSDB18 complet doit être **téléchargé manuellement** depuis [Zenodo](https://zenodo.org/records/1117372) (4.7 GB).

**Pourquoi ?**
- `musdb.DB(download=True)` télécharge seulement la **version demo** (7 secondes par track)
- Le dataset complet contient des chansons longues (nécessaires pour l'entraînement)

**Instructions** :
1. Télécharger depuis : https://zenodo.org/records/1117372
2. Extraire le fichier `musdb18.zip` (4.7 GB)
3. Placer le dossier `musdb18` dans : `/home/dyc/MUSDB18/`
4. Structure attendue : `/home/dyc/MUSDB18/musdb18/train/` et `/home/dyc/MUSDB18/musdb18/test/`

Le code détectera automatiquement ce chemin. Si le dataset complet n'est pas trouvé, il demandera confirmation avant de télécharger la version demo.

---

## 🚀 Utilisation

### Étape 1 : Tester le générateur de données

```bash
python data_generator.py
```

Cela vérifie que :
- Le dataset MUSDB est accessible
- Les spectrogrammes sont générés correctement
- Les patches ont la bonne taille (513 × 128)

### Étape 2 : Entraîner le modèle

```bash
# Entraînement de base (20 epochs, 10 chansons)
python train.py

# Personnaliser les paramètres
python train.py --epochs 20 --batch-size 16 --lr 5e-4 --n-songs 10

# Utiliser CPU (si pas de GPU)
python train.py --cpu
```

**Paramètres recommandés** :
- `--epochs` : 10-20 (suffisant pour voir la convergence)
- `--batch-size` : 16 (ajustable selon votre mémoire)
- `--lr` : 5e-4 (taux d'apprentissage)
- `--n-songs` : 5-10 (pour un entraînement rapide)

### Étape 3 : Utiliser le modèle pour séparer la voix

```bash
# Utiliser un fichier audio
python inference.py --audio ma_chanson.wav

# Utiliser le dataset MUSDB (première chanson)
python inference.py
```

Le fichier vocal sera sauvegardé avec le suffixe `_vocals.wav`.

---

## 📊 Paramètres Techniques

### Paramètres du spectrogramme (selon le papier)

- **Taux d'échantillonnage** : 8192 Hz (réduit de 44100 Hz pour accélérer)
- **Taille de fenêtre STFT** : 1024
- **Hop length** : 768
- **Taille de patch** : 128 frames
- **Overlap** : 75% (un patch tous les 32 frames)

### Architecture du modèle

- **Fréquences** : 513 bins (1024/2 + 1)
- **Frames temporelles** : 128
- **Canaux initiaux** : 64
- **Nombre de couches** : 4 (encoder + decoder)

### Fonction de perte

**Oracle Mask Loss** : `L = || mask - oracle_mask ||₁`

- `mask` : Prédiction du modèle (entre 0 et 1)
- `oracle_mask` : Vérité terrain = `vocals / (mix + eps)` (calculé dans le domaine linéaire)
- `|| ||₁` : Norme L1 (somme des valeurs absolues des différences)

**Pourquoi cette méthode ?**
- Supervision directe : le modèle apprend directement à prédire le bon masque
- Plus stable que de prédire le spectrogramme vocal directement
- Évite les problèmes de normalisation

---

## 📁 Structure des Fichiers

```
projet_tds/
├── data_generator.py      # Étape 1 : Génération de données
├── unet_model.py          # Étape 2 : Architecture du modèle
├── train.py               # Étape 2 : Script d'entraînement
├── inference.py           # Utilisation : Séparation de voix
├── requirements.txt       # Dépendances Python
├── README.md              # Ce fichier
│
├── checkpoints/           # Modèles entraînés
│   ├── best_model.pth    # Meilleur modèle
│   └── logs/             # Logs TensorBoard
│
└── (fichiers audio générés)
```

---

## 🔍 Explication des Concepts Clés

### Qu'est-ce qu'un spectrogramme ?

Imaginez une partition musicale : vous avez le temps en horizontal et les notes (fréquences) en vertical. Un spectrogramme est similaire, mais au lieu de notes, vous avez l'intensité du son à chaque fréquence. Plus c'est brillant, plus le son est fort à cette fréquence.

### Qu'est-ce qu'un mask ?

Un mask est comme un "filtre" qui dit "à chaque point du spectrogramme, garde X% du son". Par exemple :
- Mask = 0.8 → garde 80% du son à ce point
- Mask = 0.2 → garde 20% du son à ce point
- Mask = 0.0 → supprime complètement le son à ce point

Le modèle apprend à prédire ce mask pour extraire uniquement la voix.

### Pourquoi U-Net ?

U-Net est une architecture qui :
1. **Réduit** d'abord la taille (encoder) pour comprendre les grandes structures
2. **Agrandit** ensuite (decoder) pour reconstruire les détails
3. Utilise des **skip connections** pour préserver les détails fins

C'est comme regarder une photo de loin pour comprendre la composition, puis zoomer pour voir les détails.

### Pourquoi Oracle Mask ?

Au lieu de dire au modèle "voici le spectrogramme vocal que tu dois produire", on lui dit "voici le masque que tu dois prédire". C'est plus direct et plus stable.

**Oracle mask** = "masque parfait" calculé à partir des données réelles : `vocals / mix`

---

## ⚠️ Problèmes Courants

### Erreur : "ffmpeg or ffprobe could not be found"

**Solution** : Installer ffmpeg
```bash
sudo apt-get install -y ffmpeg
```

### Erreur : "ModuleNotFoundError: No module named 'torch'"

**Solution** : Installer PyTorch
```bash
pip install torch torchvision torchaudio
```

### Le loss ne diminue pas

**Causes possibles** :
- Taux d'apprentissage trop élevé → réduire `--lr` à 1e-4
- Données mal normalisées → vérifier `data_generator.py`
- Modèle trop petit → augmenter `n_channels` dans `unet_model.py`

### Mémoire insuffisante

**Solutions** :
- Réduire `--batch-size` (ex: 8 au lieu de 16)
- Réduire `n_channels` dans le modèle (ex: 32 au lieu de 64)

---

## 📈 Résultats Attendus

Après l'entraînement, vous devriez voir :
- **Loss initiale** : ~0.3-0.4
- **Loss finale** : ~0.1-0.15 (si le modèle apprend bien)
- **Mask std** : > 0.15 (indique que le mask varie, pas constant)

Le modèle sauvegardé dans `checkpoints/best_model.pth` peut être utilisé pour séparer la voix de nouvelles chansons.

---

## 📚 Références

- **Dataset** : MUSDB18 (https://sigsep.github.io/datasets/musdb.html)
- **Architecture** : U-Net (adapté pour la séparation de sources)
- **Papier de référence** : Voir `TP_M2SON (1).pdf`

---

## 🎯 Objectifs du Projet (TP)

1. ✅ **Étape 1** : Implémenter la génération de données (spectrogrammes avec overlap)
2. ✅ **Étape 2** : Implémenter et entraîner le modèle U-Net
3. ✅ **Objectif** : Faire converger le modèle (pas nécessairement obtenir les meilleures performances)

**Note** : Ce projet est une version simplifiée pour l'apprentissage. Les performances peuvent être améliorées avec plus de données, un modèle plus grand, et un entraînement plus long.

---

## 💡 Conseils

- Commencez avec peu de chansons (5-10) pour tester rapidement
- Surveillez le loss : il devrait diminuer, pas augmenter
- Si le loss stagne, essayez de réduire le taux d'apprentissage
- Utilisez TensorBoard pour visualiser les courbes d'entraînement

---

**Bon entraînement ! 🎵**
