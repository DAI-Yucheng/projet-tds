# Deuxième Étape : Implémentation du Modèle U-Net

## Vue d'ensemble

Implémentation d'un modèle U-Net simplifié mais correct pour la séparation de sources, conforme aux exigences du TP.

## Exigences du TP

### Structures Obligatoires

1. **Encoder (encodeur)**
   - Conv2D
   - stride = 2 (sous-échantillonnage)
   - Fonction d'activation LeakyReLU

2. **Decoder (décodeur)**
   - ConvTranspose2D (sur-échantillonnage)
   - **Skip connections (point important !)** - connecter les couches correspondantes de l'encoder et du decoder

3. **Dernière couche**
   - Activation Sigmoid
   - Sortie mask ∈ [0, 1]

4. **Fonction de perte**
   - L1 Loss (pas MSE)
   - Formule: `L = || mask ⊙ X - Y ||₁`
   - Où :
     - `mask`: masque prédit par le modèle
     - `X`: spectrogramme magnitude du mix
     - `Y`: spectrogramme magnitude réel des vocals
     - `⊙`: multiplication élément par élément

### Simplifications Possibles

1. ✅ **Pas besoin d'entraîner deux modèles** - seulement la séparation des vocals (pas les instruments)
2. ✅ **Nombre de canaux différent** - peut utiliser moins de canaux (par exemple 32 au lieu de plus dans l'article)
3. ✅ **Simplification du nombre de couches** - 4 couches suffisent (l'article peut en avoir plus)

## Architecture du Modèle

```
Entrée: (batch, 513, 128) - spectrogramme magnitude du mix
    ↓
Ajout dimension canal: (batch, 1, 513, 128)
    ↓
Encoder (sous-échantillonnage):
  Conv2D + LeakyReLU + stride=2  (1 → 32 canaux)
  Conv2D + LeakyReLU + stride=2  (32 → 64 canaux)
  Conv2D + LeakyReLU + stride=2  (64 → 128 canaux)
  Conv2D + LeakyReLU + stride=2  (128 → 256 canaux)
    ↓
Decoder (sur-échantillonnage) + Skip Connections:
  ConvTranspose2D + LeakyReLU  (256 → 128) + skip(128)
  ConvTranspose2D + LeakyReLU  (256 → 64) + skip(64)
  ConvTranspose2D + LeakyReLU  (128 → 32) + skip(32)
  ConvTranspose2D + Sigmoid    (64 → 1)
    ↓
Suppression dimension canal: (batch, 513, 128)
    ↓
Sortie: mask ∈ [0, 1]
    ↓
estimated_vocals = mask ⊙ mix
```

## Description des Fichiers

### `unet_model.py`
- **Classe UNet**: définition du modèle
- **test_unet()**: fonction de test pour vérifier la structure du modèle

### `train.py`
- **Script d'entraînement**: processus d'entraînement complet
- **Classe L1Loss**: implémentation de la perte L1 de l'article
- **Fonction train()**: fonction principale d'entraînement

### `inference.py`
- **Script d'inférence**: utilise le modèle entraîné pour faire des prédictions
- **Fonction de visualisation**: affiche mix, mask et estimated vocals

## Méthode d'Utilisation

### 1. Test de la Structure du Modèle

```bash
python unet_model.py
```

Cela vérifie :
- Nombre de paramètres du modèle
- Shape d'entrée/sortie
- Valeurs du mask dans [0, 1]

### 2. Entraîner le Modèle

```bash
# Entraînement basique (avec paramètres par défaut)
python train.py

# Paramètres personnalisés
python train.py --epochs 20 --batch-size 16 --lr 1e-3 --n-songs 10

# Utiliser le CPU (si pas de GPU)
python train.py --cpu
```

**Explication des paramètres d'entraînement** :
- `--epochs`: Nombre d'époques d'entraînement (recommandé 10-20, objectif : convergence)
- `--batch-size`: Taille du batch (recommandé 8-16)
- `--lr`: Taux d'apprentissage (recommandé 1e-3 à 1e-4)
- `--n-songs`: Nombre de chansons utilisées (recommandé 5-10 chansons, pour test rapide)
- `--save-dir`: Répertoire de sauvegarde du modèle (par défaut : checkpoints)

### 3. Visualiser la Progression de l'Entraînement

```bash
# Lancer TensorBoard
tensorboard --logdir checkpoints/logs

# Puis ouvrir dans le navigateur http://localhost:6006
```

### 4. Utiliser le Modèle pour l'Inférence

```bash
python inference.py
```

## Recommandations pour l'Entraînement (Exigences du TP)

Selon les directives du TP :

1. **Quantité de données**: Choisir 5-10 chansons (pas tout MUSDB)
2. **Nombre d'époques**: 10-20 epochs
3. **Taille du batch**: Petit (8-16)
4. **Objectif**: **Convergence** (pas la recherche de performance)
   - La courbe de perte doit descendre
   - Ne doit pas diverger

## Fichiers de Sortie

Après l'entraînement, les fichiers suivants seront générés :

```
checkpoints/
├── best_model.pth          # Meilleur modèle (perte de validation la plus faible)
├── final_model.pth         # Modèle final
├── checkpoint_epoch_5.pth   # Checkpoint tous les 5 epochs
├── checkpoint_epoch_10.pth
└── logs/                   # Logs TensorBoard
    └── YYYYMMDD_HHMMSS/
```

## Points Clés

### 1. Skip Connections (Point Important !)

C'est la caractéristique centrale du U-Net, elle doit être implémentée :

```python
# L'encoder sauvegarde les cartes de caractéristiques
skip_connections.append(encoder_output)

# Le decoder connecte les couches correspondantes
decoder_input = torch.cat([decoder_output, skip_connection], dim=1)
```

### 2. L1 Loss (Pas MSE)

L'article exige explicitement l'utilisation de la perte L1 :

```python
estimated_vocals = mask * mix_spec
loss = L1Loss(estimated_vocals, vocals_spec)
```

### 3. Domaine de Valeurs du Mask

La dernière couche doit être Sigmoid pour assurer que le mask soit dans [0, 1] :

```python
nn.Sigmoid()  # Dernière couche
```

## Expression dans le Rapport

Vous pouvez écrire dans le rapport :

> "Nous implémentons une version simplifiée du U-Net proposée dans l'article, tout en conservant les principes essentiels (skip connections, masque spectral). Le modèle utilise un encodeur avec des couches Conv2D (stride=2) et LeakyReLU, et un décodeur avec des couches ConvTranspose2D et des connexions de saut. La fonction de perte utilisée est la perte L1: L = || mask ⊙ X - Y ||₁, comme spécifié dans l'article."

## Problèmes Fréquents

### Q: La perte d'entraînement ne descend pas ?

R: 
- Vérifier le taux d'apprentissage (essayer 1e-4)
- Vérifier que les données sont correctement normalisées
- Vérifier si le modèle est trop petit (augmenter n_channels)

### Q: Mémoire insuffisante ?

R:
- Réduire batch_size
- Réduire n_channels ou n_layers

### Q: Comment savoir si le modèle a convergé ?

R:
- La courbe de perte doit descendre
- Ne doit pas diverger (perte qui augmente)
- La perte de validation doit aussi descendre

## Étape Suivante

Après avoir complété cette étape, vous pouvez :
1. Vérifier que le modèle peut s'entraîner et converger normalement
2. Consulter la courbe de perte dans TensorBoard
3. Passer à la troisième étape : reconstruction audio

