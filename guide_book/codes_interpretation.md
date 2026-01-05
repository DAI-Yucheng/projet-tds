## Fichiers principaux

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


#### 5. `evaluate.py` - Script principal d'évaluation

**Fonctionnalités principales** :
-  Charge le modèle entraîné depuis un checkpoint
-  Sépare les tracks du test set MUSDB en vocals et accompaniment
-  Utilise `museval` pour calculer SDR, SIR, SAR
-  Génère un rapport CSV avec les résultats détaillés
-  Calcule les moyennes et écarts-types globaux
-  Sauvegarde les résultats dans un répertoire dédié

**Fonctions clés** :
- `separate_track()` : Sépare une piste complète en vocals et accompaniment
- `evaluate_model()` : Évalue le modèle sur le test set complet
