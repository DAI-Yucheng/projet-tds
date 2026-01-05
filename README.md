# Projet de Séparation de Sources Vocales avec U-Net - Traitemenet Avancé de Son - ISI - Sorbonne Université

Etudiants :

`Minh Nhut NGUYEN` -  21107823

`DAI Yucheng`  -   

---
## Table des Matières

1. [Vue d'ensemble](#vue-densemble)
2. [Structure du Projet et Relations entre les Fichiers](#structure-du-projet-et-relations-entre-les-fichiers)
3. [Interprétation des codes](#interprétation-des-codes)
4. [Flux de Données Complet](#flux-de-données-complet)
5. [Installation](#installation)
6. [Utilisation](#utilisation)
7. [Structure des Fichiers](#structure-des-fichiers)
8. [Paramètres Techniques](#paramètres-techniques)
9. [Explication des Concepts Clés](#explication-des-concepts-clés)
10. [Problèmes Courants](#problèmes-courants)
11. [Résultats Attendus](#résultats-attendus)
12. [Conseils](#conseils)

---

## Vue d'ensemble

Ce projet implémente un modèle U-Net pour séparer la voix (vocals) d'une chanson mixée. Le modèle apprend à prédire un "masque" (mask) qui indique quelle partie du spectrogramme (représentation fréquentielle de l'audio) correspond à la voix.

**En termes simples** : On donne au modèle une chanson complète (avec voix + instruments), et il nous dit "à chaque moment et à chaque fréquence, quelle proportion est de la voix". On peut ensuite extraire uniquement la voix.

---

## Structure du Projet et Relations entre les Fichiers

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
    ↓
evaluate.py  →  Évaluation (évaluer votre modèle U-Net sur le test set de MUSDB et calculer les métriques   
                            standard (SDR, SIR, SAR))
```
## Interprétation des codes 

[codes_interpretation.md](guide_book/codes_interpretation.md)

## Flux de Données Complet

[data_flow.md](guide_book/data_flow.md)

## Installation

### 1. Dépendances système

**Important** : `musdb` nécessite `ffmpeg` pour traiter les fichiers audio.
#### Si vous utilisez l'écosystème Linux (Ubuntu/WSL) 

```bash
# Ubuntu/WSL
sudo apt-get update
sudo apt-get install -y ffmpeg

# Vérifier l'installation
ffmpeg -version
```

#### Créer l'environement conda si vous utilisez pas l'écosystème Linux (Ubuntu/WSL) 
```bash
conda create -n SON python=3.12 -y
conda activate SON 

# Conda env
conda install conda-forge::musdb -y 
conda install conda-forge::ffmpeg -y

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
3. Créer un sous répertoire `MUSDB18` dans `projet-tds` et placer le dossier `musdb18` dedans
4. Structure attendue : `MUSDB18/musdb18/train/` et `MUSDB18/musdb18/test/`

Le code détectera automatiquement ce chemin. Si le dataset complet n'est pas trouvé, il demandera confirmation avant de télécharger la version demo.

---

## Utilisation

### Étape 1 : Tester le générateur de données

```bash
python data_generator.py
```

Cela vérifie que :
- Le dataset MUSDB est accessible
- Les spectrogrammes sont générés correctement
- Les patches ont la bonne taille (512 × 128)

### Étape 2 : Entraîner le modèle
```bash
# Entraînement de base (20 epochs, 10 chansons)
python train.py

# Personnaliser les paramètres
python train.py --epochs 20 --batch-size 16 --lr 5e-4 --n-songs 10
```

**Paramètres recommandés** :
- `--epochs` : 10-20 (suffisant pour voir la convergence)
- `--batch-size` : 16 (ajustable selon votre mémoire)
- `--lr` : 5e-4 (taux d'apprentissage)
- `--n-songs` : 5-10 (pour un entraînement rapide)

### Étape 3 : Séparer les vocals du test set MUSDB

Le script `inference.py` sépare automatiquement les vocals des chansons du test set MUSDB et les sauvegarde dans `vocal_separation/`.

```bash
# Séparer la première chanson (par défaut)
python inference.py --n-songs 1

# Séparer les 10 premières chansons
python inference.py --n-songs 10

# Séparer TOUTES les chansons du test set (50 tracks)
python inference.py --n-songs 9999

# Spécifier un checkpoint particulier
python inference.py --n-songs 10 --checkpoint vocal_checkpoints/best_model.pth

# Séparer un fichier audio spécifique (mode fichier unique)
python inference.py --audio mon_mix.wav --checkpoint vocal_checkpoints/best_model.pth
```

**Paramètres** :
- `--n-songs` (défaut: None = 1ère chanson uniquement)
  - `N` : Sépare les N premières chansons du test set
  - `9999` : Sépare TOUTES les chansons du test set
- `--checkpoint` : Chemin du modèle (si None, recherche automatique dans `vocal_checkpoints/`)
- `--musdb-path` : Chemin MUSDB (défaut: `MUSDB18/musdb18/test`)
- `--output-dir` : Répertoire de sortie (défaut: `vocal_separation/`)

Les fichiers séparés seront sauvegardés dans `vocal_separation/` avec le format `{nom_track}_vocals.wav`.

### Étape 4 : Évaluer les vocals séparés avec museval

Le script `evaluate.py` évalue les vocals déjà séparés (étape 3) en les comparant aux vraies vocals de MUSDB pour calculer les métriques SDR/SIR/SAR.

```bash
# Évaluer les 8 premières chansons séparées (par défaut)
python evaluate.py

# Évaluer 10 chansons
python evaluate.py --n-tracks 10

# Évaluer TOUTES les chansons dans vocal_separation/
python evaluate.py --n-tracks 9999

# Changer le répertoire de sortie
python evaluate.py --n-tracks 9999 --output-dir ./my_eval_results
```

**Paramètres** :
- `--n-tracks` (défaut: None = 8 premières chansons)
  - `N` : Évalue les N premières chansons de `vocal_separation/`
  - `9999` : Évalue TOUTES les chansons disponibles
- `--separation-dir` : Répertoire des vocals séparés (défaut: `vocal_separation/`)
- `--musdb-path` : Chemin MUSDB pour les références (défaut: `MUSDB18/musdb18`)
- `--output-dir` : Répertoire de sortie (défaut: `./eval/`)

Les résultats seront sauvegardés dans :
- `eval/evaluation_results.csv` : Métriques détaillées par track
- `eval/summary.txt` : Résumé global (moyennes et écarts-types)

Résultats d'évaluation --> [eval_results.md](guide_book/eval_results.md)
## Structure des Fichiers

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
│   ├── best_model.pth     # Meilleur modèle
│   └── final_model.pth    # Modèle final (mais pas forcement le meilleur)
│
└── eval/                  # Évaluations et résultats
    ├── test/
    ├── evaluation_results.csv
    └── summary.txt
```

## Paramètres Techniques

[technical_parameters.md](guide_book/technical_parameters.md)

## Explication des Concepts Clés

[key_concepts.md](guide_book/key_concepts.md)

## Problèmes Courants

[encountered_problems.md](guide_book/encountered_problems.md)

## Résultats Attendus

Après l'entraînement, vous devriez voir :
- **Loss initiale** : ~0.3-0.4
- **Loss finale** : ~0.1-0.15 (si le modèle apprend bien)
- **Mask std** : > 0.15 (indique que le mask varie, pas constant)

Le modèle sauvegardé dans `checkpoints/best_model.pth` peut être utilisé pour séparer la voix de nouvelles chansons.

## Conseils

- Commencez avec peu de cycle d'entrainement (10-20) pour tester rapidement
- Surveillez le loss : il devrait diminuer, pas augmenter
- Si le loss stagne, essayez de réduire le taux d'apprentissage
