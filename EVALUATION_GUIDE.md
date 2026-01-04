# Guide d'Évaluation - Étape 5 du TP

Ce guide explique comment utiliser le script `evaluate.py` pour évaluer votre modèle U-Net sur le test set de MUSDB et calculer les métriques standard (SDR, SIR, SAR).

## 📋 Prérequis

### 1. Installation des dépendances

Assurez-vous d'avoir installé toutes les dépendances :

```bash
pip install -r requirements.txt
```

Les nouvelles dépendances nécessaires pour l'évaluation sont :
- `museval>=0.4.0` : Bibliothèque d'évaluation des métriques de séparation de sources
- `pandas>=1.3.0` : Pour le traitement des résultats

### 2. Modèle entraîné

Vous devez avoir un modèle entraîné sauvegardé dans :
- `vocal_checkpoints/best_model.pth` (recommandé)
- ou `checkpoints/best_model.pth`

### 3. Dataset MUSDB

Le dataset MUSDB doit être disponible dans :
- `MUSDB18/musdb18/` (chemin relatif)
- ou spécifiez le chemin avec `--musdb-path`

## 🚀 Utilisation

### Utilisation de base

```bash
python evaluate.py
```

Le script va automatiquement :
1. Chercher le checkpoint dans `vocal_checkpoints/` ou `checkpoints/`
2. Charger le dataset MUSDB depuis `MUSDB18/musdb18/`
3. Évaluer tous les tracks du test set
4. Sauvegarder les résultats dans `./eval/`

### Options avancées

```bash
# Spécifier un checkpoint particulier
python evaluate.py --checkpoint vocal_checkpoints/best_model.pth

# Évaluer seulement les 5 premiers tracks (pour tester rapidement)
python evaluate.py --n-tracks 5

# Spécifier un chemin différent pour MUSDB
python evaluate.py --musdb-path /chemin/vers/musdb18

# Forcer l'utilisation du CPU
python evaluate.py --cpu

# Changer le répertoire de sortie
python evaluate.py --output-dir ./my_eval_results

# Spécifier le nombre de canaux du modèle (si différent de 16)
python evaluate.py --n-channels 16
```

### Exemple complet

```bash
python evaluate.py \
    --checkpoint vocal_checkpoints/best_model.pth \
    --musdb-path MUSDB18/musdb18 \
    --n-channels 16 \
    --n-tracks 10 \
    --output-dir ./eval_results
```

## 📊 Résultats

### Fichiers générés

Après l'évaluation, vous trouverez dans le répertoire de sortie (`./eval/` par défaut) :

1. **`evaluation_results.csv`** : Tableau détaillé avec les scores pour chaque track
   - Colonnes : `track`, `SDR`, `SIR`, `SAR`
   - Format CSV pour analyse ultérieure

2. **`summary.txt`** : Résumé textuel avec les moyennes globales
   - Contient les moyennes et écarts-types pour SDR, SIR, SAR

3. **Fichiers JSON par track** : Générés automatiquement par `museval`
   - Un fichier par track avec les scores détaillés
   - Format standard BSSEval v4

### Métriques expliquées

- **SDR (Signal-to-Distortion Ratio)** : Mesure la qualité globale de la séparation
  - Plus élevé = meilleur
  - Typiquement entre -5 dB et 15 dB pour la séparation vocale

- **SIR (Signal-to-Interference Ratio)** : Mesure la capacité à séparer la source cible des autres sources
  - Plus élevé = moins d'interférence des autres instruments
  - Typiquement entre 0 dB et 20 dB

- **SAR (Signal-to-Artifacts Ratio)** : Mesure la qualité du signal reconstruit (artefacts introduits)
  - Plus élevé = moins d'artefacts
  - Typiquement entre 0 dB et 15 dB

### Exemple de sortie

```
======================================================================
RÉSULTATS GLOBAUX
======================================================================

Nombre de tracks évalués : 50

Métriques moyennes (vocals) :
  SDR : 5.23 ± 2.15 dB
  SIR : 8.45 ± 3.21 dB
  SAR : 4.12 ± 1.89 dB
```

## 🔍 Interprétation des résultats

### Résultats typiques

Pour un modèle U-Net bien entraîné sur MUSDB :
- **SDR** : 4-7 dB (bon), 7-10 dB (très bon)
- **SIR** : 6-10 dB (bon), 10-15 dB (très bon)
- **SAR** : 3-6 dB (bon), 6-9 dB (très bon)

### Comparaison avec l'article

L'article de référence (https://openaccess.city.ac.uk/id/eprint/19289/) peut servir de point de comparaison. Les résultats dépendent de :
- La taille du modèle
- Le nombre d'epochs d'entraînement
- La quantité de données utilisées
- Les hyperparamètres (learning rate, batch size, etc.)

## ⚠️ Problèmes courants

### Erreur : "No module named 'museval'"

**Solution** :
```bash
pip install museval
```

### Erreur : "Checkpoint not found"

**Solution** : Spécifiez explicitement le chemin :
```bash
python evaluate.py --checkpoint vocal_checkpoints/best_model.pth
```

### Erreur : "MUSDB dataset not found"

**Solution** : Vérifiez que le dataset est bien téléchargé et spécifiez le chemin :
```bash
python evaluate.py --musdb-path /chemin/vers/musdb18
```

### Erreur : "CUDA out of memory"

**Solution** : Utilisez le CPU ou réduisez le nombre de tracks :
```bash
python evaluate.py --cpu --n-tracks 5
```

## 📝 Pour le rapport

Pour votre rapport de TP, vous pouvez :

1. **Inclure les moyennes globales** : SDR, SIR, SAR avec écarts-types
2. **Analyser quelques tracks spécifiques** : Montrer les variations entre différents types de musique
3. **Comparer avec l'article** : Discuter des différences et similitudes
4. **Visualiser les résultats** : Créer des graphiques à partir du CSV généré

### Exemple de code pour analyser les résultats

```python
import pandas as pd
import matplotlib.pyplot as plt

# Charger les résultats
df = pd.read_csv('./eval/evaluation_results.csv')

# Afficher les statistiques
print(df.describe())

# Créer un graphique
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
df.boxplot(column=['SDR', 'SIR', 'SAR'], ax=axes)
plt.tight_layout()
plt.savefig('evaluation_metrics.png')
```

## 🎯 Prochaines étapes

Après l'évaluation :

1. **Analyser les résultats** :** Identifier les points forts et faibles du modèle
2. **Améliorer le modèle** : Ajuster les hyperparamètres si nécessaire
3. **Tester sur d'autres données** : Évaluer la généralisation (bonus du TP)
4. **Préparer le rapport** : Documenter les résultats et les analyses

---

**Bon évaluation ! 🎵**

