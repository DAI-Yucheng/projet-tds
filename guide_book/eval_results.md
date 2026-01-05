## Fichiers générés

Après l'évaluation, vous trouverez dans le répertoire de sortie (`./eval/` par défaut) :

1. **`evaluation_results.csv`** : Tableau détaillé avec les scores pour chaque track
   - Colonnes : `track`, `SDR`, `SIR`, `SAR`
   - Format CSV pour analyse ultérieure
   ```csv
   track,SDR,SIR,SAR
   track1,5.23,8.45,4.12
   track2,6.12,9.23,5.34
   ...
   ```

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