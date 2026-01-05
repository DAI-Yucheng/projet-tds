## 📝 Pour le rapport de TP

### Informations à inclure

1. **Résultats globaux** :
   - Moyennes et écarts-types de SDR, SIR, SAR
   - Nombre de tracks évalués

2. **Analyse** :
   - Comparaison avec l'article de référence
   - Discussion des points forts et faibles
   - Analyse de quelques tracks spécifiques

3. **Visualisations** (optionnel) :
   - Graphiques en boîte (boxplots) des métriques
   - Comparaison entre différents types de musique

### Exemple de code pour analyser les résultats

```python
import pandas as pd
import matplotlib.pyplot as plt

# Charger les résultats
df = pd.read_csv('./eval/evaluation_results.csv')

# Statistiques descriptives
print(df.describe())

# Graphiques
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
df.boxplot(column=['SDR', 'SIR', 'SAR'], ax=axes)
plt.tight_layout()
plt.savefig('evaluation_metrics.png')
```

## ✅ Checklist de validation

Avant de soumettre votre rapport, vérifiez que :

- [ ] Le script `evaluate.py` fonctionne sans erreur
- [ ] Les résultats sont sauvegardés dans `./eval/`
- [ ] Le fichier CSV contient les scores pour chaque track
- [ ] Le résumé contient les moyennes globales
- [ ] Les métriques sont cohérentes (SDR, SIR, SAR > 0 dB généralement)
- [ ] Les résultats sont analysés dans le rapport

## 🔗 Références

- **Article de référence** : https://openaccess.city.ac.uk/id/eprint/19289/
- **museval** : https://github.com/sigsep/sigsep-mus-eval
- **MUSDB** : https://github.com/sigsep/sigsep-mus-db

## 💡 Conseils

1. **Commencez petit** : Testez avec `--n-tracks 5` avant de lancer l'évaluation complète
2. **Vérifiez les chemins** : Assurez-vous que le checkpoint et le dataset sont accessibles
3. **Analysez les résultats** : Regardez quels tracks donnent les meilleurs/pires résultats
4. **Comparez avec l'article** : Utilisez les résultats de l'article comme référence


