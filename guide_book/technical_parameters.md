### Paramètres du spectrogramme (selon le papier)

- **Taux d'échantillonnage** : 8192 Hz (réduit de 44100 Hz pour accélérer)
- **Taille de fenêtre STFT** : 1024
- **Hop length** : 768
- **Taille de patch** : 128 frames
- **Overlap** : 50% (un patch tous les 64 frames)

### Architecture du modèle

- **Fréquences** : 512 bins
- **Frames temporelles** : 128
- **Canaux initiaux** : 16
- **Nombre de couches** : 6 (encoder + decoder)

### Fonction de perte

**$L_{1,1}$ Mask Loss** : `L = || mask * X - Y ||₁,₁`

- `mask` : Prédiction du modèle (entre 0 et 1)
- `X` : Magnitude de la spectrogramme originale (mix)
- `Y` : Magnitude de la spectrogramme target (vocal/instrumental)
- `|| ||₁` : Norme L1 (somme des valeurs absolues des différences)

**Pourquoi cette méthode ?**
- Supervision directe : le modèle apprend directement à prédire le bon masque
- Plus stable que de prédire le spectrogramme vocal directement
- Évite les problèmes de normalisation