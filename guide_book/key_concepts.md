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