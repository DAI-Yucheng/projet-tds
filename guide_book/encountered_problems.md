### Erreur : "ffmpeg or ffprobe could not be found"

**Solution** : Installer ffmpeg
```bash
sudo apt-get install -y ffmpeg
```
---

### Erreur : "ModuleNotFoundError: No module named 'torch'"

**Solution** : Installer PyTorch
```bash
pip install torch torchvision torchaudio
```
---
### Le loss ne diminue pas

**Causes possibles** :
- Taux d'apprentissage trop élevé → réduire `--lr` à 1e-4
- Données mal normalisées → vérifier `data_generator.py`
- Modèle trop petit → augmenter `n_channels` dans `unet_model.py`
---
### Mémoire insuffisante

**Solutions** :
- Réduire `--batch-size` (ex: 8 au lieu de 16)
- Réduire `n_channels` dans le modèle (ex: 32 au lieu de 64)
---
### Erreur : "No module named 'museval'"

**Solution** :
```bash
pip install museval
```
---
### Erreur : "Checkpoint not found"

**Solution** : Spécifiez explicitement le chemin :
```bash
python evaluate.py --checkpoint vocal_checkpoints/best_model.pth
```
---
### Erreur : "MUSDB dataset not found"

**Solution** : Vérifiez que le dataset est bien téléchargé et spécifiez le chemin :
```bash
python evaluate.py --musdb-path /chemin/vers/musdb18
```
---
### Erreur : "CUDA out of memory"

**Solution** : Utilisez le CPU ou réduisez le nombre de tracks :
```bash
python evaluate.py --cpu --n-tracks 5
```
---