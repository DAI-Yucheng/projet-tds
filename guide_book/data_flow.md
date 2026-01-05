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
