"""
Implémentation du modèle U-Net - pour la séparation de sources

Selon les exigences du papier :
- Encoder : Conv2D + stride=2 + Batch Normalization + LeakyReLU
- Decoder : ConvTranspose2D + + stride=2 + Batch Normalization + ReLU + Skip connections
- Dropout (50%) sur les 3 premières couches du decoder
- Dernière couche : sigmoid (mask ∈ [0,1])
- Loss : L1 loss, L = || mask ⊙ X - Y ||₁
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    """
    Modèle U-Net simplifié pour la séparation de sources
    
    Entrée : spectrogramme de magnitude du mix (batch, freq_bins, time_frames)
    Sortie : mask (batch, freq_bins, time_frames) ∈ [0, 1]
    """
    
    def __init__(
        self,
        n_freq_bins: int = 512,  # Modifié : 512 au lieu de 513 (puissance de 2, comme notebook)
        n_time_frames: int = 128,
        n_channels: int = 16,  # Nombre initial de canaux
        n_layers: int = 6  # Nombre de couches Encoder/Decoder 
    ):
        """
        Initialiser U-Net
        
        Args:
            n_freq_bins: Nombre de bins de fréquence (notebook : 512, au lieu de 513)
            n_time_frames: Nombre de frames temporelles (papier : 128)
            n_channels: Nombre initial de canaux 
            n_layers: Nombre de couches Encoder/Decoder 
        """
        super(UNet, self).__init__()
        
        self.n_freq_bins = n_freq_bins
        self.n_time_frames = n_time_frames
        self.n_channels = n_channels
        self.n_layers = n_layers
        
        # Encoder (chemin de sous-échantillonnage)
        self.encoder = nn.ModuleList()
        in_channels = 1  # L'entrée est un spectrogramme monocanal
        
        for i in range(n_layers):
            out_channels = n_channels * (2 ** i)
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=(5, 5),
                        stride=(2, 2),
                        padding=(2, 2)
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            in_channels = out_channels
        
        # Decoder (chemin de sur-échantillonnage) + Skip connections
        self.decoder = nn.ModuleList()
        
        # Pré-calculer la configuration du nombre de canaux pour chaque couche
        # decoder va de la couche la plus basse à la plus haute (i de n_layers-1 à 0)
        decoder_configs = []
        prev_out = None
        
        for i in range(n_layers - 1, -1, -1):
            if i == n_layers - 1:
                # Couche la plus basse, pas de skip connection
                in_ch = n_channels * (2 ** i)  # Sortie de la dernière couche de l'encoder
                out_ch = n_channels * (2 ** (i - 1)) if i > 0 else 1
                prev_out = out_ch
            else:
                # Avec skip connection
                skip_ch = n_channels * (2 ** i)  # Sortie de la couche i correspondante de l'encoder
                in_ch = prev_out + skip_ch  # Sortie de la couche précédente + skip
                out_ch = n_channels * (2 ** (i - 1)) if i > 0 else 1
                prev_out = out_ch
            
            decoder_configs.append((in_ch, out_ch, i))
        
        # Créer les couches du decoder selon la configuration
        for in_ch, out_ch, layer_idx in decoder_configs:
            is_output_layer = (layer_idx == 0)  # La dernière couche (layer_idx=0) est la couche de sortie
            
            conv_transpose = nn.ConvTranspose2d(
                in_ch,
                out_ch,
                kernel_size=(5, 5),
                stride=(2, 2),
                padding=(2, 2),
                output_padding=(1, 1)  # Assurer la correspondance des dimensions
            )
            
            if is_output_layer:
                # Couche de sortie : utiliser Sigmoid, nécessite une initialisation spéciale
                activation = nn.Sigmoid()
                # Initialisation spéciale : faire en sorte que la sortie initiale soit proche de la moyenne de oracle_mask (environ 0.4)
                # sigmoid(-0.4) ≈ 0.4, donc définir le bias à -0.4
                nn.init.xavier_uniform_(conv_transpose.weight, gain=0.1)
                if conv_transpose.bias is not None:
                    nn.init.constant_(conv_transpose.bias, -0.4)
                # Pas de BatchNorm pour la couche de sortie
                self.decoder.append(nn.Sequential(conv_transpose, activation))
            else:
                # Couches intermédiaires : utiliser ReLU + BatchNorm, initialisation He
                batch_norm = nn.BatchNorm2d(out_ch)
                activation = nn.ReLU(inplace=True)
                nn.init.kaiming_normal_(conv_transpose.weight, mode='fan_out', nonlinearity='relu')
                if conv_transpose.bias is not None:
                    nn.init.constant_(conv_transpose.bias, 0)
                
                # Ajouter Dropout (50%) sur les 3 premières couches du decoder
                # Les 3 premières couches correspondent aux layer_idx les plus élevés
                if layer_idx >= n_layers - 3:
                    dropout = nn.Dropout(0.5)
                    self.decoder.append(nn.Sequential(conv_transpose, batch_norm, activation, dropout))
                else:
                    self.decoder.append(nn.Sequential(conv_transpose, batch_norm, activation))
        
        # Initialiser les autres couches (encoder et couches intermédiaires du decoder)
        self._initialize_weights()
    
    def forward(self, x):
        """
        Propagation avant
        
        Args:
            x: Spectrogramme d'entrée, shape (batch, freq_bins, time_frames)
            
        Returns:
            mask: shape (batch, freq_bins, time_frames), plage de valeurs [0, 1]
        """
        # Ajouter la dimension channel : (batch, freq, time) -> (batch, 1, freq, time)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        # Chemin Encoder + sauvegarder les skip connections
        skip_connections = []
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            skip_connections.append(x)
        
        # Chemin Decoder + skip connections
        # L'ordre du decoder va de la couche la plus basse à la plus haute (i de n_layers-1 à 0)
        # L'ordre de skip_connections va de la couche la plus haute à la plus basse (0 à n_layers-1)
        # decoder[0] correspond à la couche la plus basse (i=n_layers-1), pas de skip
        # decoder[1] correspond à i=n_layers-2, connecte skip_connections[n_layers-2]
        # decoder[2] correspond à i=n_layers-3, connecte skip_connections[n_layers-3]
        # ...
        for i, decoder_layer in enumerate(self.decoder):
            # Avant de connecter le skip connection, connecter d'abord (s'il y en a)
            if i > 0:
                # Connecter le skip connection (de la couche correspondante de l'encoder)
                # La couche i du decoder (index enumerate) correspond à la couche (n_layers-1-i) de l'encoder
                # Parce que le decoder va de la couche la plus basse à la plus haute, tandis que skip_connections va de la plus haute à la plus basse
                skip_idx = len(skip_connections) - 1 - i
                skip = skip_connections[skip_idx] # features de l'encoder 
                
                # Assurer que les dimensions spatiales correspondent (peut nécessiter un recadrage)
                if x.shape[2] != skip.shape[2] or x.shape[3] != skip.shape[3]:
                    # Recadrer à la plus petite dimension
                    min_h = min(x.shape[2], skip.shape[2])
                    min_w = min(x.shape[3], skip.shape[3])
                    x = x[:, :, :min_h, :min_w]
                    skip = skip[:, :, :min_h, :min_w]
                
                # Concaténer le skip connection (dans la dimension des canaux)
                # Cela augmentera le nombre de canaux de x de prev_out à prev_out+skip_channels
                x = torch.cat([x, skip], dim=1)
            
            # Maintenant le nombre de canaux de x devrait correspondre à l'entrée attendue de decoder_layer
            # Vérifier si le nombre de canaux correspond (pour le débogage)
            expected_channels = decoder_layer[0].in_channels
            actual_channels = x.shape[1]
            if actual_channels != expected_channels:
                raise RuntimeError(
                    f"Couche Decoder {i} : nombre de canaux ne correspond pas ! "
                    f"Réel : {actual_channels}, Attendu : {expected_channels}. "
                    f"Shape d'entrée : {x.shape}"
                )
            
            # Appeler la couche decoder
            x = decoder_layer(x) # c'est ici que le upsampling se fait quand i=0 
        
        # Supprimer la dimension channel : (batch, 1, freq, time) -> (batch, freq, time)
        x = x.squeeze(1)
        
        # Recadrer à la taille d'entrée originale (car le sur-échantillonnage peut causer une non-correspondance de taille)
        # Assurer que la taille de sortie correspond exactement à la taille d'entrée
        if x.shape[1] != self.n_freq_bins or x.shape[2] != self.n_time_frames:
            x = x[:, :self.n_freq_bins, :self.n_time_frames]
        
        return x
    
    def _initialize_weights(self):
        """
        Initialiser les poids du modèle (encoder et couches intermédiaires du decoder)
        
        Note : La couche de sortie (dernière couche) a déjà été initialisée lors de la création
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Couches Encoder : initialisation He (adaptée à LeakyReLU)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                # Couches intermédiaires du Decoder : déjà initialisées lors de la création, ignorer ici
                # Ne traiter que les couches qui pourraient avoir été omises
                pass
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def test_unet():
    """Tester le modèle U-Net"""
    print("=" * 50)
    print("Test du modèle U-Net")
    print("=" * 50)
    
    # Créer le modèle
    model = UNet(
        n_freq_bins=512,  # Modifié : 512 au lieu de 513
        n_time_frames=128,
        n_channels=32,
        n_layers=4
    )
    
    # Calculer le nombre de paramètres
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nParamètres du modèle :")
    print(f"  Nombre total de paramètres : {total_params:,}")
    print(f"  Paramètres entraînables : {trainable_params:,}")
    
    # Tester la propagation avant
    batch_size = 4
    x = torch.randn(batch_size, 512, 128)  # Modifié : 512 au lieu de 513
    
    print(f"\nShape d'entrée : {x.shape}")
    
    with torch.no_grad():
        mask = model(x)
    
    print(f"Shape du mask de sortie : {mask.shape}")
    print(f"Plage de valeurs du mask : min={mask.min():.4f}, max={mask.max():.4f}, mean={mask.mean():.4f}")
    
    # Vérifier que le mask est dans la plage [0, 1]
    assert mask.min() >= 0 and mask.max() <= 1, "Le mask devrait être dans la plage [0, 1] !"
    print("\n✓ Vérification de la plage de valeurs du mask réussie !")
    
    # Tester l'application du mask
    estimated_vocals = mask * x
    print(f"\nShape après application du mask : {estimated_vocals.shape}")
    print(f"  Mix original : min={x.min():.4f}, max={x.max():.4f}")
    print(f"  Vocals estimés : min={estimated_vocals.min():.4f}, max={estimated_vocals.max():.4f}")
    
    print("\n✓ Test du modèle U-Net réussi !")
    print("\nDescription :")
    print("1. Encoder : Conv2D + stride=2 + LeakyReLU")
    print("2. Decoder : ConvTranspose2D + skip connections")
    print("3. Dernière couche : Sigmoid (mask ∈ [0,1])")
    print("4. Sortie : mask ⊙ mix = estimated_vocals")


if __name__ == "__main__":
    test_unet()
