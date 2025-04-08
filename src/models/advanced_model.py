"""
Module définissant un modèle CNN avancé avec des blocs résiduels pour la classification d'écorces d'arbres.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import NUM_CLASSES


class ResidualBlock(nn.Module):
    """
    Bloc résiduel (ResNet style) pour le modèle avancé.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Initialisation du bloc résiduel.
        
        Args:
            in_channels (int): Nombre de canaux d'entrée
            out_channels (int): Nombre de canaux de sortie
            stride (int): Pas de la convolution
        """
        super(ResidualBlock, self).__init__()
        
        # Couches de convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Connexion résiduelle (shortcut)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        """
        Passe avant du bloc résiduel.
        
        Args:
            x (torch.Tensor): Tenseur d'entrée
            
        Returns:
            torch.Tensor: Tenseur de sortie
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class AdvancedCNN(nn.Module):
    """
    Architecture CNN avancée avec des blocs résiduels pour la classification d'écorces d'arbres.
    
    Cette architecture est inspirée de ResNet avec:
    - Une couche de convolution initiale
    - Des blocs résiduels organisés en couches
    - Une couche de pooling global
    - Une couche fully connected pour la classification
    """
    
    def __init__(self, num_classes=NUM_CLASSES, num_blocks=[2, 2, 2, 2]):
        """
        Initialisation du modèle avancé.
        
        Args:
            num_classes (int): Nombre de classes à classifier
            num_blocks (list): Nombre de blocs résiduels dans chaque couche
        """
        super(AdvancedCNN, self).__init__()
        
        self.in_channels = 64
        
        # Couche de convolution initiale
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Couches de blocs résiduels
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        
        # Pooling global et classification
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, out_channels, num_blocks, stride):
        """
        Crée une couche composée de blocs résiduels.
        
        Args:
            out_channels (int): Nombre de canaux de sortie
            num_blocks (int): Nombre de blocs dans la couche
            stride (int): Pas de la convolution pour le premier bloc
            
        Returns:
            nn.Sequential: Couche de blocs résiduels
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Passe avant du modèle.
        
        Args:
            x (torch.Tensor): Batch d'images d'entrée de forme [batch_size, 3, height, width]
            
        Returns:
            torch.Tensor: Prédictions de classes de forme [batch_size, num_classes]
        """
        # Couche initiale
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        
        # Couches de blocs résiduels
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Pooling global et classification
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
    
    def predict(self, x):
        """
        Prédit les classes pour un batch d'images.
        
        Args:
            x (torch.Tensor): Batch d'images d'entrée
            
        Returns:
            torch.Tensor: Classes prédites
        """
        self.eval()
        with torch.no_grad():
            outputs = self(x)
            _, predicted = torch.max(outputs, 1)
        return predicted 