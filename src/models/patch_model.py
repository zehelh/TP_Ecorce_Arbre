"""
Module définissant un modèle CNN basé sur l'approche par patches pour la classification d'écorces d'arbres.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import NUM_CLASSES, NUM_PATCHES


class PatchFeatureExtractor(nn.Module):
    """
    Extracteur de caractéristiques pour les patches d'images d'écorces.
    """
    def __init__(self):
        """
        Initialisation de l'extracteur de caractéristiques.
        """
        super(PatchFeatureExtractor, self).__init__()
        
        # Bloc 1 de convolution
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bloc 2 de convolution
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bloc 3 de convolution
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Pooling global pour obtenir un vecteur de caractéristiques par patch
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x):
        """
        Passe avant de l'extracteur de caractéristiques.
        
        Args:
            x (torch.Tensor): Batch de patches d'entrée de forme [batch_size, num_patches, 3, height, width]
            
        Returns:
            torch.Tensor: Caractéristiques extraites de forme [batch_size, num_patches, 128]
        """
        batch_size, num_patches, channels, height, width = x.size()
        
        # Reshape pour traiter tous les patches en une seule dimension de batch
        x = x.view(batch_size * num_patches, channels, height, width)
        
        # Extraction des caractéristiques
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Pooling global
        x = self.global_pool(x)
        x = x.view(batch_size, num_patches, -1)
        
        return x


class PatchCNN(nn.Module):
    """
    Modèle CNN basé sur l'approche par patches pour la classification d'écorces d'arbres.
    
    Cette architecture comprend:
    - Un extracteur de caractéristiques appliqué à chaque patch
    - Un agrégateur de caractéristiques qui combine les caractéristiques des patches
    - Un classifieur final
    """
    
    def __init__(self, num_classes=NUM_CLASSES, num_patches=NUM_PATCHES):
        """
        Initialisation du modèle basé sur des patches.
        
        Args:
            num_classes (int): Nombre de classes à classifier
            num_patches (int): Nombre de patches par image
        """
        super(PatchCNN, self).__init__()
        
        self.num_patches = num_patches
        
        # Extracteur de caractéristiques pour les patches
        self.feature_extractor = PatchFeatureExtractor()
        
        # Agrégateur de caractéristiques
        self.feature_dim = 128
        
        # Options d'agrégation: attention, moyenne pondérée, ou simple moyenne
        self.aggregation_type = "attention"  # "weighted_avg", "avg"
        
        if self.aggregation_type == "attention":
            # Mécanisme d'attention pour pondérer les patches
            self.attention = nn.Sequential(
                nn.Linear(self.feature_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )
        elif self.aggregation_type == "weighted_avg":
            # Poids apprenables pour chaque patch
            self.patch_weights = nn.Parameter(torch.ones(1, num_patches, 1) / num_patches)
        
        # Classifieur final
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """
        Passe avant du modèle.
        
        Args:
            x (torch.Tensor): Batch de patches d'entrée de forme [batch_size, num_patches, 3, height, width]
            
        Returns:
            torch.Tensor: Prédictions de classes de forme [batch_size, num_classes]
        """
        # Extraction des caractéristiques pour chaque patch
        patch_features = self.feature_extractor(x)  # [batch_size, num_patches, feature_dim]
        
        # Agrégation des caractéristiques des patches
        if self.aggregation_type == "attention":
            # Calcul des scores d'attention pour chaque patch
            attention_scores = self.attention(patch_features)  # [batch_size, num_patches, 1]
            attention_weights = F.softmax(attention_scores, dim=1)
            
            # Pondération des caractéristiques par les poids d'attention
            aggregated_features = torch.sum(patch_features * attention_weights, dim=1)  # [batch_size, feature_dim]
            
        elif self.aggregation_type == "weighted_avg":
            # Pondération des caractéristiques par des poids apprenables
            aggregated_features = torch.sum(patch_features * self.patch_weights, dim=1)  # [batch_size, feature_dim]
            
        else:  # "avg"
            # Moyenne simple des caractéristiques
            aggregated_features = torch.mean(patch_features, dim=1)  # [batch_size, feature_dim]
        
        # Classification des caractéristiques agrégées
        logits = self.classifier(aggregated_features)
        
        return logits
    
    def predict(self, x):
        """
        Prédit les classes pour un batch d'images.
        
        Args:
            x (torch.Tensor): Batch de patches d'entrée
            
        Returns:
            torch.Tensor: Classes prédites
        """
        self.eval()
        with torch.no_grad():
            outputs = self(x)
            _, predicted = torch.max(outputs, 1)
        return predicted 