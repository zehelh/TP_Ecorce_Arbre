"""
Module définissant un modèle CNN avancé basé sur l'approche par patches pour la classification d'écorces d'arbres.

Ce modèle intègre:
1. Un extracteur de caractéristiques basé sur ResNet pre-entraîné
2. Un mécanisme d'attention multi-têtes pour l'agrégation des patches
3. Une sélection intelligente de patches basée sur la richesse d'information
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

from ..config import NUM_CLASSES, NUM_PATCHES


class MultiHeadAttention(nn.Module):
    """
    Mécanisme d'attention multi-têtes simplifié pour l'agrégation des patches.
    """
    def __init__(self, d_model, num_heads=4):
        """
        Initialise le module d'attention multi-têtes.
        
        Args:
            d_model (int): Dimension du modèle (feature dimension)
            num_heads (int): Nombre de têtes d'attention
        """
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Projections linéaires pour les têtes d'attention
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        # Projection finale
        self.out_linear = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        """
        Applique l'attention multi-têtes.
        
        Args:
            x (torch.Tensor): Tensor de caractéristiques [batch_size, num_patches, d_model]
            
        Returns:
            torch.Tensor: Caractéristiques après attention [batch_size, d_model]
        """
        batch_size, num_patches, _ = x.size()
        
        # Projections linéaires et reshape pour les têtes multiples
        q = self.q_linear(x).view(batch_size, num_patches, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, num_patches, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, num_patches, self.num_heads, self.d_k).transpose(1, 2)
        
        # Calcul de l'attention (produit scalaire)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Application de l'attention et reshape
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, num_patches, self.d_model)
        
        # Projection finale
        out = self.out_linear(out)
        
        # Agrégation pondérée des patches
        global_weights = F.softmax(torch.sum(attn_weights, dim=1).mean(dim=1), dim=-1)
        global_weights = global_weights.unsqueeze(1).unsqueeze(1)
        out = torch.sum(out * global_weights, dim=1)
        
        return out


class ResNetFeatureExtractor(nn.Module):
    """
    Extracteur de caractéristiques basé sur ResNet pre-entraîné pour les patches d'images d'écorces.
    """
    def __init__(self, pretrained=True, model_name='resnet18'):
        """
        Initialisation de l'extracteur de caractéristiques ResNet.
        
        Args:
            pretrained (bool): Si True, utilise les poids pré-entraînés sur ImageNet
            model_name (str): Nom du modèle ResNet ('resnet18', 'resnet34', etc.)
        """
        super(ResNetFeatureExtractor, self).__init__()
        
        # Charger le modèle ResNet pré-entraîné
        if model_name == 'resnet18':
            base_model = models.resnet18(pretrained=pretrained)
            self.feature_dim = 512
        elif model_name == 'resnet34':
            base_model = models.resnet34(pretrained=pretrained)
            self.feature_dim = 512
        elif model_name == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Modèle ResNet non supporté: {model_name}")
        
        # Enlever les couches fully-connected et le average pooling
        modules = list(base_model.children())[:-2]
        self.backbone = nn.Sequential(*modules)
        
        # Pooling global adaptatif
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Adaptation de domaine pour les écorces d'arbres (couche de projection)
        self.domain_adapt = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, self.feature_dim)
        )
        
        # Gel des poids initiaux (sera dégelé progressivement pendant l'entraînement)
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_layers(self, num_layers=0):
        """
        Dégèle progressivement les couches du backbone ResNet.
        
        Args:
            num_layers (int): Nombre de couches à dégeler, en partant de la fin du réseau
                             (0 = aucune couche dégelée, -1 = toutes les couches dégelées)
        """
        if num_layers == -1:
            # Dégeler toutes les couches
            for param in self.backbone.parameters():
                param.requires_grad = True
            return
            
        # Ne rien faire si num_layers est 0
        if num_layers == 0:
            return
            
        # Récupérer les modules du backbone en liste
        modules = list(self.backbone.children())
        total_modules = len(modules)
        
        # Déterminer combien de modules à dégeler (à partir de la fin)
        modules_to_unfreeze = min(num_layers, total_modules)
        
        # Dégeler les derniers modules
        for i in range(total_modules - modules_to_unfreeze, total_modules):
            for param in modules[i].parameters():
                param.requires_grad = True
    
    def forward(self, x):
        """
        Passe avant de l'extracteur de caractéristiques.
        
        Args:
            x (torch.Tensor): Batch de patches d'entrée de forme [batch_size, num_patches, 3, height, width]
            
        Returns:
            torch.Tensor: Caractéristiques extraites de forme [batch_size, num_patches, feature_dim]
        """
        batch_size, num_patches, channels, height, width = x.size()
        
        # Reshape pour traiter tous les patches en une seule dimension de batch
        x = x.view(batch_size * num_patches, channels, height, width)
        
        # Extraction des caractéristiques
        x = self.backbone(x)
        
        # Pooling global
        x = self.global_pool(x)
        x = x.view(batch_size * num_patches, -1)
        
        # Adaptation de domaine
        x = self.domain_adapt(x)
        
        # Reshape pour récupérer la dimension des patches
        x = x.view(batch_size, num_patches, -1)
        
        return x


class CrossPatchContextModule(nn.Module):
    """
    Module pour modéliser les relations contextuelles entre patches.
    """
    def __init__(self, feature_dim):
        """
        Initialise le module de contexte inter-patches.
        
        Args:
            feature_dim (int): Dimension des caractéristiques
        """
        super(CrossPatchContextModule, self).__init__()
        
        # Transformation non-linéaire pour le calcul de similarité
        self.relation_transform = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1)
        )
        
        # Projection des caractéristiques contextuelles
        self.context_projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )
    
    def forward(self, patch_features):
        """
        Calcule les relations contextuelles entre patches.
        
        Args:
            patch_features (torch.Tensor): Caractéristiques des patches [batch_size, num_patches, feature_dim]
            
        Returns:
            torch.Tensor: Caractéristiques contextualisées [batch_size, num_patches, feature_dim]
        """
        batch_size, num_patches, feature_dim = patch_features.size()
        
        # Calculer la matrice d'adjacence entre patches
        relation_matrix = torch.zeros(batch_size, num_patches, num_patches, device=patch_features.device)
        
        # Pour chaque paire de patches, calculer leur relation
        for i in range(num_patches):
            for j in range(num_patches):
                if i != j:
                    # Concaténer les caractéristiques des deux patches
                    pair_features = torch.cat([
                        patch_features[:, i, :],
                        patch_features[:, j, :]
                    ], dim=1)
                    
                    # Calculer le score de relation
                    relation_score = self.relation_transform(pair_features)
                    relation_matrix[:, i, j] = relation_score.squeeze(-1)
        
        # Normaliser la matrice de relation
        relation_weights = F.softmax(relation_matrix, dim=2)
        
        # Calculer les caractéristiques contextuelles
        context_features = torch.bmm(relation_weights, patch_features)
        
        # Projection des caractéristiques contextuelles
        context_features = self.context_projection(context_features)
        
        # Combiner avec les caractéristiques originales (connexion résiduelle)
        enhanced_features = patch_features + context_features
        
        return enhanced_features


class EnhancedPatchCNN(nn.Module):
    """
    Modèle CNN avancé basé sur l'approche par patches pour la classification d'écorces d'arbres.
    
    Cette architecture comprend:
    - Un extracteur de caractéristiques ResNet pour chaque patch
    - Un module de contexte inter-patches pour modéliser les relations
    - Un agrégateur d'attention multi-têtes pour combiner les patches
    - Un classifieur final avec one-hot encoding
    """
    
    def __init__(self, num_classes=NUM_CLASSES, num_patches=NUM_PATCHES, 
                 resnet_model='resnet18', pretrained=True, use_context=True):
        """
        Initialisation du modèle avancé basé sur des patches.
        
        Args:
            num_classes (int): Nombre de classes à classifier
            num_patches (int): Nombre de patches par image
            resnet_model (str): Modèle ResNet à utiliser ('resnet18', 'resnet34', etc.)
            pretrained (bool): Si True, utilise les poids pré-entraînés sur ImageNet
            use_context (bool): Si True, utilise le module de contexte inter-patches
        """
        super(EnhancedPatchCNN, self).__init__()
        
        self.num_patches = num_patches
        self.use_context = use_context
        
        # Extracteur de caractéristiques ResNet pour les patches
        self.feature_extractor = ResNetFeatureExtractor(pretrained=pretrained, model_name=resnet_model)
        self.feature_dim = self.feature_extractor.feature_dim
        
        # Module optionnel de contexte inter-patches
        if use_context:
            self.context_module = CrossPatchContextModule(self.feature_dim)
        
        # Agrégateur d'attention multi-têtes
        self.attention = MultiHeadAttention(self.feature_dim, num_heads=4)
        
        # Classifieur final
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.BatchNorm1d(self.feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.feature_dim // 2, num_classes)
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
        
        # Modélisation du contexte inter-patches
        if self.use_context:
            patch_features = self.context_module(patch_features)
        
        # Agrégation des caractéristiques des patches par attention multi-têtes
        aggregated_features = self.attention(patch_features)  # [batch_size, feature_dim]
        
        # Classification des caractéristiques agrégées
        logits = self.classifier(aggregated_features)
        
        return logits
    
    def get_attention_weights(self, x):
        """
        Récupère les poids d'attention pour visualisation.
        
        Args:
            x (torch.Tensor): Batch de patches d'entrée
            
        Returns:
            torch.Tensor: Poids d'attention pour chaque patch
        """
        # Extraction des caractéristiques pour chaque patch
        patch_features = self.feature_extractor(x)
        
        # Modélisation du contexte inter-patches
        if self.use_context:
            patch_features = self.context_module(patch_features)
        
        # Obtenir les poids d'attention (calcul similaire à celui dans MultiHeadAttention)
        batch_size, num_patches, _ = patch_features.size()
        
        q = self.attention.q_linear(patch_features).view(batch_size, num_patches, self.attention.num_heads, self.attention.d_k).transpose(1, 2)
        k = self.attention.k_linear(patch_features).view(batch_size, num_patches, self.attention.num_heads, self.attention.d_k).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.attention.d_k ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Moyenner les poids sur les têtes d'attention
        attn_weights = attn_weights.mean(dim=1)
        
        return attn_weights
    
    def predict(self, x):
        """
        Prédit les classes pour un batch d'images.
        
        Args:
            x (torch.Tensor): Batch de patches d'entrée
            
        Returns:
            tuple: (classes prédites, scores one-hot)
        """
        self.eval()
        with torch.no_grad():
            outputs = self(x)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        return predicted, probabilities
    
    def predict_with_confidence(self, x):
        """
        Prédit les classes avec leurs scores de confiance.
        
        Args:
            x (torch.Tensor): Batch de patches d'entrée
            
        Returns:
            tuple: (classes prédites, confiances, scores one-hot)
        """
        self.eval()
        with torch.no_grad():
            outputs = self(x)
            probabilities = F.softmax(outputs, dim=1)
            confidences, predicted = torch.max(probabilities, 1)
        return predicted, confidences, probabilities
    
    def unfreeze_backbone(self, num_layers=0):
        """
        Dégèle progressivement les couches du backbone ResNet.
        
        Args:
            num_layers (int): Nombre de couches à dégeler
        """
        self.feature_extractor.unfreeze_layers(num_layers) 