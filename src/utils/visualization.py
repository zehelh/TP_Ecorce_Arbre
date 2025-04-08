"""
Module définissant des fonctions pour la visualisation et l'interprétation des décisions du modèle.

Il implémente notamment la méthode Grad-CAM pour visualiser les zones d'intérêt.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import logging

from ..config import FIGURES_DIR


class GradCAM:
    """
    Classe implémentant Grad-CAM pour visualiser les régions d'intérêt du modèle.
    
    Grad-CAM utilise les gradients qui circulent dans la couche de convolution
    finale pour produire une carte de chaleur qui met en évidence les zones importantes
    pour la prédiction.
    
    Référence : Selvaraju et al. (2017), "Grad-CAM: Visual Explanations from Deep Networks 
                via Gradient-based Localization"
    """
    
    def __init__(self, model, target_layer_name, verbose=False):
        """
        Initialise Grad-CAM avec un modèle et une couche cible.
        
        Args:
            model (nn.Module): Modèle CNN à analyser
            target_layer_name (str): Nom de la couche de convolution cible
            verbose (bool): Si True, affiche les messages de progression
        """
        self.model = model
        self.model.eval()
        self.verbose = verbose
        
        # Identifier la couche cible
        self.target_layer = None
        for name, module in self.model.named_modules():
            if name == target_layer_name:
                self.target_layer = module
                break
        
        if self.target_layer is None:
            raise ValueError(f"Couche cible {target_layer_name} non trouvée dans le modèle.")
        
        # Paramètres pour stocker les activations et les gradients
        self.gradients = None
        self.activations = None
        
        # Enregistrer les hooks pour capturer les activations et les gradients
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_backward_hook(self._backward_hook)
        
        if self.verbose:
            logging.info(f"GradCAM initialisé avec la couche cible : {target_layer_name}")
    
    def _forward_hook(self, module, input, output):
        """Hook pour capturer les activations de la couche cible."""
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        """Hook pour capturer les gradients de la couche cible."""
        self.gradients = grad_output[0].detach()
    
    def __call__(self, input_image, class_idx=None):
        """
        Génère une carte de chaleur Grad-CAM pour l'image d'entrée.
        
        Args:
            input_image (torch.Tensor): Tensor d'image d'entrée (normalisé)
            class_idx (int, optional): Indice de classe à analyser. 
                                      Si None, utilise la classe prédite.
                                      
        Returns:
            tuple: (heatmap, output, predicted_class)
                - heatmap (numpy.ndarray): Carte de chaleur Grad-CAM
                - output (torch.Tensor): Sortie du modèle
                - predicted_class (int): Classe prédite
        """
        # Forward pass
        output = self.model(input_image)
        
        # Si class_idx n'est pas fourni, utiliser la classe prédite
        if class_idx is None:
            _, class_idx = torch.max(output, 1)
            class_idx = class_idx.item()
            if self.verbose:
                logging.info(f"Classe prédite: {class_idx}")
        
        # Réinitialiser les gradients
        self.model.zero_grad()
        
        # Cibler la classe spécifiée
        target = output[0, class_idx]
        
        # Rétropropagation
        target.backward()
        
        # Récupérer les gradients et les activations
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Calculer les poids pour chaque canal
        weights = gradients.mean(dim=(1, 2))  # [C]
        
        # Calculer la carte de chaleur
        heatmap = torch.zeros(activations.shape[1:], device=activations.device)  # [H, W]
        
        for i, w in enumerate(weights):
            heatmap += w * activations[i]
        
        # ReLU sur la carte de chaleur
        heatmap = F.relu(heatmap)
        
        # Normaliser entre 0 et 1
        heatmap = heatmap - heatmap.min()
        heatmap = heatmap / (heatmap.max() + 1e-8)
        
        # Convertir en numpy
        heatmap = heatmap.cpu().numpy()
        
        return heatmap, output, class_idx


def apply_heatmap_to_image(image, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Applique une carte de chaleur sur une image.
    
    Args:
        image (PIL.Image or numpy.ndarray): Image d'origine
        heatmap (numpy.ndarray): Carte de chaleur (valeurs entre 0 et 1)
        alpha (float): Coefficient de mélange (0.0 à 1.0)
        colormap (int): Colormap OpenCV à utiliser
        
    Returns:
        numpy.ndarray: Image avec la carte de chaleur superposée
    """
    # Convertir l'image en numpy si nécessaire
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Redimensionner la carte de chaleur à la taille de l'image
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Appliquer le colormap à la carte de chaleur
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), colormap)
    
    # Convertir l'image en RGB si elle est en BGR (OpenCV)
    if image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    # Superposer la carte de chaleur et l'image
    superimposed = cv2.addWeighted(image_rgb, 1 - alpha, heatmap_colored, alpha, 0)
    
    return superimposed


def visualize_model_attention(model, image_tensor, class_idx=None, target_layer_name=None, save_path=None, verbose=False):
    """
    Visualise l'attention du modèle sur une image à l'aide de Grad-CAM.
    
    Args:
        model (nn.Module): Modèle CNN à analyser
        image_tensor (torch.Tensor): Tensor d'image d'entrée (normalisé)
        class_idx (int, optional): Indice de classe à analyser
        target_layer_name (str, optional): Nom de la couche de convolution cible
        save_path (str, optional): Chemin pour sauvegarder la figure
        verbose (bool): Si True, affiche les messages de progression
        
    Returns:
        numpy.ndarray: Image avec la carte de chaleur superposée
    """
    # Détecter automatiquement la dernière couche de convolution si target_layer_name n'est pas fourni
    if target_layer_name is None:
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, torch.nn.Conv2d):
                target_layer_name = name
                break
        if verbose:
            logging.info(f"Couche cible automatiquement détectée: {target_layer_name}")
    
    # Créer une instance de GradCAM
    grad_cam = GradCAM(model, target_layer_name, verbose=verbose)
    
    # Générer la carte de chaleur
    heatmap, output, pred_class = grad_cam(image_tensor.unsqueeze(0), class_idx)
    
    # Récupérer l'image originale
    # Dénormaliser l'image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    original_image = image_tensor.cpu() * std + mean
    original_image = original_image.permute(1, 2, 0).numpy()
    original_image = np.clip(original_image, 0, 1)
    
    # Appliquer la carte de chaleur à l'image originale
    superimposed = apply_heatmap_to_image(
        (original_image * 255).astype(np.uint8), 
        heatmap, 
        alpha=0.5
    )
    
    # Afficher les résultats
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title(f"Image originale\nClasse prédite: {pred_class}")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap='jet')
    plt.title("Carte d'attention (Grad-CAM)")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(superimposed)
    plt.title("Superposition")
    plt.axis('off')
    
    plt.tight_layout()
    
    # Sauvegarde de la figure
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        if verbose:
            print(f"Figure sauvegardée dans {save_path}")
    
    plt.close()
    
    return superimposed


def visualize_patch_attention(model, patches_tensor, save_path=None, verbose=False):
    """
    Visualise les poids d'attention attribués aux différents patches par le modèle PatchCNN.
    
    Args:
        model (PatchCNN): Modèle basé sur des patches à analyser
        patches_tensor (torch.Tensor): Tensor de patches d'entrée [num_patches, 3, H, W]
        save_path (str, optional): Chemin pour sauvegarder la figure
        verbose (bool): Si True, affiche les messages de progression
        
    Returns:
        numpy.ndarray: Visualisation des poids d'attention des patches
    """
    model.eval()
    
    with torch.no_grad():
        # Extraire les caractéristiques pour chaque patch
        patch_features = model.feature_extractor(patches_tensor.unsqueeze(0))
        
        # Récupérer les poids d'attention
        if model.aggregation_type == "attention":
            attention_scores = model.attention(patch_features)
            attention_weights = F.softmax(attention_scores, dim=1)
            weights = attention_weights.squeeze().cpu().numpy()
        elif model.aggregation_type == "weighted_avg":
            weights = model.patch_weights.squeeze().cpu().numpy()
        else:  # "avg"
            weights = np.ones(patches_tensor.size(0)) / patches_tensor.size(0)
    
    # Convertir les patches en images pour la visualisation
    patches_np = []
    for i in range(patches_tensor.size(0)):
        # Dénormaliser le patch
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        patch = patches_tensor[i].cpu() * std + mean
        patch = patch.permute(1, 2, 0).numpy()
        patch = np.clip(patch, 0, 1)
        patches_np.append(patch)
    
    # Créer une figure pour visualiser les patches et leurs poids
    n_patches = len(patches_np)
    n_cols = min(4, n_patches)
    n_rows = (n_patches + n_cols - 1) // n_cols
    
    plt.figure(figsize=(n_cols * 3, n_rows * 3))
    
    for i in range(n_patches):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(patches_np[i])
        plt.title(f"Patch {i}\nPoids: {weights[i]:.3f}")
        plt.axis('off')
    
    plt.tight_layout()
    
    # Sauvegarde de la figure
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        if verbose:
            print(f"Figure sauvegardée dans {save_path}")
    
    plt.close()
    
    # Créer une visualisation des patches triés par poids d'attention
    sorted_indices = np.argsort(-weights)
    sorted_patches = [patches_np[i] for i in sorted_indices]
    sorted_weights = weights[sorted_indices]
    
    # Créer une grille d'images avec des bordures colorées selon le poids
    grid_size = int(np.ceil(np.sqrt(n_patches)))
    grid_height = grid_size * patches_np[0].shape[0]
    grid_width = grid_size * patches_np[0].shape[1]
    grid = np.ones((grid_height, grid_width, 3))
    
    for i in range(n_patches):
        row = i // grid_size
        col = i % grid_size
        h_start = row * patches_np[0].shape[0]
        w_start = col * patches_np[0].shape[1]
        
        # Ajouter le patch à la grille
        patch_with_border = sorted_patches[i].copy()
        
        # Ajouter une bordure colorée selon le poids (rouge = important, bleu = moins important)
        border_width = 3
        color = np.array([1-sorted_weights[i], 0, sorted_weights[i]])  # [R, G, B]
        
        # Bordure supérieure et inférieure
        patch_with_border[:border_width, :, :] = color
        patch_with_border[-border_width:, :, :] = color
        
        # Bordure gauche et droite
        patch_with_border[:, :border_width, :] = color
        patch_with_border[:, -border_width:, :] = color
        
        grid[h_start:h_start+patches_np[0].shape[0], w_start:w_start+patches_np[0].shape[1], :] = patch_with_border
    
    return grid 