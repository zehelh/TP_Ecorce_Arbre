"""
Module contenant des fonctions pour calculer diverses métriques d'évaluation
pour les modèles de classification d'écorces.
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm


def calculate_metrics(model, data_loader, criterion=None, device='cpu', verbose=True):
    """
    Calcule les métriques d'évaluation pour un modèle donné sur un jeu de données.
    
    Args:
        model (torch.nn.Module): Le modèle à évaluer
        data_loader (DataLoader): Le DataLoader contenant les données d'évaluation
        criterion (callable, optional): Fonction de perte pour calculer la loss
        device (str ou torch.device): Device à utiliser pour l'évaluation
        verbose (bool): Si True, affiche une barre de progression
    
    Returns:
        dict: Dictionnaire contenant les métriques calculées
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    running_loss = 0.0
    
    # Désactiver le calcul de gradient pour l'évaluation
    with torch.no_grad():
        # Créer une barre de progression si verbose est True
        data_iter = tqdm(data_loader, desc="Évaluation", disable=not verbose)
        
        for inputs, targets in data_iter:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculer la perte si un critère est fourni
            if criterion is not None:
                loss = criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)
            
            # Obtenir les prédictions
            _, predictions = torch.max(outputs, 1)
            
            # Stocker les prédictions et cibles pour calculer les métriques
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Convertir les listes en tableaux NumPy
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Calculer les métriques
    metrics = {
        'accuracy': accuracy_score(all_targets, all_preds),
        'precision_macro': precision_score(all_targets, all_preds, average='macro', zero_division=0),
        'recall_macro': recall_score(all_targets, all_preds, average='macro', zero_division=0),
        'f1_macro': f1_score(all_targets, all_preds, average='macro', zero_division=0),
        'precision_weighted': precision_score(all_targets, all_preds, average='weighted', zero_division=0),
        'recall_weighted': recall_score(all_targets, all_preds, average='weighted', zero_division=0),
        'f1_weighted': f1_score(all_targets, all_preds, average='weighted', zero_division=0),
    }
    
    # Ajouter la perte si un critère a été fourni
    if criterion is not None:
        metrics['loss'] = running_loss / len(data_loader.dataset)
    
    return metrics


def get_class_metrics(model, data_loader, class_names=None, device='cpu'):
    """
    Calcule les métriques par classe pour un modèle donné sur un jeu de données.
    
    Args:
        model (torch.nn.Module): Le modèle à évaluer
        data_loader (DataLoader): Le DataLoader contenant les données d'évaluation
        class_names (list, optional): Liste des noms de classes
        device (str ou torch.device): Device à utiliser pour l'évaluation
    
    Returns:
        dict: Dictionnaire contenant les métriques par classe et la matrice de confusion
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    
    # Désactiver le calcul de gradient pour l'évaluation
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="Évaluation par classe"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Obtenir les prédictions
            _, predictions = torch.max(outputs, 1)
            
            # Stocker les prédictions et cibles
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Convertir les listes en tableaux NumPy
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Calculer la matrice de confusion
    cm = confusion_matrix(all_targets, all_preds)
    
    # Obtenir le rapport de classification détaillé
    report = classification_report(all_targets, all_preds, output_dict=True, zero_division=0)
    
    # Extraire les métriques par classe
    class_metrics = {}
    for class_idx, metrics in report.items():
        if class_idx.isdigit():
            class_idx = int(class_idx)
            class_name = class_names[class_idx] if class_names and class_idx < len(class_names) else f"Classe {class_idx}"
            class_metrics[class_name] = {
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1-score': metrics['f1-score'],
                'support': metrics['support']
            }
    
    # Créer le dictionnaire de résultats
    results = {
        'confusion_matrix': cm,
        'class_metrics': class_metrics,
        'accuracy': report['accuracy'],
        'macro_avg': report['macro avg'],
        'weighted_avg': report['weighted avg']
    }
    
    return results


def calculate_prediction_confidence(model, data_loader, device='cpu'):
    """
    Calcule les scores de confiance des prédictions pour un modèle donné sur un jeu de données.
    
    Args:
        model (torch.nn.Module): Le modèle à évaluer
        data_loader (DataLoader): Le DataLoader contenant les données d'évaluation
        device (str ou torch.device): Device à utiliser pour l'évaluation
    
    Returns:
        tuple: (prédictions, cibles, scores de confiance)
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    all_confidences = []
    
    # Désactiver le calcul de gradient pour l'évaluation
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="Calcul des confidences"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Appliquer softmax pour obtenir les probabilités
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Obtenir les prédictions et leurs confidences (probabilités max)
            confidences, predictions = torch.max(probabilities, 1)
            
            # Stocker les prédictions, confidences et cibles
            all_preds.extend(predictions.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Convertir les listes en tableaux NumPy
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_confidences = np.array(all_confidences)
    
    return all_preds, all_targets, all_confidences 