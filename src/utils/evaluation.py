"""
Module contenant les fonctions pour l'évaluation des modèles.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import seaborn as sns

from ..config import FIGURES_DIR


def evaluate_model(model, dataloader, criterion=None, device=None, verbose=True):
    """
    Évalue le modèle sur un ensemble de données.
    
    Args:
        model (nn.Module): Modèle à évaluer
        dataloader (DataLoader): DataLoader pour les données de test
        criterion (nn.Module, optional): Fonction de perte
        device (torch.device, optional): Appareil à utiliser (auto-détecté si None)
        verbose (bool): Si True, affiche la barre de progression
        
    Returns:
        dict: Dictionnaire contenant les métriques d'évaluation
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    running_loss = 0.0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Évaluation", disable=not verbose):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calcul de la perte
            if criterion is not None:
                loss = criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)
            
            # Prédictions
            _, predicted = torch.max(outputs, 1)
            
            # Stockage des prédictions et des cibles
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            total += targets.size(0)
    
    # Conversion en tableaux numpy
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Calcul des métriques
    accuracy = accuracy_score(all_targets, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_predictions, average='weighted')
    
    # Perte moyenne (si criterion est fourni)
    avg_loss = running_loss / total if criterion is not None else None
    
    # Rapport de classification
    class_report = classification_report(all_targets, all_predictions, output_dict=True)
    
    # Construction du dictionnaire de métriques
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'loss': avg_loss,
        'class_report': class_report,
        'all_predictions': all_predictions,
        'all_targets': all_targets
    }
    
    return metrics


def plot_confusion_matrix(cm, class_names=None, save_path=None, figsize=(12, 10), verbose=True):
    """
    Affiche et sauvegarde la matrice de confusion.
    
    Args:
        cm (numpy.ndarray): Matrice de confusion
        class_names (list, optional): Noms des classes
        save_path (str, optional): Chemin pour sauvegarder la figure
        figsize (tuple): Taille de la figure
        verbose (bool): Si True, affiche un message lors de la sauvegarde
    """
    plt.figure(figsize=figsize)
    
    ax = sns.heatmap(cm, annot=False, cmap='Blues', fmt='g')
    
    # Labels
    ax.set_xlabel('Classe prédite')
    ax.set_ylabel('Classe réelle')
    ax.set_title('Matrice de confusion')
    
    # Ajout des noms de classes (si fournis)
    if class_names is not None:
        # Si trop de classes, on ne les affiche pas toutes
        if len(class_names) > 20:
            tick_marks = np.arange(0, len(class_names), len(class_names) // 10)
            plt.xticks(tick_marks + 0.5, [class_names[i] for i in tick_marks], rotation=90)
            plt.yticks(tick_marks + 0.5, [class_names[i] for i in tick_marks], rotation=0)
        else:
            plt.xticks(np.arange(len(class_names)) + 0.5, class_names, rotation=90)
            plt.yticks(np.arange(len(class_names)) + 0.5, class_names, rotation=0)
    
    # Sauvegarde de la figure
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        if verbose:
            print(f"Figure sauvegardée dans {save_path}")
    
    plt.tight_layout()
    plt.close()


def plot_metrics_history(history, save_path=None, verbose=True):
    """
    Affiche et sauvegarde les courbes d'apprentissage.
    
    Args:
        history (dict): Historique des métriques (loss, accuracy)
        save_path (str, optional): Chemin pour sauvegarder la figure
        verbose (bool): Si True, affiche un message lors de la sauvegarde
    """
    plt.figure(figsize=(12, 5))
    
    # Perte
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Évolution de la perte')
    plt.xlabel('Époque')
    plt.ylabel('Perte')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Précision
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Évolution de la précision')
    plt.xlabel('Époque')
    plt.ylabel('Précision')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Sauvegarde de la figure
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        if verbose:
            print(f"Figure sauvegardée dans {save_path}")
    
    plt.close()


def plot_class_metrics(class_report, save_path=None, verbose=True):
    """
    Affiche et sauvegarde les métriques par classe.
    
    Args:
        class_report (dict): Rapport de classification
        save_path (str, optional): Chemin pour sauvegarder la figure
        verbose (bool): Si True, affiche un message lors de la sauvegarde
    """
    # Extraction des métriques pertinentes
    classes = []
    precision = []
    recall = []
    f1 = []
    support = []
    
    for class_id, metrics in class_report.items():
        # Ignorer les métriques 'accuracy', 'macro avg' et 'weighted avg'
        if class_id not in ['accuracy', 'macro avg', 'weighted avg']:
            classes.append(class_id)
            precision.append(metrics['precision'])
            recall.append(metrics['recall'])
            f1.append(metrics['f1-score'])
            support.append(metrics['support'])
    
    # Conversion des listes en tableaux numpy
    classes = np.array(classes)
    precision = np.array(precision)
    recall = np.array(recall)
    f1 = np.array(f1)
    support = np.array(support)
    
    # Tri par support
    sort_idx = np.argsort(-support)
    classes = classes[sort_idx]
    precision = precision[sort_idx]
    recall = recall[sort_idx]
    f1 = f1[sort_idx]
    support = support[sort_idx]
    
    # Ne garder que les 20 classes les plus représentées
    if len(classes) > 20:
        classes = classes[:20]
        precision = precision[:20]
        recall = recall[:20]
        f1 = f1[:20]
        support = support[:20]
    
    # Création de la figure
    plt.figure(figsize=(12, 8))
    
    width = 0.2
    x = np.arange(len(classes))
    
    plt.bar(x - width, precision, width, label='Précision')
    plt.bar(x, recall, width, label='Rappel')
    plt.bar(x + width, f1, width, label='F1-score')
    
    plt.xlabel('Classe')
    plt.ylabel('Score')
    plt.title('Métriques par classe')
    plt.xticks(x, classes, rotation=90)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Sauvegarde de la figure
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        if verbose:
            print(f"Figure sauvegardée dans {save_path}")
    
    plt.close()


def evaluate_and_visualize(model, test_loader, model_name="model", history=None, device=None, verbose=True):
    """
    Évalue le modèle et génère les visualisations des résultats.
    
    Args:
        model (nn.Module): Modèle à évaluer
        test_loader (DataLoader): DataLoader pour les données de test
        model_name (str): Nom du modèle pour la sauvegarde des figures
        history (dict, optional): Historique des métriques d'entraînement
        device (torch.device, optional): Appareil à utiliser
        verbose (bool): Si True, affiche les messages de progression
        
    Returns:
        dict: Métriques d'évaluation
    """
    # Évaluation du modèle
    metrics = evaluate_model(model, test_loader, device=device, verbose=verbose)
    
    # Création du répertoire pour les figures
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # Matrice de confusion
    cm = confusion_matrix(metrics['all_targets'], metrics['all_predictions'])
    plot_confusion_matrix(
        cm, 
        save_path=os.path.join(FIGURES_DIR, f"{model_name}_confusion_matrix.png"),
        verbose=verbose
    )
    
    # Métriques par classe
    plot_class_metrics(
        metrics['class_report'],
        save_path=os.path.join(FIGURES_DIR, f"{model_name}_class_metrics.png"),
        verbose=verbose
    )
    
    # Courbes d'apprentissage (si history est fourni)
    if history:
        plot_metrics_history(
            history,
            save_path=os.path.join(FIGURES_DIR, f"{model_name}_learning_curves.png"),
            verbose=verbose
        )
    
    # Affichage des métriques globales
    if verbose:
        print(f"Évaluation du modèle {model_name}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-score: {metrics['f1']:.4f}")
        if metrics['loss'] is not None:
            print(f"  Loss: {metrics['loss']:.4f}")
    
    return metrics 