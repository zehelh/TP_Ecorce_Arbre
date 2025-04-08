"""
Module contenant les fonctions pour l'entraînement des modèles.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime

from ..config import MODEL_SAVE_DIR, LOG_DIR, NUM_EPOCHS, LEARNING_RATE, MOMENTUM, WEIGHT_DECAY


class EarlyStopping:
    """
    Classe pour arrêter l'entraînement si la validation loss n'améliore plus.
    """
    
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): Nombre d'époques à attendre après la dernière amélioration
            verbose (bool): Si True, affiche un message pour chaque amélioration
            delta (float): Différence minimale pour considérer une amélioration
            path (str): Chemin pour sauvegarder le meilleur modèle
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
    
    def __call__(self, val_loss, model, optimizer, epoch):
        """
        Vérifie si l'entraînement doit être arrêté.
        
        Args:
            val_loss (float): Perte de validation actuelle
            model (nn.Module): Modèle à sauvegarder
            optimizer (optim.Optimizer): Optimiseur
            epoch (int): Époque actuelle
        """
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping: patience {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model, optimizer, epoch):
        """
        Sauvegarde le modèle lorsqu'une meilleure validation loss est obtenue.
        
        Args:
            val_loss (float): Perte de validation
            model (nn.Module): Modèle à sauvegarder
            optimizer (optim.Optimizer): Optimiseur
            epoch (int): Époque actuelle
        """
        if self.verbose:
            print(f'Validation loss améliorée ({self.val_loss_min:.6f} --> {val_loss:.6f}). Sauvegarde du modèle...')
        
        # Sauvegarde du modèle
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss
        }
        torch.save(checkpoint, self.path)
        self.val_loss_min = val_loss


def train_epoch(model, train_loader, criterion, optimizer, device, disable_tqdm=False):
    """
    Entraîne le modèle pour une époque.
    
    Args:
        model (nn.Module): Modèle à entraîner
        train_loader (DataLoader): DataLoader pour les données d'entraînement
        criterion (nn.Module): Fonction de perte
        optimizer (optim.Optimizer): Optimiseur
        device (torch.device): Appareil à utiliser
        disable_tqdm (bool): Si True, désactive la barre de progression tqdm
        
    Returns:
        tuple: (perte moyenne, précision moyenne)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Boucle d'entraînement
    for inputs, targets in tqdm(train_loader, desc="Train", leave=False, disable=disable_tqdm):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Réinitialiser les gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Métriques
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    
    # Calculer les métriques finales
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def validate_epoch(model, val_loader, criterion, device, disable_tqdm=False):
    """
    Valide le modèle sur l'ensemble de validation.
    
    Args:
        model (nn.Module): Modèle à valider
        val_loader (DataLoader): DataLoader pour les données de validation
        criterion (nn.Module): Fonction de perte
        device (torch.device): Appareil à utiliser
        disable_tqdm (bool): Si True, désactive la barre de progression tqdm
        
    Returns:
        tuple: (perte moyenne, précision moyenne)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Valid", leave=False, disable=disable_tqdm):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Métriques
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    # Calculer les métriques finales
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, model_name="model", num_epochs=100, 
                learning_rate=0.001, momentum=0.9, weight_decay=1e-4, patience=10, device=None,
                verbose=True):
    """
    Entraîne le modèle avec early stopping.
    
    Args:
        model (nn.Module): Modèle à entraîner
        train_loader (DataLoader): DataLoader pour les données d'entraînement
        val_loader (DataLoader): DataLoader pour les données de validation
        model_name (str): Nom du modèle pour la sauvegarde
        num_epochs (int): Nombre maximum d'époques
        learning_rate (float): Taux d'apprentissage
        momentum (float): Momentum pour l'optimiseur SGD
        weight_decay (float): Régularisation L2
        patience (int): Patience pour l'early stopping
        device (torch.device, optional): Appareil à utiliser (auto-détecté si None)
        verbose (bool): Si True, affiche la progression pendant l'entraînement
        
    Returns:
        tuple: (model, history_dict)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    
    # Critère et optimiseur
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    
    # Créer le répertoire pour sauvegarder les modèles si nécessaire
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    best_model_path = os.path.join(MODEL_SAVE_DIR, f"{model_name}_best.pth")
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience, verbose=verbose, path=best_model_path)
    
    # Historique des métriques
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # Désactiver tqdm si non verbeux
    disable_tqdm = not verbose
    
    # Boucle d'entraînement
    for epoch in range(num_epochs):
        # Entraînement et validation
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, disable_tqdm)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device, disable_tqdm)
        
        # Sauvegarde de l'historique
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Affichage des métriques
        if verbose:
            print(f"Époque {epoch+1}/{num_epochs} : "
                  f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        
        # Early stopping
        early_stopping(val_loss, model, optimizer, epoch)
        if early_stopping.early_stop:
            if verbose:
                print(f"Early stopping à l'époque {epoch+1}")
            break
    
    # Charger le meilleur modèle
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Sauvegarde du modèle final
    final_model_path = os.path.join(MODEL_SAVE_DIR, f"{model_name}_final.pth")
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss
    }, final_model_path)
    
    if verbose:
        print(f"Entraînement terminé. Meilleur modèle sauvegardé à {best_model_path}")
    
    return model, history 