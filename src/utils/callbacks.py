"""
Module contenant des callbacks pour l'entraînement des modèles,
notamment EarlyStopping et CheckpointSaver.
"""

import os
import numpy as np
import torch
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Callback pour arrêter l'entraînement si la métrique surveillée cesse de s'améliorer.
    
    Args:
        patience (int): Nombre d'époques à attendre après la dernière amélioration
        verbose (bool): Si True, affiche un message quand un arrêt anticipé est déclenché
        delta (float): Seuil de changement minimum pour qualifier une amélioration
        mode (str): 'min' ou 'max', selon la direction de la métrique surveillée
    """
    
    def __init__(self, patience=10, verbose=True, delta=0.0, mode='min'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.mode = mode
        self.val_loss_min = np.inf if mode == 'min' else -np.inf
        self.improvement = False
    
    def __call__(self, val_loss):
        """
        Vérifie si l'entraînement doit être arrêté en fonction de la métrique surveillée.
        
        Args:
            val_loss (float): Valeur de la métrique surveillée
            
        Returns:
            bool: True si l'entraînement doit être arrêté, False sinon
        """
        score = -val_loss if self.mode == 'min' else val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.improvement = True
            if self.mode == 'min':
                self.val_loss_min = val_loss
            return False
        
        if score < self.best_score + self.delta:
            self.counter += 1
            self.improvement = False
            if self.verbose:
                logger.info(f'EarlyStopping: patience {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_score = score
            self.counter = 0
            self.improvement = True
            if self.mode == 'min':
                self.val_loss_min = val_loss
            if self.verbose:
                delta_message = f'{val_loss:.6f} vs {self.val_loss_min:.6f}' if self.mode == 'min' else f'{val_loss:.6f}'
                logger.info(f'EarlyStopping: amélioration ({delta_message})')
        
        return False


class CheckpointSaver:
    """
    Callback pour sauvegarder les checkpoints du modèle.
    
    Args:
        checkpoint_dir (str): Répertoire où sauvegarder les checkpoints
        metric_name (str): Nom de la métrique à surveiller
        maximize (bool): Si True, considère des valeurs plus élevées comme meilleures
        save_best_only (bool): Si True, ne sauvegarde que le meilleur modèle
        checkpoint_prefix (str): Préfixe pour les noms de fichiers de checkpoint
    """
    
    def __init__(self, checkpoint_dir, metric_name='val_loss', maximize=False, 
                 save_best_only=True, checkpoint_prefix='checkpoint'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metric_name = metric_name
        self.maximize = maximize
        self.save_best_only = save_best_only
        self.checkpoint_prefix = checkpoint_prefix
        
        self.best_value = -np.inf if maximize else np.inf
        self.best_checkpoint_path = None
        
        # Initialiser le fichier de suivi des métriques
        self.metrics_file = self.checkpoint_dir / 'training_metrics.json'
        self.metrics_history = []
    
    def is_better(self, current_value):
        """
        Vérifie si la valeur actuelle est meilleure que la valeur précédente.
        
        Args:
            current_value (float): Valeur actuelle de la métrique
            
        Returns:
            bool: True si la valeur actuelle est meilleure, False sinon
        """
        if self.maximize:
            return current_value > self.best_value
        else:
            return current_value < self.best_value
    
    def save_checkpoint(self, model, optimizer, epoch, metrics=None):
        """
        Sauvegarde le modèle et, éventuellement, l'optimiseur et les métriques.
        
        Args:
            model (torch.nn.Module): Modèle à sauvegarder
            optimizer (torch.optim.Optimizer): Optimiseur à sauvegarder
            epoch (int): Époque actuelle
            metrics (dict): Dictionnaire de métriques à sauvegarder
            
        Returns:
            str: Chemin du fichier de checkpoint sauvegardé, ou None si aucun fichier n'a été sauvegardé
        """
        if metrics is None:
            metrics = {}
        
        # Sauvegarder l'historique des métriques
        epoch_metrics = {'epoch': epoch}
        epoch_metrics.update(metrics)
        self.metrics_history.append(epoch_metrics)
        
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        # Vérifier si on doit sauvegarder un checkpoint
        current_value = metrics.get(self.metric_name)
        if current_value is None:
            return None
        
        is_best = self.is_better(current_value)
        
        # Toujours sauvegarder le dernier modèle
        last_checkpoint_path = self.checkpoint_dir / f"{self.checkpoint_prefix}_last.pth"
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        torch.save(checkpoint, last_checkpoint_path)
        
        # Sauvegarder le meilleur modèle
        if is_best:
            self.best_value = current_value
            best_checkpoint_path = self.checkpoint_dir / f"{self.checkpoint_prefix}_best.pth"
            torch.save(checkpoint, best_checkpoint_path)
            self.best_checkpoint_path = best_checkpoint_path
            
            # Créer un fichier pour indiquer la performance du meilleur modèle
            best_info = {
                'epoch': epoch,
                'metrics': metrics,
                'checkpoint_path': str(best_checkpoint_path)
            }
            with open(self.checkpoint_dir / 'best_model_info.json', 'w') as f:
                json.dump(best_info, f, indent=2)
            
            logger.info(f"Sauvegarde du meilleur modèle avec {self.metric_name}={current_value:.4f} à l'époque {epoch}")
            return str(best_checkpoint_path)
        
        # Sauvegarder un checkpoint à chaque époque si demandé
        if not self.save_best_only:
            checkpoint_path = self.checkpoint_dir / f"{self.checkpoint_prefix}_{epoch:03d}.pth"
            torch.save(checkpoint, checkpoint_path)
            return str(checkpoint_path)
        
        return None
    
    def get_best_checkpoint_path(self):
        """
        Retourne le chemin du meilleur checkpoint sauvegardé.
        
        Returns:
            str: Chemin du meilleur checkpoint, ou None si aucun checkpoint n'a été sauvegardé
        """
        return self.best_checkpoint_path 