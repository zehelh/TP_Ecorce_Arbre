"""
Script d'entraînement en deux phases pour le modèle EnhancedPatchOptimizedModel.

Ce script implémente une stratégie d'entraînement en deux phases:
1. Phase 1: Entraînement des couches supérieures avec backbone gelé (transfer learning)
2. Phase 2: Fine-tuning progressif du backbone ResNet50 avec learning rates différenciés

Cela permet d'obtenir de meilleures performances en évitant le surapprentissage
et en adaptant progressivement le modèle pré-entraîné aux spécificités du dataset d'écorces.

Inspiré par les techniques utilisées dans le fine-tuning de modèles de vision avancés
comme EfficientNet et ViT.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

from src.data import get_dataloaders
from src.utils.callbacks import EarlyStopping, CheckpointSaver
from src.utils.metrics import calculate_metrics
from src.config import MODEL_SAVE_DIR, LOG_DIR, PATCH_SIZE, PATCH_STRIDE, NUM_PATCHES

from enhanced_patch_optimized_model import EnhancedPatchOptimizedModel


def set_seed(seed):
    """
    Fixe les graines aléatoires pour la reproductibilité.
    
    Args:
        seed (int): Graine aléatoire
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_optimizers(model, phase, lr, momentum, weight_decay, lr_backbone_factor=0.1):
    """
    Crée les optimiseurs pour les différentes phases d'entraînement.
    
    Args:
        model (EnhancedPatchOptimizedModel): Le modèle à entraîner
        phase (int): Phase d'entraînement (1 ou 2)
        lr (float): Learning rate de base
        momentum (float): Momentum pour l'optimiseur SGD
        weight_decay (float): Weight decay pour la régularisation L2
        lr_backbone_factor (float): Facteur de réduction du learning rate pour le backbone
        
    Returns:
        torch.optim.Optimizer: L'optimiseur configuré
    """
    # Récupérer les groupes de paramètres
    param_groups = model.get_parameter_groups()
    
    # Configurer le modèle pour la phase spécifiée
    lr_multipliers = model.configure_phase(
        phase=phase,
        unfreeze_layers=3 if phase == 2 else 0,  # Dégeler les 3 dernières couches en phase 2
        lr_multiplier=lr_backbone_factor
    )
    
    # Appliquer les learning rates spécifiques à chaque groupe
    for group in param_groups:
        group_name = group.pop('group')
        group['lr'] = lr * lr_multipliers.get(group_name, 1.0)
    
    # Créer l'optimiseur
    optimizer = optim.SGD(
        param_groups,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True
    )
    
    return optimizer


def create_scheduler(optimizer, phase, num_epochs, warmup_epochs=5):
    """
    Crée un scheduler de learning rate adapté à la phase d'entraînement.
    
    Args:
        optimizer (torch.optim.Optimizer): L'optimiseur à utiliser
        phase (int): Phase d'entraînement (1 ou 2)
        num_epochs (int): Nombre total d'époques
        warmup_epochs (int): Nombre d'époques de warm-up
        
    Returns:
        torch.optim.lr_scheduler._LRScheduler: Le scheduler configuré
    """
    # Phase 1: Scheduling plus simple
    if phase == 1:
        return CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # Phase 2: Warm-up linéaire puis décroissance cosinus
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=0.1, 
        end_factor=1.0, 
        total_iters=warmup_epochs
    )
    
    cosine_scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs - warmup_epochs,
        eta_min=1e-6
    )
    
    # Combiner les deux schedulers
    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )


def train_phase(model, phase, train_loader, val_loader, device, args):
    """
    Entraîne le modèle pour une phase spécifique.
    
    Args:
        model (EnhancedPatchOptimizedModel): Le modèle à entraîner
        phase (int): Numéro de phase (1 ou 2)
        train_loader (DataLoader): DataLoader pour l'ensemble d'entraînement
        val_loader (DataLoader): DataLoader pour l'ensemble de validation
        device (torch.device): Device à utiliser (CPU ou GPU)
        args (argparse.Namespace): Arguments de ligne de commande
        
    Returns:
        tuple: (modèle entraîné, historique d'entraînement, meilleur modèle)
    """
    # Paramètres spécifiques à la phase
    if phase == 1:
        phase_name = "phase1_head"
        lr = args.lr
        num_epochs = args.phase1_epochs
        patience = args.patience
    else:
        phase_name = "phase2_finetune"
        lr = args.lr * 0.1  # Learning rate réduit pour la phase 2
        num_epochs = args.phase2_epochs
        patience = args.patience * 2  # Plus de patience pour la phase 2
    
    # Nom du modèle
    model_name = f"enhanced_patch_optimized_{phase_name}"
    
    print(f"\n{'='*80}")
    print(f"DÉMARRAGE DE L'ENTRAÎNEMENT - PHASE {phase}")
    print(f"Learning rate: {lr}")
    print(f"Nombre d'époques: {num_epochs}")
    print(f"{'='*80}\n")
    
    # Critère de perte
    criterion = nn.CrossEntropyLoss()
    
    # Optimiseur différencié selon la phase
    optimizer = create_optimizers(
        model=model,
        phase=phase,
        lr=lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        lr_backbone_factor=0.1 if phase == 2 else 0
    )
    
    # Scheduler de learning rate
    scheduler = create_scheduler(
        optimizer=optimizer,
        phase=phase,
        num_epochs=num_epochs,
        warmup_epochs=5 if phase == 2 else 0
    )
    
    # Callbacks pour early stopping et sauvegarde des checkpoints
    early_stopping = EarlyStopping(patience=patience, verbose=True, delta=0.001)
    checkpoint_dir = os.path.join(MODEL_SAVE_DIR, model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_saver = CheckpointSaver(
        checkpoint_dir=checkpoint_dir,
        metric_name='val_acc',
        maximize=True
    )
    
    # Historique d'entraînement
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Pour suivre le meilleur modèle
    best_model = None
    best_val_acc = 0.0
    
    # Heure de début
    start_time = time.time()
    
    # Boucle d'entraînement
    for epoch in range(num_epochs):
        # Mode entraînement
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Progression de l'époque
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass et optimisation
            loss.backward()
            optimizer.step()
            
            # Statistiques
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Mise à jour de la barre de progression
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': 100. * correct / total,
                'lr': optimizer.param_groups[0]['lr']
            })
        
        # Calcul des métriques d'entraînement
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100. * correct / total
        
        # Mode évaluation
        model.eval()
        val_metrics = calculate_metrics(model, val_loader, criterion, device)
        val_loss = val_metrics['loss']
        val_acc = val_metrics['accuracy'] * 100
        
        # Mise à jour du scheduler
        scheduler.step()
        
        # Mise à jour de l'historique
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Affichage des résultats
        print(f"\nÉpoque {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.7f}")
        
        # Sauvegarde du meilleur modèle
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }
        
        # Sauvegarde du checkpoint
        checkpoint_saver.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics={
                'val_acc': val_acc,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'train_loss': train_loss
            }
        )
        
        # Early stopping
        early_stop = early_stopping(val_loss)
        if early_stop:
            print(f"Early stopping à l'époque {epoch+1}")
            break
    
    # Durée totale de l'entraînement
    training_time = time.time() - start_time
    hours, rem = divmod(training_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\nTemps d'entraînement de la phase {phase}: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    # Sauvegarder les hyperparamètres et les résultats
    hyperparams_file = os.path.join(LOG_DIR, f"{model_name}_hyperparams.txt")
    with open(hyperparams_file, 'w') as f:
        f.write(f"model: enhanced_patch_optimized_phase{phase}\n")
        f.write(f"lr: {lr}\n")
        f.write(f"momentum: {args.momentum}\n")
        f.write(f"weight_decay: {args.weight_decay}\n")
        f.write(f"backbone: resnet50\n")
        f.write(f"use_context: True\n")
        f.write(f"num_patches: {args.num_patches}\n")
        f.write(f"patch_size: {args.patch_size}\n")
        f.write(f"patch_stride: {args.patch_stride}\n")
        f.write(f"epochs: {epoch+1}\n")
        f.write(f"patience: {patience}\n")
        f.write(f"training_time: {training_time:.2f} seconds\n")
    
    # Sauvegarder les courbes d'apprentissage
    plot_learning_curves(
        history, 
        os.path.join(LOG_DIR, f"{model_name}_learning_curves.png")
    )
    
    return model, history, best_model


def plot_learning_curves(history, save_path):
    """
    Trace et sauvegarde les courbes d'apprentissage.
    
    Args:
        history (dict): Historique d'entraînement
        save_path (str): Chemin pour sauvegarder le graphique
    """
    plt.figure(figsize=(12, 5))
    
    # Précision
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Précision')
    plt.xlabel('Époque')
    plt.ylabel('Précision (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Perte
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Perte')
    plt.xlabel('Époque')
    plt.ylabel('Perte')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main(args):
    """
    Fonction principale d'entraînement en deux phases.
    
    Args:
        args (argparse.Namespace): Arguments de ligne de commande
    """
    # Création des répertoires de sauvegarde
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Fixer la graine aléatoire
    set_seed(args.seed)
    
    # Déterminer l'appareil à utiliser (CPU ou GPU)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Utilisation de l'appareil: {device}")
    
    # Créer les dataloaders
    train_loader, val_loader = get_dataloaders(
        use_patches=True,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        num_patches=args.num_patches
    )
    
    print(f"Dataset d'entraînement: {len(train_loader.dataset)} images")
    print(f"Dataset de validation: {len(val_loader.dataset)} images")
    
    # Créer le modèle
    model = EnhancedPatchOptimizedModel(
        num_patches=args.num_patches,
        pretrained=True
    )
    model = model.to(device)
    
    if args.skip_phase1 and args.phase1_checkpoint:
        # Charger le checkpoint de la phase 1
        print(f"Chargement du checkpoint de la phase 1: {args.phase1_checkpoint}")
        checkpoint = torch.load(args.phase1_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        phase1_history = None
        phase1_best_model = None
    else:
        # Phase 1: Entraînement des couches supérieures
        print("\nDÉBUT DE LA PHASE 1: ENTRAÎNEMENT DES COUCHES SUPÉRIEURES")
        model, phase1_history, phase1_best_model = train_phase(
            model=model,
            phase=1,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            args=args
        )
        
        if args.use_best_model and phase1_best_model:
            print("Utilisation du meilleur modèle de la phase 1")
            model.load_state_dict(phase1_best_model['model_state_dict'])
    
    # Phase 2: Fine-tuning du backbone
    print("\nDÉBUT DE LA PHASE 2: FINE-TUNING DU BACKBONE")
    model, phase2_history, phase2_best_model = train_phase(
        model=model,
        phase=2,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        args=args
    )
    
    # Sauvegarder le modèle final
    if args.use_best_model and phase2_best_model:
        print("Utilisation du meilleur modèle de la phase 2")
        model.load_state_dict(phase2_best_model['model_state_dict'])
    
    # Sauvegarder le modèle final
    final_path = os.path.join(MODEL_SAVE_DIR, "enhanced_patch_2phases_final.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
    }, final_path)
    print(f"Modèle final sauvegardé: {final_path}")
    
    print("\nEntraînement en deux phases terminé.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement en deux phases pour le modèle EnhancedPatchOptimizedModel")
    
    # Paramètres généraux
    parser.add_argument("--cuda", action="store_true", help="Utiliser CUDA si disponible")
    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire pour la reproductibilité")
    
    # Paramètres des patches
    parser.add_argument("--patch_size", type=int, default=96, help="Taille des patches")
    parser.add_argument("--patch_stride", type=int, default=24, help="Pas entre les patches")
    parser.add_argument("--num_patches", type=int, default=9, help="Nombre de patches par image")
    
    # Paramètres d'optimisation
    parser.add_argument("--lr", type=float, default=0.00044, help="Learning rate initial")
    parser.add_argument("--momentum", type=float, default=0.938, help="Momentum pour SGD")
    parser.add_argument("--weight_decay", type=float, default=0.00018, help="Weight decay (L2)")
    
    # Paramètres de phases
    parser.add_argument("--phase1_epochs", type=int, default=30, help="Nombre d'époques pour la phase 1")
    parser.add_argument("--phase2_epochs", type=int, default=50, help="Nombre d'époques pour la phase 2")
    parser.add_argument("--patience", type=int, default=10, help="Patience pour early stopping (phase 1)")
    parser.add_argument("--skip_phase1", action="store_true", help="Sauter la phase 1")
    parser.add_argument("--phase1_checkpoint", type=str, default="", help="Chemin vers le checkpoint de la phase 1")
    parser.add_argument("--use_best_model", action="store_true", help="Utiliser le meilleur modèle au lieu du dernier")
    
    args = parser.parse_args()
    
    # Exécuter la fonction principale
    main(args) 