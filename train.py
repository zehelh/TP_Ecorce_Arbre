"""
Script principal pour l'entraînement des modèles de classification d'écorces d'arbres.
"""

import os
import argparse
import time
import torch
import numpy as np
import random
import warnings
import logging

# Supprimer les avertissements
warnings.filterwarnings("ignore")

# Configurer le niveau de journalisation
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Tensorflow logs
logging.getLogger("PIL").setLevel(logging.WARNING)  # Pillow logs
logging.getLogger("matplotlib").setLevel(logging.WARNING)  # Matplotlib logs

# Désactiver les messages de debug de PyTorch
torch.set_printoptions(precision=8)
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True

from src.data import get_dataloaders
from src.models import get_model
from src.utils import train_model
from src.config import (
    NUM_EPOCHS, LEARNING_RATE, MOMENTUM, WEIGHT_DECAY, MODEL_TYPES,
    PATCH_SIZE, PATCH_STRIDE, NUM_PATCHES, EARLY_STOPPING_PATIENCE
)


def set_seed(seed):
    """
    Fixe les graines aléatoires pour la reproductibilité.
    
    Args:
        seed (int): Graine aléatoire
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    """
    Parse les arguments de ligne de commande.
    
    Returns:
        argparse.Namespace: Arguments parsés
    """
    parser = argparse.ArgumentParser(description="Entraînement d'un modèle CNN pour la classification d'écorces d'arbres")
    
    # Arguments du modèle
    parser.add_argument(
        "--model", type=str, default="base", choices=MODEL_TYPES,
        help="Type de modèle à entraîner (base, advanced, patch)"
    )
    
    # Arguments d'entraînement
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Nombre d'époques d'entraînement")
    parser.add_argument("--batch_size", type=int, default=32, help="Taille du batch")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Taux d'apprentissage")
    parser.add_argument("--momentum", type=float, default=MOMENTUM, help="Momentum pour l'optimiseur SGD")
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY, help="Regularization L2")
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE, help="Patience pour l'arrêt précoce")
    
    # Arguments pour l'approche par patches
    parser.add_argument("--patch_size", type=int, default=PATCH_SIZE, help="Taille des patches")
    parser.add_argument("--patch_stride", type=int, default=PATCH_STRIDE, help="Pas entre les patches")
    parser.add_argument("--num_patches", type=int, default=NUM_PATCHES, help="Nombre de patches par image")
    
    # Arguments divers
    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire pour la reproductibilité")
    parser.add_argument("--cuda", action="store_true", help="Utiliser CUDA si disponible")
    
    return parser.parse_args()


def main():
    """
    Fonction principale d'entraînement.
    """
    # Parsing des arguments
    args = parse_args()
    
    # Fixer la graine aléatoire
    set_seed(args.seed)
    
    # Déterminer l'appareil à utiliser
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Utilisation de l'appareil: {device}")
    
    # Créer les dataloaders
    use_patches = args.model == "patch"
    train_loader, val_loader = get_dataloaders(
        use_patches=use_patches,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        num_patches=args.num_patches
    )
    
    print(f"Dataset d'entraînement: {len(train_loader.dataset)} images")
    print(f"Dataset de validation: {len(val_loader.dataset)} images")
    
    # Créer le modèle
    model_params = {}
    if args.model == "patch":
        model_params["num_patches"] = args.num_patches
    
    model = get_model(
        model_type=args.model,
        **model_params
    )
    print(f"Modèle créé: {args.model}")
    
    # Entraîner le modèle
    print(f"Début de l'entraînement...")
    start_time = time.time()
    
    model_name = f"{args.model}_model"
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_name=model_name,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=device
    )
    
    training_time = time.time() - start_time
    print(f"Entraînement terminé en {training_time:.2f} secondes")
    
    # Sauvegarder les hyperparamètres utilisés
    with open(os.path.join("results", "logs", f"{model_name}_hyperparams.txt"), "w") as f:
        f.write(f"model: {args.model}\n")
        f.write(f"epochs: {args.epochs}\n")
        f.write(f"batch_size: {args.batch_size}\n")
        f.write(f"lr: {args.lr}\n")
        f.write(f"momentum: {args.momentum}\n")
        f.write(f"weight_decay: {args.weight_decay}\n")
        f.write(f"patience: {args.patience}\n")
        if use_patches:
            f.write(f"patch_size: {args.patch_size}\n")
            f.write(f"patch_stride: {args.patch_stride}\n")
            f.write(f"num_patches: {args.num_patches}\n")
        f.write(f"training_time: {training_time:.2f} seconds\n")
    
    print(f"Hyperparamètres sauvegardés.")


if __name__ == "__main__":
    main() 