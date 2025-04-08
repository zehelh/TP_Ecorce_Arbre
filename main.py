"""
Script principal pour le projet de classification d'écorces d'arbres.

Ce script centralise les différentes fonctionnalités du projet :
- Exploration du dataset
- Entraînement des modèles
- Évaluation des modèles
"""

import os
import argparse
import torch
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

from src.data.explore import main as explore_main
from train import main as train_main, set_seed
from evaluate import main as evaluate_main
from src.models import get_model
from src.data import get_dataloaders, get_transforms
from src.utils import visualize_model_attention, visualize_patch_attention
from src.config import MODEL_SAVE_DIR, FIGURES_DIR


def parse_args():
    """
    Parse les arguments de ligne de commande.
    
    Returns:
        argparse.Namespace: Arguments parsés
    """
    parser = argparse.ArgumentParser(description="Classification d'écorces d'arbres avec CNN")
    
    parser.add_argument(
        "--action", type=str, required=True,
        choices=["explore", "train", "evaluate", "visualize"],
        help="Action à effectuer"
    )
    
    # Arguments spécifiques à chaque action
    parser.add_argument("--model", type=str, default="base", help="Type de modèle (base, advanced, patch)")
    parser.add_argument("--model_path", type=str, help="Chemin vers le fichier du modèle sauvegardé")
    parser.add_argument("--image_index", type=int, help="Index de l'image à visualiser")
    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire")
    parser.add_argument("--cuda", action="store_true", help="Utiliser CUDA si disponible")
    
    return parser.parse_args()


def visualize_attention(args):
    """
    Visualise l'attention du modèle sur une image du dataset.
    
    Args:
        args (argparse.Namespace): Arguments parsés
    """
    # Vérification des arguments
    if args.model_path is None:
        raise ValueError("L'argument --model_path est requis pour l'action 'visualize'")
    if args.image_index is None:
        raise ValueError("L'argument --image_index est requis pour l'action 'visualize'")
    
    # Déterminer l'appareil à utiliser
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Utilisation de l'appareil: {device}")
    
    # Créer le modèle
    use_patches = "patch" in args.model
    model_params = {}
    if use_patches:
        model_params["num_patches"] = 16  # Valeur par défaut pour la visualisation
    
    model = get_model(model_type=args.model, **model_params)
    
    # Charger les poids du modèle
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Charger le dataset
    transform = get_transforms(is_train=False)
    _, test_loader = get_dataloaders(use_patches=use_patches)
    
    # Récupérer l'image à visualiser
    if args.image_index >= len(test_loader.dataset):
        raise ValueError(f"L'index {args.image_index} est hors limites pour le dataset de test qui contient {len(test_loader.dataset)} images")
    
    image, label = test_loader.dataset[args.image_index]
    
    # Créer le répertoire pour les figures
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # Visualiser l'attention du modèle
    if use_patches:
        # Pour le modèle par patches, visualiser l'attention entre les patches
        save_path = os.path.join(FIGURES_DIR, f"patch_attention_image_{args.image_index}.png")
        visualize_patch_attention(model, image, save_path=save_path)
    else:
        # Pour les modèles CNN classiques, utiliser Grad-CAM
        save_path = os.path.join(FIGURES_DIR, f"gradcam_image_{args.image_index}.png")
        image_tensor = image.to(device)
        visualize_model_attention(model, image_tensor, class_idx=label, save_path=save_path)
    
    print(f"Visualisation sauvegardée dans {save_path}")


def main():
    """
    Fonction principale.
    """
    # Parsing des arguments
    args = parse_args()
    
    # Fixer la graine aléatoire
    set_seed(args.seed)
    
    # Exécuter l'action demandée
    if args.action == "explore":
        print("Exploration du dataset...")
        explore_main()
    elif args.action == "train":
        print("Entraînement du modèle...")
        train_main()
    elif args.action == "evaluate":
        print("Évaluation du modèle...")
        evaluate_main()
    elif args.action == "visualize":
        print("Visualisation de l'attention du modèle...")
        visualize_attention(args)
    else:
        raise ValueError(f"Action inconnue: {args.action}")


if __name__ == "__main__":
    main() 