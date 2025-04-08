"""
Script pour l'évaluation des modèles de classification d'écorces d'arbres.
"""

import argparse
import torch
import os
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
from src.utils import evaluate_and_visualize
from src.config import PATCH_SIZE, PATCH_STRIDE, NUM_PATCHES


def parse_args():
    """
    Parse les arguments de ligne de commande.
    
    Returns:
        argparse.Namespace: Arguments parsés
    """
    parser = argparse.ArgumentParser(description="Évaluation d'un modèle CNN pour la classification d'écorces d'arbres")
    
    # Arguments du modèle
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Chemin vers le fichier du modèle sauvegardé (.pth)"
    )
    parser.add_argument(
        "--model_type", type=str, default=None, 
        help="Type de modèle (base, advanced, patch). Si non fourni, déduit du nom du fichier."
    )
    
    # Arguments pour l'approche par patches
    parser.add_argument("--patch_size", type=int, default=PATCH_SIZE, help="Taille des patches")
    parser.add_argument("--patch_stride", type=int, default=PATCH_STRIDE, help="Pas entre les patches")
    parser.add_argument("--num_patches", type=int, default=NUM_PATCHES, help="Nombre de patches par image")
    
    # Arguments divers
    parser.add_argument("--cuda", action="store_true", help="Utiliser CUDA si disponible")
    parser.add_argument("--verbose", action="store_true", help="Afficher les messages détaillés")
    
    return parser.parse_args()


def main():
    """
    Fonction principale d'évaluation.
    """
    # Parsing des arguments
    args = parse_args()
    
    # Déterminer l'appareil à utiliser
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Utilisation de l'appareil: {device}")
    
    # Déterminer le type de modèle s'il n'est pas fourni
    if args.model_type is None:
        model_filename = os.path.basename(args.model_path)
        if "base" in model_filename:
            args.model_type = "base"
        elif "advanced" in model_filename:
            args.model_type = "advanced"
        elif "patch" in model_filename:
            args.model_type = "patch"
        else:
            raise ValueError(f"Impossible de déterminer le type de modèle à partir du nom de fichier {model_filename}. Veuillez spécifier --model_type.")
    
    # Créer les dataloaders
    use_patches = args.model_type == "patch"
    _, test_loader = get_dataloaders(
        use_patches=use_patches,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        num_patches=args.num_patches
    )
    
    print(f"Dataset de test: {len(test_loader.dataset)} images")
    
    # Créer le modèle
    model_params = {}
    if use_patches:
        model_params["num_patches"] = args.num_patches
    
    model = get_model(
        model_type=args.model_type,
        **model_params
    )
    
    # Charger les poids du modèle
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Modèle chargé: {args.model_path}")
    
    # Évaluer le modèle
    print(f"Début de l'évaluation...")
    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    
    metrics = evaluate_and_visualize(
        model=model,
        test_loader=test_loader,
        model_name=model_name,
        device=device,
        verbose=args.verbose
    )
    
    print(f"Évaluation terminée.")
    
    # Sauvegarder les résultats
    results_file = os.path.join("results", "logs", f"{model_name}_results.txt")
    with open(results_file, "w") as f:
        f.write(f"model_path: {args.model_path}\n")
        f.write(f"model_type: {args.model_type}\n")
        f.write(f"accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"precision: {metrics['precision']:.4f}\n")
        f.write(f"recall: {metrics['recall']:.4f}\n")
        f.write(f"f1_score: {metrics['f1']:.4f}\n")
        if metrics['loss'] is not None:
            f.write(f"loss: {metrics['loss']:.4f}\n")
        
        f.write("\nClass report:\n")
        for class_id, class_metrics in metrics['class_report'].items():
            if class_id not in ['accuracy', 'macro avg', 'weighted avg']:
                f.write(f"  Class {class_id}: precision={class_metrics['precision']:.4f}, "
                        f"recall={class_metrics['recall']:.4f}, "
                        f"f1-score={class_metrics['f1-score']:.4f}, "
                        f"support={class_metrics['support']}\n")
    
    print(f"Résultats sauvegardés dans {results_file}")


if __name__ == "__main__":
    main() 