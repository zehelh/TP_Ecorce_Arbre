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
    parser.add_argument("--all", action="store_true", help="Évaluer tous les modèles dans results/models")
    
    return parser.parse_args()


def load_model_with_fallback(model_path, model_type, model_params, device):
    """
    Charge un modèle avec une gestion des erreurs et tentatives alternatives.
    
    Args:
        model_path (str): Chemin vers le fichier du modèle
        model_type (str): Type de modèle
        model_params (dict): Paramètres du modèle
        device (torch.device): Appareil pour le chargement
        
    Returns:
        torch.nn.Module: Le modèle chargé
    """
    # Essai de chargement avec les paramètres standard
    model = get_model(model_type=model_type, **model_params)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Modèle chargé avec succès: {model_path}")
        return model
    except Exception as e:
        print(f"Erreur lors du chargement du modèle avec la configuration standard: {str(e)}")
        
        # Si c'est un modèle enhanced_patch, essayer avec différentes configurations
        if model_type == "enhanced_patch" and "optimized" in model_path:
            print("Tentative avec des configurations alternatives pour le modèle optimisé...")
            
            # Liste des modèles ResNet à essayer
            resnet_models = ['resnet18', 'resnet34', 'resnet50']
            
            for resnet_model in resnet_models:
                try:
                    # Essayer avec un modèle ResNet différent
                    model_params['resnet_model'] = resnet_model
                    model = get_model(model_type=model_type, **model_params)
                    checkpoint = torch.load(model_path, map_location=device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"Modèle chargé avec succès en utilisant {resnet_model}")
                    return model
                except:
                    print(f"Échec avec {resnet_model}...")
            
            print("Toutes les tentatives ont échoué. Création d'un nouveau modèle à partir des poids...")
            try:
                # Dernier recours : essayer de charger directement les poids sans correspondance exacte
                model = get_model(model_type=model_type, **model_params)
                checkpoint = torch.load(model_path, map_location=device)
                
                # Filtrer les poids qui correspondent
                model_dict = model.state_dict()
                pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() 
                                  if k in model_dict and v.shape == model_dict[k].shape}
                
                # Mettre à jour les poids correspondants
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
                
                print(f"Modèle partiellement chargé avec {len(pretrained_dict)}/{len(model_dict)} couches")
                return model
            except Exception as e:
                print(f"Échec du chargement partiel : {str(e)}")
                raise RuntimeError(f"Impossible de charger le modèle {model_path}. Veuillez vérifier le type de modèle et les paramètres.")
        else:
            # Pour les autres types de modèles, propager l'erreur
            raise RuntimeError(f"Impossible de charger le modèle {model_path}: {str(e)}")


def evaluate_single_model(args, model_path=None):
    """
    Évalue un seul modèle.
    
    Args:
        args (argparse.Namespace): Arguments de ligne de commande
        model_path (str, optional): Chemin du modèle à évaluer (remplace args.model_path)
    
    Returns:
        dict: Métriques d'évaluation ou None en cas d'échec
    """
    # Utiliser le chemin fourni en paramètre si présent
    if model_path:
        args.model_path = model_path
    
    # Déterminer l'appareil à utiliser
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"\nUtilisation de l'appareil: {device}")
    
    # Déterminer le type de modèle s'il n'est pas fourni
    if args.model_type is None:
        model_filename = os.path.basename(args.model_path)
        if "base" in model_filename:
            args.model_type = "base"
        elif "advanced" in model_filename:
            args.model_type = "advanced"
        elif "patch" in model_filename:
            if "enhanced" in model_filename:
                args.model_type = "enhanced_patch"
            else:
                args.model_type = "patch"
        else:
            print(f"Impossible de déterminer le type de modèle à partir du nom de fichier {model_filename}.")
            return None
    
    print(f"Type de modèle: {args.model_type}")
    
    # Créer les dataloaders
    use_patches = args.model_type == "patch" or args.model_type == "enhanced_patch"
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
    
    try:
        # Charger le modèle avec gestion des erreurs
        model = load_model_with_fallback(args.model_path, args.model_type, model_params, device)
        
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
        return metrics
    
    except Exception as e:
        print(f"Erreur lors de l'évaluation du modèle {args.model_path}: {str(e)}")
        return None


def main():
    """
    Fonction principale d'évaluation.
    """
    # Parsing des arguments
    args = parse_args()
    
    # Mode évaluation de tous les modèles
    if args.all:
        print("Mode évaluation de tous les modèles activé")
        models_dir = os.path.join("results", "models")
        
        if not os.path.exists(models_dir):
            print(f"Répertoire {models_dir} introuvable")
            return
        
        # Récupérer tous les fichiers .pth
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
        
        if not model_files:
            print("Aucun modèle trouvé")
            return
        
        print(f"Modèles trouvés: {len(model_files)}")
        
        # Créer un dictionnaire pour stocker les résultats
        results = {}
        
        # Évaluer chaque modèle
        for model_file in model_files:
            model_path = os.path.join(models_dir, model_file)
            print(f"\n{'='*50}")
            print(f"Évaluation de {model_file}...")
            
            # Évaluer le modèle
            metrics = evaluate_single_model(args, model_path)
            
            if metrics:
                results[model_file] = metrics
            
            print(f"{'='*50}")
        
        # Afficher un résumé des résultats
        print("\nRésumé des résultats:")
        print(f"{'Modèle':<40} | {'Accuracy':<10} | {'F1-Score':<10}")
        print(f"{'-'*40}-+-{'-'*10}-+-{'-'*10}")
        
        for model_file, metrics in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
            print(f"{model_file:<40} | {metrics['accuracy']:.4f} | {metrics['f1']:.4f}")
    else:
        # Mode évaluation d'un seul modèle
        evaluate_single_model(args)


if __name__ == "__main__":
    main() 