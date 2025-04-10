"""
Script d'évaluation pour le modèle EnhancedPatchOptimizedModel.
"""

import argparse
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

from src.data import get_dataloaders
from src.config import FIGURES_DIR
from enhanced_patch_optimized_model import EnhancedPatchOptimizedModel


def set_seed(seed=42):
    """Fixe les graines aléatoires pour la reproductibilité."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(description="Évaluation du modèle EnhancedPatchOptimizedModel")
    
    # Arguments du modèle
    parser.add_argument(
        "--model_path", type=str, 
        default="results/models/enhanced_patch_2phases_final.pth",
        help="Chemin vers le fichier du modèle sauvegardé (.pth)"
    )
    
    # Arguments pour les patches
    parser.add_argument("--patch_size", type=int, default=96, help="Taille des patches")
    parser.add_argument("--patch_stride", type=int, default=24, help="Pas entre les patches")
    parser.add_argument("--num_patches", type=int, default=9, help="Nombre de patches par image")
    
    # Arguments divers
    parser.add_argument("--cuda", action="store_true", help="Utiliser CUDA si disponible")
    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire pour la reproductibilité")
    
    return parser.parse_args()


def evaluate_model(model, dataloader, device):
    """Évalue le modèle sur un ensemble de données."""
    model.eval()
    
    all_predictions = []
    all_targets = []
    confidences = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Évaluation"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Prédiction avec confiance
            predicted, conf, _ = model.predict_with_confidence(inputs)
            
            # Stockage des prédictions et des cibles
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            confidences.extend(conf.cpu().numpy())
    
    # Conversion en tableaux numpy
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    confidences = np.array(confidences)
    
    # Calcul des métriques
    accuracy = accuracy_score(all_targets, all_predictions)
    report = classification_report(all_targets, all_predictions, output_dict=True)
    cm = confusion_matrix(all_targets, all_predictions)
    
    return {
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': cm,
        'predictions': all_predictions,
        'targets': all_targets,
        'confidences': confidences
    }


def plot_confusion_matrix(cm, save_path=None):
    """Affiche et sauvegarde la matrice de confusion."""
    plt.figure(figsize=(16, 14))
    
    ax = sns.heatmap(cm, annot=False, cmap='Blues', fmt='g')
    
    # Labels
    ax.set_xlabel('Classe prédite', fontsize=14)
    ax.set_ylabel('Classe réelle', fontsize=14)
    ax.set_title('Matrice de confusion', fontsize=16)
    
    plt.tight_layout()
    
    # Sauvegarde de la figure
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Matrice de confusion sauvegardée dans {save_path}")
    
    plt.close()


def plot_confidence_distribution(confidences, predictions, targets, save_path=None):
    """Affiche la distribution des scores de confiance (correct vs incorrect)."""
    correct = predictions == targets
    
    plt.figure(figsize=(12, 6))
    plt.hist([confidences[correct], confidences[~correct]], bins=20, 
             label=['Prédictions correctes', 'Prédictions incorrectes'], alpha=0.7)
    plt.xlabel('Score de confiance')
    plt.ylabel('Nombre de prédictions')
    plt.title('Distribution des scores de confiance')
    plt.legend()
    plt.grid(alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Distribution des confidences sauvegardée dans {save_path}")
    
    plt.close()


def plot_class_metrics(report, save_path=None):
    """Affiche et sauvegarde les métriques par classe."""
    # Extraire les métriques par classe
    classes = []
    precision = []
    recall = []
    f1 = []
    support = []
    
    for class_id, metrics in report.items():
        # Ignorer les métriques 'accuracy', 'macro avg' et 'weighted avg'
        if class_id not in ['accuracy', 'macro avg', 'weighted avg']:
            classes.append(class_id)
            precision.append(metrics['precision'])
            recall.append(metrics['recall'])
            f1.append(metrics['f1-score'])
            support.append(metrics['support'])
    
    # Trier par f1-score
    indices = np.argsort(f1)
    classes = [classes[i] for i in indices]
    precision = [precision[i] for i in indices]
    recall = [recall[i] for i in indices]
    f1 = [f1[i] for i in indices]
    support = [support[i] for i in indices]
    
    # Créer le graphique
    fig, ax1 = plt.figure(figsize=(15, 10)), plt.gca()
    
    x = np.arange(len(classes))
    width = 0.25
    
    # Barres pour precision, recall et f1
    ax1.bar(x - width, precision, width, label='Précision', color='#5DA5DA', alpha=0.7)
    ax1.bar(x, recall, width, label='Rappel', color='#FAA43A', alpha=0.7)
    ax1.bar(x + width, f1, width, label='F1-score', color='#60BD68', alpha=0.7)
    
    # Configurer l'axe y1
    ax1.set_ylabel('Score', fontsize=14)
    ax1.set_ylim(0, 1.0)
    
    # Créer un second axe y pour le support
    ax2 = ax1.twinx()
    ax2.plot(x, support, 'r-', marker='o', label='Support')
    ax2.set_ylabel('Nombre d\'échantillons', color='r', fontsize=14)
    
    # Configurer l'axe x
    if len(classes) > 20:
        # Si trop de classes, afficher un sous-ensemble
        step = max(1, len(classes) // 10)
        plt.xticks(x[::step], classes[::step], rotation=90)
    else:
        plt.xticks(x, classes, rotation=90)
    
    # Légendes et titres
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title('Métriques par classe', fontsize=16)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Sauvegarde
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Métriques par classe sauvegardées dans {save_path}")
    
    plt.close()


def main(args):
    """Fonction principale."""
    # Fixer la graine aléatoire
    set_seed(args.seed)
    
    # Déterminer l'appareil à utiliser
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Utilisation de l'appareil: {device}")
    
    # Créer les dataloaders
    _, test_loader = get_dataloaders(
        use_patches=True,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        num_patches=args.num_patches
    )
    
    print(f"Dataset de test: {len(test_loader.dataset)} images")
    
    # Créer le modèle optimisé
    model = EnhancedPatchOptimizedModel(num_patches=args.num_patches)
    
    # Charger les poids du modèle
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Modèle chargé avec succès: {args.model_path}")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {str(e)}")
        return
    
    # Déplacer le modèle sur le device
    model = model.to(device)
    
    # Évaluer le modèle
    print("Début de l'évaluation...")
    metrics = evaluate_model(model, test_loader, device)
    
    # Afficher les résultats
    print(f"\nRésultats d'évaluation:")
    print(f"Précision globale: {metrics['accuracy']:.4f}")
    print(f"Précision macro: {metrics['report']['macro avg']['precision']:.4f}")
    print(f"Rappel macro: {metrics['report']['macro avg']['recall']:.4f}")
    print(f"F1-score macro: {metrics['report']['macro avg']['f1-score']:.4f}")
    
    # Sauvegarder les visualisations
    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    
    # Créer le répertoire pour les figures
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # Matrice de confusion
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        save_path=os.path.join(FIGURES_DIR, f"{model_name}_confusion_matrix.png")
    )
    
    # Métriques par classe
    plot_class_metrics(
        metrics['report'],
        save_path=os.path.join(FIGURES_DIR, f"{model_name}_class_metrics.png")
    )
    
    # Distribution des confidences
    plot_confidence_distribution(
        metrics['confidences'],
        metrics['predictions'],
        metrics['targets'],
        save_path=os.path.join(FIGURES_DIR, f"{model_name}_confidence_distribution.png")
    )
    
    print("Évaluation terminée.")


if __name__ == "__main__":
    args = parse_args()
    main(args) 