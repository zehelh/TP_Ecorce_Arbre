"""
Module pour explorer le dataset Bark-101 et générer des statistiques.
"""

import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import collections
from pathlib import Path

# Ajouter le répertoire parent au path pour pouvoir importer les modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import DATA_DIR, TRAIN_FILE, TEST_FILE, FIGURES_DIR
from src.data.dataset import BarkDataset, get_transforms


def analyze_dataset_stats():
    """
    Analyse et affiche les statistiques du dataset Bark-101.
    """
    print("Analyse des statistiques du dataset Bark-101...")
    
    # Création des datasets
    train_transform = get_transforms(is_train=False)  # Utiliser les mêmes transformations pour l'analyse
    train_dataset = BarkDataset(TRAIN_FILE, transform=None, is_train=True)
    test_dataset = BarkDataset(TEST_FILE, transform=None, is_train=False)
    
    print(f"Nombre total d'images d'entraînement: {len(train_dataset)}")
    print(f"Nombre total d'images de test: {len(test_dataset)}")
    print(f"Nombre total de classes: {train_dataset.num_classes}")
    
    # Distribution des classes
    train_class_counts = train_dataset.class_counts
    test_class_counts = test_dataset.class_counts
    
    print("\nRépartition des classes (entraînement):")
    for class_id, count in sorted(train_class_counts.items()):
        print(f"  Classe {class_id}: {count} images")
    
    print("\nRépartition des classes (test):")
    for class_id, count in sorted(test_class_counts.items()):
        print(f"  Classe {class_id}: {count} images")
    
    # Statistiques sur les dimensions des images
    widths = []
    heights = []
    aspect_ratios = []
    
    print("\nAnalyse des dimensions des images...")
    for img_path in train_dataset.images[:100]:  # Limiter à 100 images pour l'analyse
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                widths.append(width)
                heights.append(height)
                aspect_ratios.append(width / height)
        except Exception as e:
            print(f"Erreur lors de l'analyse de {img_path}: {str(e)}")
    
    print(f"Largeur min: {min(widths)}, max: {max(widths)}, moyenne: {np.mean(widths):.2f}, médiane: {np.median(widths)}")
    print(f"Hauteur min: {min(heights)}, max: {max(heights)}, moyenne: {np.mean(heights):.2f}, médiane: {np.median(heights)}")
    print(f"Ratio d'aspect min: {min(aspect_ratios):.2f}, max: {max(aspect_ratios):.2f}, moyenne: {np.mean(aspect_ratios):.2f}")
    
    return {
        "train_count": len(train_dataset),
        "test_count": len(test_dataset),
        "num_classes": train_dataset.num_classes,
        "train_class_counts": train_class_counts,
        "test_class_counts": test_class_counts,
        "widths": widths,
        "heights": heights,
        "aspect_ratios": aspect_ratios
    }


def visualize_class_distribution(train_class_counts, test_class_counts, save_path=None):
    """
    Visualise la distribution des classes dans les ensembles d'entraînement et de test.
    
    Args:
        train_class_counts (dict): Nombre d'images par classe dans l'ensemble d'entraînement
        test_class_counts (dict): Nombre d'images par classe dans l'ensemble de test
        save_path (str, optional): Chemin pour sauvegarder la figure
    """
    plt.figure(figsize=(15, 8))
    
    class_ids = sorted(train_class_counts.keys())
    train_counts = [train_class_counts.get(class_id, 0) for class_id in class_ids]
    test_counts = [test_class_counts.get(class_id, 0) for class_id in class_ids]
    
    x = np.arange(len(class_ids))
    width = 0.35
    
    plt.bar(x - width/2, train_counts, width, label='Entraînement')
    plt.bar(x + width/2, test_counts, width, label='Test')
    
    plt.xlabel('Classe')
    plt.ylabel('Nombre d\'images')
    plt.title('Distribution des classes dans les ensembles d\'entraînement et de test')
    plt.xticks(x, class_ids, rotation=90)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Figure sauvegardée dans {save_path}")
    
    plt.close()


def visualize_image_dimensions(widths, heights, aspect_ratios, save_dir=None):
    """
    Visualise les distributions des dimensions des images.
    
    Args:
        widths (list): Liste des largeurs des images
        heights (list): Liste des hauteurs des images
        aspect_ratios (list): Liste des ratios d'aspect des images
        save_dir (str, optional): Répertoire pour sauvegarder les figures
    """
    # Figure 1: Distribution des largeurs
    plt.figure(figsize=(10, 6))
    plt.hist(widths, bins=20, alpha=0.7, color='blue')
    plt.xlabel('Largeur (pixels)')
    plt.ylabel('Fréquence')
    plt.title('Distribution des largeurs des images')
    plt.grid(True, alpha=0.3)
    
    if save_dir:
        save_path = os.path.join(save_dir, 'widths_distribution.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    plt.close()
    
    # Figure 2: Distribution des hauteurs
    plt.figure(figsize=(10, 6))
    plt.hist(heights, bins=20, alpha=0.7, color='green')
    plt.xlabel('Hauteur (pixels)')
    plt.ylabel('Fréquence')
    plt.title('Distribution des hauteurs des images')
    plt.grid(True, alpha=0.3)
    
    if save_dir:
        save_path = os.path.join(save_dir, 'heights_distribution.png')
        plt.savefig(save_path)
    
    plt.close()
    
    # Figure 3: Distribution des ratios d'aspect
    plt.figure(figsize=(10, 6))
    plt.hist(aspect_ratios, bins=20, alpha=0.7, color='red')
    plt.xlabel('Ratio d\'aspect (largeur/hauteur)')
    plt.ylabel('Fréquence')
    plt.title('Distribution des ratios d\'aspect des images')
    plt.grid(True, alpha=0.3)
    
    if save_dir:
        save_path = os.path.join(save_dir, 'aspect_ratios_distribution.png')
        plt.savefig(save_path)
    
    plt.close()
    
    # Figure 4: Nuage de points largeur vs hauteur
    plt.figure(figsize=(10, 8))
    plt.scatter(widths, heights, alpha=0.5, color='purple')
    plt.xlabel('Largeur (pixels)')
    plt.ylabel('Hauteur (pixels)')
    plt.title('Largeur vs Hauteur des images')
    plt.grid(True, alpha=0.3)
    
    if save_dir:
        save_path = os.path.join(save_dir, 'width_vs_height.png')
        plt.savefig(save_path)
    
    plt.close()


def visualize_sample_images(num_samples=9, save_path=None):
    """
    Visualise des exemples d'images du dataset.
    
    Args:
        num_samples (int): Nombre d'images à visualiser
        save_path (str, optional): Chemin pour sauvegarder la figure
    """
    train_dataset = BarkDataset(TRAIN_FILE, transform=None, is_train=True)
    
    plt.figure(figsize=(12, 12))
    
    # Sélectionner des classes aléatoires
    class_ids = list(train_dataset.class_counts.keys())
    selected_classes = np.random.choice(class_ids, min(num_samples, len(class_ids)), replace=False)
    
    # Collecter les images pour chaque classe sélectionnée
    sample_images = []
    sample_labels = []
    
    for class_id in selected_classes:
        class_indices = [i for i, label in enumerate(train_dataset.labels) if label == class_id]
        if class_indices:
            sample_idx = np.random.choice(class_indices)
            img_path = train_dataset.images[sample_idx]
            try:
                img = Image.open(img_path).convert('RGB')
                sample_images.append(img)
                sample_labels.append(class_id)
            except Exception as e:
                print(f"Erreur lors du chargement de {img_path}: {str(e)}")
    
    # Afficher les images
    for i, (img, label) in enumerate(zip(sample_images, sample_labels)):
        if i >= num_samples:
            break
        
        plt.subplot(3, 3, i + 1)
        plt.imshow(img)
        plt.title(f'Classe {label}')
        plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Figure sauvegardée dans {save_path}")
    
    plt.close()


def main():
    """
    Fonction principale pour l'exploration du dataset.
    """
    # Création du répertoire pour les figures
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # Analyser les statistiques du dataset
    stats = analyze_dataset_stats()
    
    # Visualiser la distribution des classes
    visualize_class_distribution(
        stats['train_class_counts'], 
        stats['test_class_counts'],
        save_path=os.path.join(FIGURES_DIR, 'class_distribution.png')
    )
    
    # Visualiser les dimensions des images
    visualize_image_dimensions(
        stats['widths'], 
        stats['heights'], 
        stats['aspect_ratios'],
        save_dir=FIGURES_DIR
    )
    
    # Visualiser des exemples d'images
    visualize_sample_images(
        num_samples=9,
        save_path=os.path.join(FIGURES_DIR, 'sample_images.png')
    )


if __name__ == "__main__":
    main() 