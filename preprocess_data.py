"""
Script de prétraitement pour augmenter les données des classes sous-représentées dans le dataset Bark-101.

Ce script:
1. Analyse le dataset pour identifier les classes sous-représentées (<15 images)
2. Génère des images augmentées pour ces classes
3. Sauvegarde les images augmentées dans un répertoire spécifique
4. Met à jour les fichiers train.txt et test.txt pour inclure ces nouvelles images
"""

import os
import numpy as np
import random
import argparse
import shutil
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import torchvision.transforms.functional as TF
from tqdm import tqdm
import cv2
import albumentations as A
from pathlib import Path
import matplotlib.pyplot as plt

# Configurer les chemins
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
AUGMENTED_DIR = os.path.join(DATA_DIR, "augmented")
TRAIN_FILE = os.path.join(DATA_DIR, "train.txt")
TEST_FILE = os.path.join(DATA_DIR, "test.txt")
NEW_TRAIN_FILE = os.path.join(DATA_DIR, "train_augmented.txt")
MIN_IMAGES_PER_CLASS = 15  # Seuil minimum d'images par classe


def load_dataset_info():
    """
    Charge les informations du dataset à partir des fichiers train.txt et test.txt.
    
    Returns:
        tuple: (train_images, train_labels, test_images, test_labels, class_counts_train)
    """
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    
    # Charger les données d'entraînement
    with open(TRAIN_FILE, 'r') as f:
        for line in f:
            image_path, label = line.strip().split()
            train_images.append(image_path)
            train_labels.append(int(label))
    
    # Charger les données de test
    with open(TEST_FILE, 'r') as f:
        for line in f:
            image_path, label = line.strip().split()
            test_images.append(image_path)
            test_labels.append(int(label))
    
    # Calculer le nombre d'images par classe dans l'ensemble d'entraînement
    class_counts_train = {}
    for label in train_labels:
        if label in class_counts_train:
            class_counts_train[label] += 1
        else:
            class_counts_train[label] = 1
    
    return train_images, train_labels, test_images, test_labels, class_counts_train


def calculate_texture_richness(image, kernel_size=5):
    """
    Calcule la richesse texturale d'une région d'image.
    
    Args:
        image (numpy.ndarray): Image en format BGR
        kernel_size (int): Taille du kernel pour le calcul de texture
        
    Returns:
        numpy.ndarray: Carte de richesse texturale
    """
    # Convertir en niveaux de gris
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Appliquer un filtre de variance (mesure la variation locale)
    mean, variance = cv2.meanStdDev(gray)
    
    # Calculer l'entropie locale (mesure le désordre local)
    texture_map = np.zeros_like(gray, dtype=np.float32)
    h, w = gray.shape
    
    # Utiliser des filtres de Gabor pour détecter les textures orientées
    kernels = []
    for theta in range(0, 180, 45):  # Orientations: 0°, 45°, 90°, 135°
        theta_rad = theta * np.pi / 180
        kernel = cv2.getGaborKernel((kernel_size, kernel_size), 4.0, theta_rad, 8.0, 0.5, 0, ktype=cv2.CV_32F)
        kernels.append(kernel)
    
    # Appliquer les filtres et combiner les résultats
    for kernel in kernels:
        filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
        texture_map += np.abs(filtered)
    
    # Normaliser
    texture_map = cv2.normalize(texture_map, None, 0, 255, cv2.NORM_MINMAX)
    
    return texture_map


def get_info_based_patches(image, patch_size=128, stride=64, num_patches=16):
    """
    Extrait des patches basés sur la richesse d'information texturale.
    
    Args:
        image (PIL.Image): Image d'entrée
        patch_size (int): Taille des patches
        stride (int): Pas entre les patches
        num_patches (int): Nombre de patches à extraire
        
    Returns:
        list: Liste des patches extraits (PIL.Image)
    """
    # Convertir en numpy pour le traitement
    img_np = np.array(image)
    
    # Calculer la carte de richesse texturale
    texture_map = calculate_texture_richness(img_np, kernel_size=5)
    
    # Extraire tous les patches possibles avec leurs scores
    height, width = texture_map.shape
    patches = []
    scores = []
    
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            # Extraire le patch
            patch = image.crop((x, y, x + patch_size, y + patch_size))
            
            # Calculer le score de texture pour ce patch
            texture_score = np.mean(texture_map[y:y+patch_size, x:x+patch_size])
            
            patches.append(patch)
            scores.append(texture_score)
    
    # Si pas assez de patches, réduire le stride et recommencer
    if len(patches) < num_patches:
        return get_info_based_patches(image, patch_size, stride // 2, num_patches)
    
    # Sélectionner les meilleurs patches selon leur score de texture
    top_indices = np.argsort(scores)[-num_patches:]
    selected_patches = [patches[i] for i in top_indices]
    
    return selected_patches


def create_advanced_augmentations(image, num_augmentations=10):
    """
    Crée des augmentations avancées d'une image pour les classes sous-représentées.
    
    Args:
        image (PIL.Image): Image d'origine
        num_augmentations (int): Nombre d'augmentations à créer
        
    Returns:
        list: Liste des images augmentées
    """
    # Convertir en numpy pour Albumentations
    img_np = np.array(image)
    
    # Créer une transformation forte avec Albumentations
    transform = A.Compose([
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.8),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
        A.GaussNoise(p=0.4),
        A.OneOf([
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=0.2, shift_limit=0.15, p=0.5),
        ], p=0.7),
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.5),
        ], p=0.7),
    ])
    
    # Calculer les patches riches en information
    patch_size = min(image.width, image.height) // 2
    patches = get_info_based_patches(image, patch_size=patch_size, stride=patch_size // 2, num_patches=4)
    
    augmented_images = []
    
    # Créer des augmentations globales
    for i in range(num_augmentations // 2):
        aug_img = transform(image=img_np)['image']
        augmented_images.append(Image.fromarray(aug_img))
    
    # Créer des augmentations basées sur les patches
    for patch in patches:
        # Augmenter chaque patch
        patch_np = np.array(patch)
        aug_patch = transform(image=patch_np)['image']
        
        # Créer une nouvelle image de la taille originale
        new_img = image.copy()
        
        # Remplacer une région aléatoire par le patch augmenté
        x = random.randint(0, max(0, new_img.width - patch_size))
        y = random.randint(0, max(0, new_img.height - patch_size))
        
        # Convertir le patch augmenté en PIL
        aug_patch_pil = Image.fromarray(aug_patch)
        
        # Coller le patch augmenté
        new_img.paste(aug_patch_pil, (x, y))
        
        augmented_images.append(new_img)
    
    # Limiter au nombre demandé
    return augmented_images[:num_augmentations]


def augment_underrepresented_classes(class_counts_train, train_images, train_labels):
    """
    Augmente les données pour les classes sous-représentées.
    
    Args:
        class_counts_train (dict): Nombre d'images par classe
        train_images (list): Chemins des images d'entraînement
        train_labels (list): Labels des images d'entraînement
        
    Returns:
        list: Chemins des nouvelles images augmentées
    """
    # Utiliser la variable globale MIN_IMAGES_PER_CLASS définie dans main()
    global MIN_IMAGES_PER_CLASS
    
    # Créer le répertoire pour les images augmentées s'il n'existe pas
    os.makedirs(AUGMENTED_DIR, exist_ok=True)
    
    # Identifier les classes sous-représentées
    underrepresented_classes = {
        cls: count for cls, count in class_counts_train.items() 
        if count < MIN_IMAGES_PER_CLASS
    }
    
    # Afficher les classes sous-représentées
    print(f"Classes sous-représentées (<{MIN_IMAGES_PER_CLASS} images):")
    for cls, count in sorted(underrepresented_classes.items()):
        print(f"  Classe {cls}: {count} images")
    
    # Chemin des nouvelles images augmentées
    new_image_paths = []
    
    # Pour chaque classe sous-représentée
    for cls, count in tqdm(underrepresented_classes.items(), desc="Augmentation des classes"):
        # Trouver toutes les images de cette classe
        class_image_indices = [i for i, label in enumerate(train_labels) if label == cls]
        class_images = [train_images[i] for i in class_image_indices]
        
        # Créer le répertoire pour cette classe
        class_aug_dir = os.path.join(AUGMENTED_DIR, str(cls))
        os.makedirs(class_aug_dir, exist_ok=True)
        
        # Nombre d'augmentations nécessaires
        num_augmentations_needed = MIN_IMAGES_PER_CLASS - count
        
        # Augmenter les images existantes
        augmentation_count = 0
        
        while augmentation_count < num_augmentations_needed:
            # Sélectionner une image aléatoire de cette classe
            img_path = os.path.join(DATA_DIR, random.choice(class_images))
            try:
                image = Image.open(img_path).convert('RGB')
                
                # Créer des augmentations
                num_aug_per_image = min(5, num_augmentations_needed - augmentation_count)
                augmented_images = create_advanced_augmentations(image, num_aug_per_image)
                
                # Sauvegarder les images augmentées
                for i, aug_img in enumerate(augmented_images):
                    if augmentation_count >= num_augmentations_needed:
                        break
                    
                    aug_img_path = os.path.join(class_aug_dir, f"aug_{Path(img_path).stem}_{i}.jpg")
                    aug_img.save(aug_img_path)
                    
                    # Ajouter le chemin relatif à la liste
                    rel_path = os.path.relpath(aug_img_path, DATA_DIR)
                    new_image_paths.append((rel_path, cls))
                    
                    augmentation_count += 1
                    
            except Exception as e:
                print(f"Erreur lors de l'augmentation de {img_path}: {e}")
                continue
    
    print(f"Augmentation terminée. {len(new_image_paths)} nouvelles images créées.")
    return new_image_paths


def update_train_file(new_image_paths):
    """
    Met à jour le fichier train.txt avec les nouvelles images augmentées.
    
    Args:
        new_image_paths (list): Liste des tuples (chemin, label) des nouvelles images
    """
    # Copier le fichier train.txt original
    shutil.copy(TRAIN_FILE, TRAIN_FILE + '.backup')
    
    # Ajouter les nouvelles images au fichier train.txt
    with open(NEW_TRAIN_FILE, 'w') as f_new:
        # Copier le contenu original
        with open(TRAIN_FILE, 'r') as f_orig:
            f_new.write(f_orig.read())
        
        # Ajouter les nouvelles images
        for img_path, label in new_image_paths:
            f_new.write(f"{img_path} {label}\n")
    
    print(f"Fichier d'entraînement mis à jour: {NEW_TRAIN_FILE}")


def visualize_augmentations(original_path, aug_dir, n_samples=3):
    """
    Visualise les augmentations pour validation.
    
    Args:
        original_path (str): Chemin d'une image originale
        aug_dir (str): Répertoire des images augmentées
        n_samples (int): Nombre d'échantillons à visualiser
    """
    # Charger l'image originale
    original = Image.open(os.path.join(DATA_DIR, original_path)).convert('RGB')
    
    # Trouver quelques augmentations
    base_name = Path(original_path).stem
    aug_files = [f for f in os.listdir(aug_dir) if f.startswith(f"aug_{base_name}")]
    
    if not aug_files:
        print(f"Aucune augmentation trouvée pour {original_path}")
        return
    
    # Limiter le nombre d'échantillons
    aug_files = aug_files[:n_samples]
    
    # Afficher l'original et les augmentations
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, n_samples + 1, 1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis('off')
    
    for i, aug_file in enumerate(aug_files):
        aug_img = Image.open(os.path.join(aug_dir, aug_file)).convert('RGB')
        plt.subplot(1, n_samples + 1, i + 2)
        plt.imshow(aug_img)
        plt.title(f"Aug {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(AUGMENTED_DIR, "augmentation_samples.png"))
    plt.close()
    print(f"Visualisation sauvegardée dans {os.path.join(AUGMENTED_DIR, 'augmentation_samples.png')}")


def main():
    """
    Fonction principale pour le prétraitement des données.
    """
    # Utiliser la variable globale
    global MIN_IMAGES_PER_CLASS
    
    # Parser les arguments de ligne de commande
    parser = argparse.ArgumentParser(description="Prétraitement des données pour le dataset Bark-101")
    parser.add_argument("--visualize", action="store_true", help="Visualiser les augmentations")
    parser.add_argument("--augment", action="store_true", help="Augmenter les classes sous-représentées")
    parser.add_argument("--target_count", type=int, default=MIN_IMAGES_PER_CLASS, 
                        help=f"Nombre cible d'images par classe (défaut: {MIN_IMAGES_PER_CLASS})")
    args = parser.parse_args()
    
    # Mettre à jour la valeur en fonction de l'argument
    MIN_IMAGES_PER_CLASS = args.target_count
    
    # Charger les informations du dataset
    train_images, train_labels, test_images, test_labels, class_counts_train = load_dataset_info()
    
    print(f"Dataset original:")
    print(f"  Images d'entraînement: {len(train_images)}")
    print(f"  Images de test: {len(test_images)}")
    print(f"  Nombre de classes: {len(class_counts_train)}")
    
    if args.augment:
        # Augmenter les classes sous-représentées
        new_image_paths = augment_underrepresented_classes(class_counts_train, train_images, train_labels)
        
        # Mettre à jour le fichier train.txt
        update_train_file(new_image_paths)
        
        # Visualiser quelques augmentations si demandé
        if args.visualize and new_image_paths:
            cls = new_image_paths[0][1]  # Label de la première image augmentée
            original_path = next(img for img, label in zip(train_images, train_labels) if label == cls)
            aug_dir = os.path.join(AUGMENTED_DIR, str(cls))
            visualize_augmentations(original_path, aug_dir)
    elif args.visualize:
        # Juste visualiser une augmentation sans modifier les fichiers
        
        # Pour la visualisation, nous devons identifier les classes sous-représentées
        underrepresented_classes = {
            cls: count for cls, count in class_counts_train.items() 
            if count < MIN_IMAGES_PER_CLASS
        }
        
        print(f"Classes sous-représentées (<{MIN_IMAGES_PER_CLASS} images):")
        for cls, count in sorted(underrepresented_classes.items()):
            print(f"  Classe {cls}: {count} images")
        
        # Choisir une classe sous-représentée au hasard
        if underrepresented_classes:
            cls = random.choice(list(underrepresented_classes.keys()))
            class_images = [img for img, label in zip(train_images, train_labels) if label == cls]
            if class_images:
                # Créer temporairement des augmentations pour visualisation
                temp_dir = os.path.join(AUGMENTED_DIR, "temp")
                os.makedirs(temp_dir, exist_ok=True)
                
                image = Image.open(os.path.join(DATA_DIR, class_images[0])).convert('RGB')
                augmented_images = create_advanced_augmentations(image, 3)
                
                for i, aug_img in enumerate(augmented_images):
                    aug_img_path = os.path.join(temp_dir, f"aug_temp_{i}.jpg")
                    aug_img.save(aug_img_path)
                
                visualize_augmentations(class_images[0], temp_dir)
                
                # Nettoyer
                shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main() 