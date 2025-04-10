"""
Module de prétraitement pour l'augmentation des classes sous-représentées et l'extraction intelligente des patches.

Ce module:
1. Analyse le déséquilibre des classes dans le jeu de données
2. Applique des techniques avancées d'augmentation pour les classes sous-représentées
3. Extrait des patches informatifs en fonction de la richesse texturale
4. Sauvegarde les images augmentées pour accélérer l'entraînement ultérieur
"""

import os
import sys
import csv
import argparse
import random
import logging
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.color import rgb2gray, rgb2hsv
from skimage.util import img_as_ubyte
from scipy.stats import entropy
from concurrent.futures import ProcessPoolExecutor, as_completed

# Ajout du répertoire parent au path pour permettre les imports depuis src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import DATA_DIR, NUM_CLASSES, IMAGE_SIZE, NUM_PATCHES, PATCH_SIZE
from src.data.dataset import load_dataset_info


# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def analyze_class_distribution(dataset_info, threshold_percentile=25):
    """
    Analyse la distribution des classes dans le jeu de données et identifie les classes sous-représentées.
    
    Args:
        dataset_info (dict): Informations sur le jeu de données
        threshold_percentile (int): Percentile à utiliser comme seuil (les classes sous ce percentile sont considérées sous-représentées)
        
    Returns:
        tuple: (distribution des classes, classes sous-représentées, nombre moyen d'échantillons par classe)
    """
    # Compter les occurrences de chaque classe
    class_counter = Counter()
    for _, label in dataset_info["train"]:
        class_counter[label] += 1
    
    # Calculer les statistiques de base
    class_counts = list(class_counter.values())
    mean_count = sum(class_counts) / len(class_counter)
    median_count = sorted(class_counts)[len(class_counter) // 2]
    threshold_count = np.percentile(class_counts, threshold_percentile)
    
    logger.info(f"Nombre moyen d'images par classe: {mean_count:.2f}")
    logger.info(f"Nombre médian d'images par classe: {median_count}")
    logger.info(f"Seuil de sous-représentation ({threshold_percentile}e percentile): {threshold_count}")
    
    # Identifier les classes sous-représentées (en dessous du seuil)
    underrepresented_classes = {cls: count for cls, count in class_counter.items() if count < threshold_count}
    logger.info(f"Classes sous-représentées identifiées: {len(underrepresented_classes)}")
    
    # Classes avec nombre d'échantillons au-dessus de la moyenne
    well_represented_classes = {cls: count for cls, count in class_counter.items() if count >= mean_count}
    logger.info(f"Classes bien représentées (>= moyenne): {len(well_represented_classes)}")
    
    # Afficher la distribution des classes sous forme d'histogramme
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(class_counter)), sorted(class_counter.values(), reverse=True))
    plt.axhline(y=threshold_count, color='r', linestyle='--', label=f'Seuil ({threshold_percentile}e percentile)')
    plt.axhline(y=mean_count, color='g', linestyle='--', label='Moyenne')
    plt.xlabel('Classes (triées par nombre d\'images)')
    plt.ylabel('Nombre d\'images')
    plt.title('Distribution des classes dans le jeu d\'entraînement')
    plt.legend()
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    
    return class_counter, underrepresented_classes, mean_count


def compute_richness_score(image_path):
    """
    Calcule un score de richesse texturale pour une image.
    
    Args:
        image_path (str): Chemin vers l'image
        
    Returns:
        dict: Scores de richesse texturale pour différentes régions de l'image
    """
    # Charger l'image
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Impossible de charger l'image: {image_path}")
        return None
    
    # Convertir en RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Redimensionner pour accélérer le calcul
    img_resized = cv2.resize(img, (256, 256))
    
    # Convertir en niveaux de gris pour certaines mesures
    gray = rgb2gray(img_resized)
    gray_ubyte = img_as_ubyte(gray)
    
    # Calculer les caractéristiques LBP (Local Binary Pattern)
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    
    # Diviser l'image en régions (grille 4x4)
    h, w = img_resized.shape[:2]
    region_h, region_w = h // 4, w // 4
    
    region_scores = {}
    
    # Pour chaque région, calculer un score de richesse
    for i in range(4):
        for j in range(4):
            # Coordonnées de la région
            y1, y2 = i * region_h, (i + 1) * region_h
            x1, x2 = j * region_w, (j + 1) * region_w
            
            # Extraire la région
            region_gray = gray_ubyte[y1:y2, x1:x2]
            region_lbp = lbp[y1:y2, x1:x2]
            region_color = img_resized[y1:y2, x1:x2]
            
            # Calculer diverses mesures de texture
            # 1. Entropie LBP
            lbp_hist, _ = np.histogram(region_lbp, bins=n_points+2, range=(0, n_points+2), density=True)
            lbp_entropy = entropy(lbp_hist)
            
            # 2. Matrice de co-occurrence des niveaux de gris (GLCM)
            distances = [1, 2]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            glcm = graycomatrix(region_gray, distances, angles, 256, symmetric=True, normed=True)
            contrast = graycoprops(glcm, 'contrast').mean()
            dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
            homogeneity = graycoprops(glcm, 'homogeneity').mean()
            energy = graycoprops(glcm, 'energy').mean()
            correlation = graycoprops(glcm, 'correlation').mean()
            
            # 3. Variabilité des couleurs
            hsv = rgb2hsv(region_color)
            color_std = np.std(hsv[:,:,0]) * 3 + np.std(hsv[:,:,1]) * 2 + np.std(hsv[:,:,2])
            
            # 4. Variance du gradient
            gx = cv2.Sobel(region_gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(region_gray, cv2.CV_32F, 0, 1, ksize=3)
            gradient_mag = np.sqrt(gx**2 + gy**2)
            gradient_var = np.var(gradient_mag)
            
            # Combiner les mesures en un score global
            richness_score = (
                lbp_entropy * 2.0 +
                contrast * 0.5 +
                dissimilarity * 0.5 +
                (1 - homogeneity) * 0.5 +
                (1 - energy) * 0.5 +
                (1 - correlation) * 0.2 +
                color_std * 0.8 +
                gradient_var * 0.3
            )
            
            region_scores[f"{i}_{j}"] = richness_score
    
    return region_scores


def extract_informative_patches(image_path, patch_size=PATCH_SIZE, num_patches=NUM_PATCHES, overlap_ratio=0.25):
    """
    Extrait les patches les plus informatifs d'une image en fonction des scores de richesse.
    
    Args:
        image_path (str): Chemin vers l'image
        patch_size (int): Taille des patches
        num_patches (int): Nombre de patches à extraire
        overlap_ratio (float): Ratio de chevauchement autorisé entre les patches
        
    Returns:
        list: Liste des patches extraits
    """
    # Calculer les scores de richesse
    richness_scores = compute_richness_score(image_path)
    if richness_scores is None:
        return None
    
    # Charger l'image pour l'extraction des patches
    img = Image.open(image_path).convert('RGB')
    img_width, img_height = img.size
    
    # Transformer les scores de régions en grille de positions potentielles
    positions = []
    for pos, score in richness_scores.items():
        i, j = map(int, pos.split('_'))
        # Calculer les coordonnées centrales de la région
        region_h, region_w = img_height // 4, img_width // 4
        center_y = i * region_h + region_h // 2
        center_x = j * region_w + region_w // 2
        
        # Ajuster pour éviter les débordements
        half_patch = patch_size // 2
        center_x = max(half_patch, min(center_x, img_width - half_patch))
        center_y = max(half_patch, min(center_y, img_height - half_patch))
        
        positions.append((center_x, center_y, score))
    
    # Trier les positions par score de richesse (décroissant)
    positions.sort(key=lambda x: x[2], reverse=True)
    
    # Sélectionner les meilleures positions en évitant trop de chevauchement
    selected_positions = []
    for pos_x, pos_y, score in positions:
        # Vérifier le chevauchement avec les positions déjà sélectionnées
        overlap = False
        for sel_x, sel_y, _ in selected_positions:
            # Calculer la distance euclidienne
            dist = np.sqrt((pos_x - sel_x)**2 + (pos_y - sel_y)**2)
            # Chevauchement si la distance est inférieure à (1-overlap_ratio) * taille_patch
            if dist < (1 - overlap_ratio) * patch_size:
                overlap = True
                break
        
        # Si pas trop de chevauchement, ajouter à la sélection
        if not overlap:
            selected_positions.append((pos_x, pos_y, score))
            
            # Si on a assez de positions, arrêter
            if len(selected_positions) >= num_patches:
                break
    
    # Si on n'a pas assez de positions, compléter aléatoirement
    if len(selected_positions) < num_patches:
        remaining = num_patches - len(selected_positions)
        logger.debug(f"Besoin de {remaining} patches supplémentaires aléatoires")
        
        # Créer des positions aléatoires
        for _ in range(remaining * 3):  # On en génère plus pour augmenter les chances d'en trouver sans chevauchement
            pos_x = random.randint(half_patch, img_width - half_patch)
            pos_y = random.randint(half_patch, img_height - half_patch)
            
            # Vérifier le chevauchement
            overlap = False
            for sel_x, sel_y, _ in selected_positions:
                dist = np.sqrt((pos_x - sel_x)**2 + (pos_y - sel_y)**2)
                if dist < (1 - overlap_ratio) * patch_size:
                    overlap = True
                    break
            
            # Si pas trop de chevauchement, ajouter à la sélection
            if not overlap:
                selected_positions.append((pos_x, pos_y, 0))
                
                # Si on a assez de positions, arrêter
                if len(selected_positions) >= num_patches:
                    break
    
    # Extraire les patches aux positions sélectionnées
    patches = []
    for pos_x, pos_y, _ in selected_positions[:num_patches]:
        # Calculer les coordonnées du patch
        left = pos_x - patch_size // 2
        top = pos_y - patch_size // 2
        right = left + patch_size
        bottom = top + patch_size
        
        # Extraire le patch
        patch = img.crop((left, top, right, bottom))
        patches.append(patch)
    
    # S'il manque encore des patches, dupliquer les existants
    while len(patches) < num_patches:
        idx = random.randint(0, len(patches) - 1)
        patches.append(patches[idx])
    
    return patches


def create_advanced_augmentations(image_path, num_augmentations=5):
    """
    Crée des augmentations avancées pour une image donnée.
    
    Args:
        image_path (str): Chemin vers l'image originale
        num_augmentations (int): Nombre d'augmentations à créer
        
    Returns:
        list: Liste des images augmentées
    """
    # Charger l'image
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        logger.error(f"Erreur lors du chargement de {image_path}: {e}")
        return []
    
    # Définir les transformations possibles
    # Transformations de base
    basic_transforms = [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        transforms.RandomAutocontrast(p=0.3),
        transforms.RandomEqualize(p=0.2),
    ]
    
    # Transformations avancées via des fonctions personnalisées
    def apply_random_rotations(img):
        angle = random.uniform(-20, 20)
        return TF.rotate(img, angle)
    
    def apply_random_perspective(img):
        distortion_scale = random.uniform(0.1, 0.3)
        return TF.perspective(img, 
                             startpoints=[(0, 0), (img.width - 1, 0), (img.width - 1, img.height - 1), (0, img.height - 1)],
                             endpoints=[(random.uniform(0, distortion_scale) * img.width, random.uniform(0, distortion_scale) * img.height),
                                       ((1 - random.uniform(0, distortion_scale)) * img.width, random.uniform(0, distortion_scale) * img.height),
                                       ((1 - random.uniform(0, distortion_scale)) * img.width, (1 - random.uniform(0, distortion_scale)) * img.height),
                                       (random.uniform(0, distortion_scale) * img.width, (1 - random.uniform(0, distortion_scale)) * img.height)])
    
    def apply_random_noise(img):
        img_np = np.array(img)
        noise = np.random.normal(0, random.uniform(3, 10), img_np.shape)
        img_noisy = np.clip(img_np + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(img_noisy)
    
    def apply_random_crop_resize(img):
        # Crop aléatoire puis redimensionnement à la taille originale
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            img, scale=(0.7, 0.9), ratio=(0.75, 1.33))
        img = TF.crop(img, i, j, h, w)
        return TF.resize(img, (img.height, img.width))
    
    advanced_transforms = [
        apply_random_rotations,
        apply_random_perspective,
        apply_random_noise,
        apply_random_crop_resize
    ]
    
    # Appliquer des combinaisons de transformations
    augmented_images = []
    for _ in range(num_augmentations):
        # Copie de l'image originale
        aug_img = img.copy()
        
        # Appliquer toutes les transformations de base avec leur probabilité
        for transform in basic_transforms:
            aug_img = transform(aug_img)
        
        # Appliquer une ou plusieurs transformations avancées
        num_advanced = random.randint(1, 2)
        selected_advanced = random.sample(advanced_transforms, num_advanced)
        for transform in selected_advanced:
            aug_img = transform(aug_img)
        
        augmented_images.append(aug_img)
    
    return augmented_images


def augment_class(class_id, dataset_info, target_count=None, max_count=None, mean_count=None, output_dir=None, return_image_paths=False):
    """
    Augmente les images d'une classe sous-représentée.
    
    Args:
        class_id (int): ID de la classe à augmenter
        dataset_info (dict): Informations sur le jeu de données
        target_count (int, optional): Nombre cible minimal d'images après augmentation
        max_count (int, optional): Nombre maximal d'images à générer
        mean_count (float, optional): Nombre moyen d'échantillons par classe dans le dataset
        output_dir (str): Répertoire de sortie pour les images augmentées
        return_image_paths (bool): Si True, retourne les chemins des images augmentées
        
    Returns:
        list or None: Liste des chemins des nouvelles images si return_image_paths=True
    """
    # Trouver toutes les images de cette classe
    class_images = [(img_path, label) for img_path, label in dataset_info["train"] if label == class_id]
    
    if not class_images:
        logger.warning(f"Pas d'images trouvées pour la classe {class_id}")
        return [] if return_image_paths else None
    
    current_count = len(class_images)
    
    # Déterminer le nombre cible d'images
    if mean_count and target_count is None:
        # Si la moyenne est fournie, viser 80% de la moyenne
        target_count = int(mean_count * 0.8)
    
    if target_count is None:
        target_count = 15  # Valeur par défaut
    
    logger.info(f"Classe {class_id}: {current_count} images existantes, cible: {target_count}")
    
    if current_count >= target_count:
        logger.info(f"La classe {class_id} a déjà suffisamment d'images")
        return [] if return_image_paths else None
    
    # Nombre d'augmentations nécessaires
    needed_count = min(target_count - current_count, max_count or float('inf'))
    
    # Créer le répertoire de sortie si nécessaire
    if output_dir:
        class_output_dir = os.path.join(output_dir, f"class_{class_id}")
        os.makedirs(class_output_dir, exist_ok=True)
    
    # Déterminer le nombre d'augmentations par image source
    augs_per_image = max(1, needed_count // len(class_images) + (1 if needed_count % len(class_images) > 0 else 0))
    
    logger.info(f"Classe {class_id}: génération de ~{augs_per_image} augmentations par image source")
    
    new_image_paths = []
    total_augmentations = 0
    
    # Pour chaque image de la classe, créer plusieurs augmentations
    for img_path, _ in class_images:
        # Limiter la génération au nombre nécessaire
        remaining = needed_count - total_augmentations
        if remaining <= 0:
            break
            
        # Nombre d'augmentations pour cette image
        num_augs = min(augs_per_image, remaining)
        
        # Créer les augmentations
        augmented_images = create_advanced_augmentations(img_path, num_augmentations=num_augs)
        
        if not augmented_images:
            continue
        
        # Sauvegarder les images augmentées
        for i, aug_img in enumerate(augmented_images):
            if output_dir:
                # Extraire le nom de fichier d'origine
                orig_filename = os.path.basename(img_path)
                new_filename = f"aug_{i}_{orig_filename}"
                save_path = os.path.join(class_output_dir, new_filename)
                
                aug_img.save(save_path)
                logger.debug(f"Image augmentée sauvegardée: {save_path}")
                
                if return_image_paths:
                    new_image_paths.append((save_path, class_id))
                    
        total_augmentations += len(augmented_images)
    
    logger.info(f"Classe {class_id}: {total_augmentations} nouvelles images générées")
    return new_image_paths if return_image_paths else None


def update_dataset_info(dataset_info, new_images):
    """
    Met à jour les informations du jeu de données avec les nouvelles images augmentées.
    
    Args:
        dataset_info (dict): Informations originales du jeu de données
        new_images (list): Liste des nouvelles images (chemin, classe)
        
    Returns:
        dict: Informations du jeu de données mises à jour
    """
    # Copier les informations originales
    updated_info = {
        "train": dataset_info["train"].copy(),
        "val": dataset_info["val"].copy(),
        "test": dataset_info["test"].copy()
    }
    
    # Ajouter les nouvelles images à l'ensemble d'entraînement
    updated_info["train"].extend(new_images)
    
    return updated_info


def save_updated_dataset_info(dataset_info, output_file):
    """
    Sauvegarde les informations mises à jour du jeu de données dans un fichier CSV.
    
    Args:
        dataset_info (dict): Informations du jeu de données
        output_file (str): Chemin du fichier de sortie
    """
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'label', 'split'])
        
        for split_name, split_data in dataset_info.items():
            for img_path, label in split_data:
                writer.writerow([img_path, label, split_name])
    
    logger.info(f"Informations du jeu de données mises à jour sauvegardées dans {output_file}")


def visualize_augmentations(orig_image_path, augmented_images):
    """
    Visualise les augmentations générées pour une image.
    
    Args:
        orig_image_path (str): Chemin de l'image originale
        augmented_images (list): Liste des images augmentées
    """
    plt.figure(figsize=(12, 6))
    
    # Afficher l'image originale
    orig_img = Image.open(orig_image_path).convert('RGB')
    plt.subplot(2, 3, 1)
    plt.imshow(np.array(orig_img))
    plt.title("Image originale")
    plt.axis('off')
    
    # Afficher les images augmentées
    for i, aug_img in enumerate(augmented_images[:5], 2):
        plt.subplot(2, 3, i)
        plt.imshow(np.array(aug_img))
        plt.title(f"Augmentation {i-1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('augmentation_examples.png')
    plt.close()


def visualize_patches(image_path, patches):
    """
    Visualise les patches extraits d'une image.
    
    Args:
        image_path (str): Chemin de l'image originale
        patches (list): Liste des patches extraits
    """
    plt.figure(figsize=(12, 6))
    
    # Afficher l'image originale
    orig_img = Image.open(image_path).convert('RGB')
    plt.subplot(2, 3, 1)
    plt.imshow(np.array(orig_img))
    plt.title("Image originale")
    plt.axis('off')
    
    # Afficher les patches extraits
    for i, patch in enumerate(patches[:5], 2):
        plt.subplot(2, 3, i)
        plt.imshow(np.array(patch))
        plt.title(f"Patch {i-1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('patch_extraction_example.png')
    plt.close()


def main(args):
    """
    Fonction principale du script de prétraitement.
    
    Args:
        args: Arguments de la ligne de commande
    """
    logger.info("Démarrage du prétraitement des données")
    
    # Charger les informations du jeu de données
    logger.info("Chargement des informations du jeu de données")
    dataset_info = load_dataset_info(DATA_DIR)
    
    # Analyser la distribution des classes
    logger.info("Analyse de la distribution des classes")
    class_distribution, underrepresented_classes, mean_count = analyze_class_distribution(
        dataset_info, 
        threshold_percentile=args.threshold_percentile
    )
    
    # Créer le répertoire pour les images augmentées
    augmented_dir = os.path.join(DATA_DIR, "augmented")
    os.makedirs(augmented_dir, exist_ok=True)
    
    # Exemple de visualisation d'augmentations pour une classe
    if args.visualize:
        # Sélectionner une classe sous-représentée pour visualisation
        if underrepresented_classes:
            example_class = next(iter(underrepresented_classes.keys()))
            class_images = [(img_path, label) for img_path, label in dataset_info["train"] if label == example_class]
            
            if class_images:
                example_image_path = class_images[0][0]
                logger.info(f"Visualisation des augmentations pour {example_image_path}")
                augmented_images = create_advanced_augmentations(example_image_path, num_augmentations=5)
                visualize_augmentations(example_image_path, augmented_images)
                
                # Visualisation des patches extraits
                logger.info(f"Visualisation des patches extraits pour {example_image_path}")
                patches = extract_informative_patches(example_image_path)
                if patches:
                    visualize_patches(example_image_path, patches)
    
    # Augmenter uniquement les classes sous-représentées
    if args.augment:
        logger.info(f"Augmentation des {len(underrepresented_classes)} classes sous-représentées")
        
        all_new_images = []
        
        with ProcessPoolExecutor(max_workers=min(os.cpu_count(), 4)) as executor:
            futures = {}
            for class_id, count in underrepresented_classes.items():
                future = executor.submit(
                    augment_class, 
                    class_id, 
                    dataset_info, 
                    target_count=args.target_count,
                    max_count=args.max_per_class,
                    mean_count=mean_count,
                    output_dir=augmented_dir,
                    return_image_paths=True
                )
                futures[future] = class_id
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Augmentation des classes"):
                class_id = futures[future]
                try:
                    new_images = future.result()
                    if new_images:
                        all_new_images.extend(new_images)
                        logger.info(f"Classe {class_id}: {len(new_images)} nouvelles images générées")
                except Exception as e:
                    logger.error(f"Erreur lors de l'augmentation de la classe {class_id}: {e}")
        
        # Mettre à jour et sauvegarder les informations du jeu de données
        if all_new_images:
            logger.info(f"Total de {len(all_new_images)} nouvelles images générées")
            updated_info = update_dataset_info(dataset_info, all_new_images)
            
            # Générer le fichier train_augmented.txt
            output_file = os.path.join(DATA_DIR, "train_augmented.txt")
            with open(output_file, 'w') as f:
                for img_path, label in updated_info["train"]:
                    # Extraire le chemin relatif à DATA_DIR
                    rel_path = os.path.relpath(img_path, DATA_DIR)
                    f.write(f"{rel_path} {label}\n")
            
            logger.info(f"Fichier train_augmented.txt généré avec {len(updated_info['train'])} images")
            logger.info("Augmentation des données terminée avec succès!")
    
    logger.info("Prétraitement terminé")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prétraitement des données pour le classificateur d'écorces")
    parser.add_argument("--visualize", action="store_true", help="Visualiser des exemples d'augmentation")
    parser.add_argument("--augment", action="store_true", help="Augmenter les classes sous-représentées")
    parser.add_argument("--target_count", type=int, default=None, help="Nombre cible minimal d'images par classe")
    parser.add_argument("--max_per_class", type=int, default=50, help="Nombre maximal d'images générées par classe")
    parser.add_argument("--threshold_percentile", type=int, default=25, 
                        help="Percentile à utiliser comme seuil pour identifier les classes sous-représentées")
    args = parser.parse_args()
    
    main(args) 