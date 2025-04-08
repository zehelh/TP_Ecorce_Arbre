"""
Module définissant les classes pour charger et préparer le dataset Bark-101.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms
import random

from ..config import DATA_DIR, TRAIN_FILE, TEST_FILE, BATCH_SIZE, NUM_WORKERS, IMAGE_SIZE


class BarkDataset(Dataset):
    """
    Classe pour charger le dataset Bark-101 à partir des fichiers train.txt et test.txt.
    """
    def __init__(self, file_path, transform=None, is_train=True):
        """
        Initialisation du dataset.
        
        Args:
            file_path (str): Chemin vers le fichier train.txt ou test.txt
            transform (callable, optional): Transformations à appliquer aux images
            is_train (bool): Si True, c'est le dataset d'entraînement, sinon de test
        """
        self.file_path = file_path
        self.transform = transform
        self.is_train = is_train
        self.data_dir = DATA_DIR
        
        # Chargement des chemins d'images et de leurs labels
        self.images, self.labels = self._load_data()
        
        # Statistiques du dataset
        self._calculate_stats()
    
    def _load_data(self):
        """
        Chargement des chemins d'images et de leurs labels à partir du fichier texte.
        
        Returns:
            tuple: (Liste des chemins d'images, liste des labels)
        """
        images = []
        labels = []
        
        with open(self.file_path, 'r') as f:
            for line in f:
                image_path, label = line.strip().split()
                images.append(os.path.join(self.data_dir, image_path))
                labels.append(int(label))
        
        return images, labels
    
    def _calculate_stats(self):
        """
        Calcule et stocke les statistiques du dataset (nombre d'images par classe, etc.).
        """
        self.num_samples = len(self.images)
        self.class_counts = {}
        
        for label in self.labels:
            if label in self.class_counts:
                self.class_counts[label] += 1
            else:
                self.class_counts[label] = 1
        
        self.num_classes = len(self.class_counts)
    
    def __len__(self):
        """
        Renvoie le nombre total d'images dans le dataset.
        """
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Renvoie une image et son label à l'index spécifié.
        
        Args:
            idx (int): Index de l'élément à retourner
            
        Returns:
            tuple: (image, label)
        """
        # Charger l'image
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        # Appliquer les transformations si elles sont définies
        if self.transform:
            image = self.transform(image)
        
        return image, label


class PatchBarkDataset(Dataset):
    """
    Classe pour charger le dataset Bark-101 en utilisant l'approche par patches.
    """
    def __init__(self, file_path, patch_size=64, patch_stride=32, num_patches=9, transform=None, is_train=True):
        """
        Initialisation du dataset basé sur des patches.
        
        Args:
            file_path (str): Chemin vers le fichier train.txt ou test.txt
            patch_size (int): Taille des patches (carrés)
            patch_stride (int): Pas entre les patches
            num_patches (int): Nombre de patches à extraire par image
            transform (callable, optional): Transformations à appliquer aux patches
            is_train (bool): Si True, c'est le dataset d'entraînement, sinon de test
        """
        self.file_path = file_path
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.num_patches = num_patches
        self.transform = transform
        self.is_train = is_train
        self.data_dir = DATA_DIR
        
        # Chargement des chemins d'images et de leurs labels
        self.images, self.labels = self._load_data()
        
        # Statistiques du dataset
        self.num_samples = len(self.images)
    
    def _load_data(self):
        """
        Chargement des chemins d'images et de leurs labels à partir du fichier texte.
        
        Returns:
            tuple: (Liste des chemins d'images, liste des labels)
        """
        images = []
        labels = []
        
        with open(self.file_path, 'r') as f:
            for line in f:
                image_path, label = line.strip().split()
                images.append(os.path.join(self.data_dir, image_path))
                labels.append(int(label))
        
        return images, labels
    
    def _extract_patches(self, image):
        """
        Extrait des patches aléatoires d'une image.
        
        Args:
            image (PIL.Image): Image d'origine
            
        Returns:
            list: Liste des patches extraits
        """
        width, height = image.size
        patches = []
        
        # Si l'image est plus petite que la taille du patch, redimensionner
        if width < self.patch_size or height < self.patch_size:
            image = image.resize((max(width, self.patch_size), max(height, self.patch_size)))
            width, height = image.size
        
        # Calculer les positions valides pour les patches
        valid_h = height - self.patch_size + 1
        valid_w = width - self.patch_size + 1
        
        if self.is_train:
            # En mode entraînement, extraire des patches aléatoires
            for _ in range(self.num_patches):
                h_start = random.randint(0, valid_h - 1)
                w_start = random.randint(0, valid_w - 1)
                patch = image.crop((w_start, h_start, w_start + self.patch_size, h_start + self.patch_size))
                patches.append(patch)
        else:
            # En mode test, extraire des patches réguliers
            h_starts = np.linspace(0, valid_h - 1, min(3, valid_h), dtype=int)
            w_starts = np.linspace(0, valid_w - 1, min(3, valid_w), dtype=int)
            
            for h_start in h_starts:
                for w_start in w_starts:
                    patch = image.crop((w_start, h_start, w_start + self.patch_size, h_start + self.patch_size))
                    patches.append(patch)
            
            # Si on n'a pas assez de patches, dupliquer les existants
            while len(patches) < self.num_patches:
                patches.append(random.choice(patches))
            
            # Si on a trop de patches, n'en garder que le nombre demandé
            if len(patches) > self.num_patches:
                patches = patches[:self.num_patches]
        
        return patches
    
    def __len__(self):
        """
        Renvoie le nombre total d'images dans le dataset.
        """
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Renvoie les patches d'une image et son label à l'index spécifié.
        
        Args:
            idx (int): Index de l'élément à retourner
            
        Returns:
            tuple: (patches, label)
        """
        # Charger l'image
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        # Extraire les patches
        patches = self._extract_patches(image)
        
        # Appliquer les transformations si elles sont définies
        if self.transform:
            patches = [self.transform(patch) for patch in patches]
        
        # Convertir la liste de patches en un tenseur
        patches = torch.stack(patches)
        
        return patches, label


def get_transforms(is_train=True):
    """
    Définit les transformations à appliquer aux images.
    
    Args:
        is_train (bool): Si True, ajoute les augmentations pour l'entraînement
        
    Returns:
        transforms.Compose: Composition des transformations
    """
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE[0]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE[0] + 32, IMAGE_SIZE[1] + 32)),
            transforms.CenterCrop(IMAGE_SIZE[0]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def get_dataloaders(use_patches=False, patch_size=64, patch_stride=32, num_patches=9):
    """
    Crée et retourne les dataloaders pour l'entraînement et l'évaluation.
    
    Args:
        use_patches (bool): Si True, utilise l'approche par patches
        patch_size (int): Taille des patches (si use_patches=True)
        patch_stride (int): Pas entre les patches (si use_patches=True)
        num_patches (int): Nombre de patches par image (si use_patches=True)
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    train_transform = get_transforms(is_train=True)
    test_transform = get_transforms(is_train=False)
    
    if use_patches:
        train_dataset = PatchBarkDataset(
            TRAIN_FILE, 
            patch_size=patch_size,
            patch_stride=patch_stride,
            num_patches=num_patches,
            transform=train_transform, 
            is_train=True
        )
        
        test_dataset = PatchBarkDataset(
            TEST_FILE, 
            patch_size=patch_size,
            patch_stride=patch_stride,
            num_patches=num_patches,
            transform=test_transform, 
            is_train=False
        )
    else:
        train_dataset = BarkDataset(TRAIN_FILE, transform=train_transform, is_train=True)
        test_dataset = BarkDataset(TEST_FILE, transform=test_transform, is_train=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, test_loader 