"""
Utilitaires pour l'entraînement, l'évaluation et la visualisation des modèles.

Ce module fournit des fonctions et classes utilitaires pour:
- Entraîner et tester les modèles
- Évaluer les performances
- Visualiser les prédictions et les cartes d'attention
- Sauvegarder et charger des modèles
- Configurer les callbacks pour le training
"""

from .training import train_model
from .evaluation import evaluate_model, evaluate_and_visualize
from .visualization import visualize_model_attention, visualize_patch_attention
from .callbacks import EarlyStopping, CheckpointSaver
from .metrics import calculate_metrics, get_class_metrics

__all__ = [
    'train_model', 
    'evaluate_model',
    'evaluate_and_visualize',
    'calculate_metrics',
    'get_class_metrics',
    'visualize_model_attention',
    'visualize_patch_attention',
    'EarlyStopping',
    'CheckpointSaver'
] 