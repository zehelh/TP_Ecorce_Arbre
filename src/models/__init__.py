"""
Initialisation du module models.
"""

from .base_model import BaseCNN
from .advanced_model import AdvancedCNN
from .patch_model import PatchCNN
from .enhanced_patch_model_ray_tune import EnhancedPatchCNN

__all__ = ["BaseCNN", "AdvancedCNN", "PatchCNN", "EnhancedPatchCNN"]


def get_model(model_type, num_classes=None, **kwargs):
    """
    Retourne une instance du modèle spécifié.
    
    Args:
        model_type (str): Type de modèle ("base", "advanced", "patch", "enhanced_patch")
        num_classes (int, optional): Nombre de classes
        **kwargs: Arguments supplémentaires pour le modèle
        
    Returns:
        nn.Module: Instance du modèle
        
    Raises:
        ValueError: Si le type de modèle n'est pas reconnu
    """
    if model_type == "base":
        model = BaseCNN(num_classes=num_classes) if num_classes else BaseCNN()
    elif model_type == "advanced":
        # Filtrer les arguments qui ne sont pas utilisés par AdvancedCNN
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'num_patches'}
        model = AdvancedCNN(num_classes=num_classes, **filtered_kwargs) if num_classes else AdvancedCNN(**filtered_kwargs)
    elif model_type == "patch":
        model = PatchCNN(num_classes=num_classes, **kwargs) if num_classes else PatchCNN(**kwargs)
    elif model_type == "enhanced_patch":
        model = EnhancedPatchCNN(num_classes=num_classes, **kwargs) if num_classes else EnhancedPatchCNN(**kwargs)
    else:
        raise ValueError(f"Type de modèle inconnu: {model_type}")
    
    return model 