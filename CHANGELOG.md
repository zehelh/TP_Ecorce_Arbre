# Journal des modifications

Ce fichier recense toutes les modifications notables apportées au projet.

## [0.1.5] - 2024-07-07

### Ajouté
- Support pour le modèle EnhancedPatchCNN avec mécanisme d'attention multi-têtes
- Mise à jour du README.md avec les instructions pour utiliser le modèle enhanced_patch
- Visualisation spécifique pour les modèles basés sur des patches

## [0.1.4] - 2024-07-06

### Corrigé
- Erreur avec NumPy 2.0 : remplacement de `np.Inf` par `np.inf` dans le module d'early stopping
- Compatibilité assurée avec les versions récentes de NumPy (>= 2.0)

## [0.1.3] - 2024-07-05

### Corrigé
- Erreur lors de l'utilisation de modèles avancés avec l'option `--model advanced --cuda`
- Filtrage des arguments spécifiques à chaque type de modèle (num_patches uniquement pour PatchCNN)
- Restructuration des passages de paramètres pour éviter les erreurs d'arguments inattendus

## [0.1.2] - 2024-07-04

### Modifié
- Ajout de la suppression des avertissements dans les scripts principaux
- Réduction de la verbosité dans les modules d'évaluation et de visualisation
- Ajout d'options verbose=False pour désactiver les barres de progression tqdm
- Configuration des niveaux de journalisation pour PyTorch, PIL et Matplotlib

## [0.1.1] - 2024-07-03

### Modifié
- Mise à jour du README.md avec les commandes correctes pour utiliser main.py et train.py
- Ajout de nouvelles commandes pour la visualisation des modèles
- Clarification des options --cuda pour les commandes d'entraînement et d'évaluation

## [0.1.0] - 2024-07-01

### Ajouté
- Structure initiale du projet
- Fichiers README.md et CHANGELOG.md
- Configuration de base du projet
- Roadmap pour le développement

## À venir

- Exploration du dataset Bark-101
- Implémentation des classes de chargement du dataset
- Création des architectures CNN
- Implémentation de l'approche par patches
- Optimisation des hyperparamètres
- Évaluation et comparaison des modèles
- Visualisation des résultats 