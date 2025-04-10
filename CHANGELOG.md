# Journal des modifications

Ce fichier recense toutes les modifications notables apportées au projet.

## [0.3.0] - 2024-04-11

### Ajouté
- Fonction `load_dataset_info()` dans dataset.py pour le chargement des données
- Ajout de scikit-image à requirements.txt pour l'analyse de texture
- Implémentation du modèle optimisé avec support pour l'entraînement en 2 phases
- Script `train_2phases.py` pour l'entraînement progressif (transfer learning puis fine-tuning)
- Augmentation ciblée des données pour les classes sous-représentées
- Documentation détaillée sur l'utilisation des différentes fonctionnalités

### Modifié
- Amélioration du processus d'augmentation de données avec analyse de distribution
- Support pour différentes architectures backbone (ResNet18/34/50)
- Mise à jour du README avec instructions pour toutes les fonctionnalités
- Organisation du code pour une meilleure maintenance et réutilisation

### Corrigé
- Correction de l'erreur d'importation dans preprocess_data.py
- Compatibilité avec Ray Tune 2.0+ pour l'optimisation d'hyperparamètres
- Gestion robuste du chargement des modèles et des données

## [0.2.0] - 2024-04-10

### Ajouté
- Optimiseur d'hyperparamètres basé sur Ray Tune
- Script hyperopt.py pour la recherche automatique des meilleurs hyperparamètres
- Support pour la désactivation de l'enregistrement des modèles pendant l'optimisation
- Callback d'estimation de temps pour l'optimisation d'hyperparamètres

### Modifié
- Amélioration de la fonction train_model pour supporter l'optimisation
- Optimisation de la classe EarlyStopping pour le mode hyperopt

### Corrigé
- Gestion de ressources CPU/GPU pour l'optimisation
- Compatibilité avec différents systèmes d'exploitation
- Validation des chemins de fichiers pour éviter les erreurs

## [0.1.0] - 2024-04-09

### Ajouté
- Structure initiale du projet
- Implémentation des modèles de classification (BaseCNN, AdvancedCNN, PatchCNN, EnhancedPatchCNN)
- Scripts d'entraînement, d'évaluation et de visualisation
- Système de visualisation des cartes d'attention pour les modèles à base de patches
- Documentation de base (README, CHANGELOG)

## [0.3.4] - 2024-07-24

### Ajouté
- Nouveau fichier `QUESTIONS.md` avec les réponses détaillées aux questions du TP
- Mise à jour complète du README.md avec une structure améliorée et une documentation plus précise
- Tableau récapitulatif des performances des différents modèles dans le README.md

### Modifié
- Restructuration du README pour une meilleure lisibilité
- Correction et mise à jour des commandes d'utilisation
- Documentation complète de toutes les fonctionnalités du projet

## [0.3.3] - 2024-07-24

### Ajouté
- Nouveau script `evaluate_optimized.py` pour l'évaluation du modèle EnhancedPatchOptimizedModel basé sur ResNet50
- Documentation dans le README pour l'utilisation du script d'évaluation optimisé

### Modifié
- Mise à jour de la section "Évaluation des modèles" dans le README.md

## À venir

- Ajout de nouvelles architectures CNN
- Évaluation comparative approfondie des modèles
- Support pour l'apprentissage semi-supervisé
- Optimisation des performances sur différents matériels
- Interface utilisateur pour la visualisation et l'analyse des résultats 