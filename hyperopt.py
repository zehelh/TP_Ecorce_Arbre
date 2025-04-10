"""
Script pour l'optimisation d'hyperparamètres du modèle enhanced_patch utilisant Ray Tune.

Ce script:
1. Définit l'espace de recherche des hyperparamètres
2. Utilise Ray Tune pour trouver la meilleure configuration
3. Entraîne le modèle final avec les meilleurs hyperparamètres
"""

import os
import time
import argparse
import torch
import numpy as np
import json
from functools import partial
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from datetime import datetime, timedelta
from ray.tune import Callback

from src.models import get_model
from src.data import get_dataloaders
from src.utils import train_model
from src.config import NUM_PATCHES, MODEL_SAVE_DIR, LOG_DIR


def set_seed(seed):
    """
    Fixe les graines aléatoires pour la reproductibilité.
    
    Args:
        seed (int): Graine aléatoire
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TimeEstimationCallback(Callback):
    """Callback pour l'estimation du temps d'exécution."""
    
    def __init__(self, total_trials, epochs_per_trial):
        self.total_trials = total_trials
        self.epochs_per_trial = epochs_per_trial
        self.start_time = time.time()
        self.last_print_time = time.time()
        self.print_interval = 30  # secondes entre les mises à jour
    
    def on_trial_result(self, iteration, trials, trial, result, **info):
        """Appelé à chaque nouveau résultat d'un essai."""
        current_time = time.time()
        if current_time - self.last_print_time >= self.print_interval:
            self.last_print_time = current_time
            self._print_time_estimate(trials)
    
    def on_trial_complete(self, iteration, trials, trial, **info):
        """Appelé lorsqu'un essai est terminé."""
        self._print_time_estimate(trials)
    
    def _print_time_estimate(self, trials):
        """Affiche l'estimation du temps restant."""
        # Calculer les statistiques des essais
        running_trials = [t for t in trials if t.status == "RUNNING"]
        completed_trials = [t for t in trials if t.status == "TERMINATED"]
        
        # Calculer le temps écoulé
        elapsed_time = time.time() - self.start_time
        elapsed_str = str(timedelta(seconds=int(elapsed_time)))
        
        # Calculer le temps estimé restant
        total_iters = 0
        max_iter_running = 0
        
        for t in running_trials + completed_trials:
            if hasattr(t, "last_result") and t.last_result:
                iters = t.last_result.get("training_iteration", 0)
                total_iters += iters
                if t.status == "RUNNING" and iters > max_iter_running:
                    max_iter_running = iters
        
        # Estimer le temps restant
        if total_iters > 0:
            time_per_iter = elapsed_time / total_iters
            remaining_iters = (self.total_trials * self.epochs_per_trial) - total_iters
            remaining_seconds = remaining_iters * time_per_iter
            remaining_str = str(timedelta(seconds=int(remaining_seconds)))
            
            # Estimer le temps pour l'essai actuel
            if running_trials and max_iter_running > 0:
                remaining_iters_current = self.epochs_per_trial - max_iter_running
                current_remaining = remaining_iters_current * time_per_iter
                current_str = str(timedelta(seconds=int(current_remaining)))
            else:
                current_str = "inconnu"
            
            # Estimer la date de fin
            end_time = datetime.now() + timedelta(seconds=int(remaining_seconds))
            
            # Afficher les estimations
            print("\n--- ESTIMATION DE TEMPS ---")
            print(f"Temps écoulé: {elapsed_str}")
            print(f"Essais: {len(completed_trials)} terminés, {len(running_trials)} en cours, {self.total_trials - len(completed_trials) - len(running_trials)} en attente")
            print(f"Temps estimé pour l'essai en cours: {current_str}")
            print(f"Temps total restant estimé: {remaining_str}")
            print(f"Date de fin estimée: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("-------------------------\n")
        else:
            print("\n--- ESTIMATION DE TEMPS ---")
            print(f"Temps écoulé: {elapsed_str}")
            print("Calcul du temps restant en cours...")
            print("-------------------------\n")


def train_enhanced_patch(config, checkpoint_dir=None, data_dir=None, num_epochs=10, use_gpu=True):
    """
    Fonction d'entraînement pour Ray Tune.
    
    Args:
        config (dict): Configuration des hyperparamètres
        checkpoint_dir (str): Répertoire pour les points de contrôle
        data_dir (str): Répertoire des données
        num_epochs (int): Nombre maximal d'époques
        use_gpu (bool): Utiliser GPU si disponible
    """
    # Déterminer l'appareil
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    
    # Créer les dataloaders avec num_workers=0 pour éviter les problèmes de multiprocessing sous Windows
    train_loader, val_loader = get_dataloaders(
        use_patches=True,
        patch_size=config["patch_size"],
        patch_stride=config["patch_stride"],
        num_patches=config["num_patches"],
        num_workers=0  # Éviter le multiprocessing sous Windows
    )
    
    # Créer le modèle
    model_params = {
        "num_patches": config["num_patches"],
        "resnet_model": config["backbone"],
        "pretrained": True,
        "use_context": config["use_context"]
    }
    
    model = get_model(model_type="enhanced_patch", **model_params)
    model = model.to(device)
    
    # Reprendre l'entraînement si checkpoint_dir est fourni
    if checkpoint_dir:
        try:
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            print(f"Reprise de l'entraînement depuis {checkpoint_path}")
        except Exception as e:
            print(f"Erreur lors du chargement du checkpoint: {e}")
            print("Démarrage d'un nouvel entraînement...")
    
    # Entraîner le modèle avec les hyperparamètres spécifiés
    model_name = "enhanced_patch_tune"
    
    # Utiliser une patience plus courte pour l'optimisation
    patience = config.get("patience", 5)
    
    # Désactiver la sauvegarde du modèle pendant l'optimisation
    save_best = False
    
    # Entraîner le modèle
    _, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_name=model_name,
        num_epochs=num_epochs,
        learning_rate=config["lr"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
        patience=patience,
        device=device,
        save_best=save_best,
        verbose=False
    )
    
    # Récupérer la meilleure précision de validation
    best_val_acc = max(history["val_acc"])
    best_epoch = history["val_acc"].index(best_val_acc)
    best_val_loss = history["val_loss"][best_epoch]
    
    # Rapport pour Ray Tune - au lieu d'utiliser des arguments nommés, utiliser un dictionnaire
    # pour compatibilité avec les versions récentes de Ray Tune
    tune.report({
        "val_acc": best_val_acc,
        "val_loss": best_val_loss,
        "training_iteration": best_epoch + 1
    })


def trial_dirname_creator(trial):
    """
    Crée un nom de répertoire court pour un essai Ray Tune.
    
    Args:
        trial (Trial): L'essai pour lequel créer un nom de répertoire
        
    Returns:
        str: Nom de répertoire court
    """
    return f"trial_{trial.trial_id}"


def main(args):
    """
    Fonction principale pour l'optimisation des hyperparamètres.
    """
    ray.init(num_cpus=args.num_cpus, num_gpus=args.num_gpus)
    
    # Configuration de l'espace de recherche
    config = {
        "lr": tune.loguniform(1e-4, 1e-2),
        "momentum": tune.uniform(0.8, 0.99),
        "weight_decay": tune.loguniform(1e-5, 1e-3),
        "backbone": tune.choice(["resnet18", "resnet34", "resnet50"]),
        "use_context": tune.choice([True, False]),
        "num_patches": tune.choice([9, 16, 25]),
        "patch_size": tune.choice([48, 64, 96]),
        "patch_stride": tune.choice([24, 32, 48]),
        "patience": 5
    }
    
    # Planificateur ASHA pour l'arrêt précoce des essais peu prometteurs
    grace_period = min(5, max(1, args.epochs // 3))  # Au moins 1, au maximum 5, ou 1/3 du nombre d'époques
    scheduler = ASHAScheduler(
        metric="val_acc",
        mode="max",
        max_t=args.epochs,
        grace_period=grace_period,
        reduction_factor=2
    )
    
    # Reporter standard pour les résultats
    reporter = CLIReporter(
        metric_columns=["val_loss", "val_acc", "training_iteration"],
        parameter_columns=["lr", "momentum", "backbone", "num_patches"]
    )
    
    # Callback pour l'estimation du temps
    time_callback = TimeEstimationCallback(
        total_trials=args.num_trials,
        epochs_per_trial=args.epochs
    )
    
    # Exécuter l'optimisation
    storage_path = os.path.abspath(os.path.join(os.getcwd(), "results", "hyperopt"))
    result = tune.run(
        partial(
            train_enhanced_patch,
            num_epochs=args.epochs,
            use_gpu=args.cuda
        ),
        resources_per_trial={"cpu": args.cpu_per_trial, "gpu": 1 if args.cuda else 0},
        config=config,
        num_samples=args.num_trials,
        scheduler=scheduler,
        progress_reporter=reporter,
        callbacks=[time_callback],  # Ajouter notre callback d'estimation de temps
        name="enhanced_patch_hyperopt",
        storage_path=storage_path,
        trial_dirname_creator=trial_dirname_creator  # Utiliser des noms courts pour les répertoires
    )
    
    # Obtenir la meilleure configuration
    try:
        best_trial = result.get_best_trial("val_acc", "max", "last")
        if best_trial:
            best_config = best_trial.config
            best_val_acc = best_trial.last_result["val_acc"]
            
            print(f"Meilleure configuration trouvée: {best_config}")
            print(f"Meilleure précision de validation: {best_val_acc:.4f}")
            
            # Enregistrer la meilleure configuration
            os.makedirs(LOG_DIR, exist_ok=True)
            with open(os.path.join(LOG_DIR, "enhanced_patch_best_config.json"), "w") as f:
                json.dump(best_config, f, indent=4)
            
            # Entraîner le modèle final avec la meilleure configuration
            if args.train_final:
                train_final_model(best_config, args)
        else:
            print("Aucun essai n'a produit de résultats valides. Essayez de modifier les paramètres d'optimisation.")
    except Exception as e:
        print(f"Erreur lors de la récupération du meilleur essai: {e}")
        print("Aucun modèle final ne sera entraîné.")


def train_final_model(best_config, args):
    """
    Entraîne le modèle final avec les meilleurs hyperparamètres.
    
    Args:
        best_config (dict): Configuration optimale des hyperparamètres
        args (argparse.Namespace): Arguments de ligne de commande
    """
    print("\nEntraînement du modèle final avec la meilleure configuration...")
    
    # Déterminer l'appareil
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    
    # Créer les dataloaders avec num_workers=0 pour éviter les problèmes de multiprocessing sous Windows
    train_loader, val_loader = get_dataloaders(
        use_patches=True,
        patch_size=best_config["patch_size"],
        patch_stride=best_config["patch_stride"],
        num_patches=best_config["num_patches"],
        num_workers=0  # Éviter le multiprocessing sous Windows
    )
    
    # Créer le modèle
    model_params = {
        "num_patches": best_config["num_patches"],
        "resnet_model": best_config["backbone"],
        "pretrained": True,
        "use_context": best_config["use_context"]
    }
    
    model = get_model(model_type="enhanced_patch", **model_params)
    model = model.to(device)
    
    # Entraîner le modèle final
    model_name = "enhanced_patch_optimized"
    start_time = time.time()
    
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_name=model_name,
        num_epochs=args.final_epochs,
        learning_rate=best_config["lr"],
        momentum=best_config["momentum"],
        weight_decay=best_config["weight_decay"],
        patience=args.patience,
        device=device
    )
    
    training_time = time.time() - start_time
    print(f"Entraînement terminé en {training_time:.2f} secondes")
    
    # Sauvegarder les hyperparamètres utilisés
    with open(os.path.join(LOG_DIR, f"{model_name}_hyperparams.txt"), "w") as f:
        f.write(f"model: enhanced_patch_optimized\n")
        f.write(f"lr: {best_config['lr']}\n")
        f.write(f"momentum: {best_config['momentum']}\n")
        f.write(f"weight_decay: {best_config['weight_decay']}\n")
        f.write(f"backbone: {best_config['backbone']}\n")
        f.write(f"use_context: {best_config['use_context']}\n")
        f.write(f"num_patches: {best_config['num_patches']}\n")
        f.write(f"patch_size: {best_config['patch_size']}\n")
        f.write(f"patch_stride: {best_config['patch_stride']}\n")
        f.write(f"epochs: {args.final_epochs}\n")
        f.write(f"patience: {args.patience}\n")
        f.write(f"training_time: {training_time:.2f} seconds\n")
    
    print(f"Hyperparamètres sauvegardés.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimisation d'hyperparamètres pour le modèle enhanced_patch")
    
    # Paramètres d'optimisation
    parser.add_argument("--num_trials", type=int, default=20, help="Nombre d'essais d'hyperparamètres")
    parser.add_argument("--epochs", type=int, default=20, help="Nombre d'époques pour chaque essai")
    parser.add_argument("--train_final", action="store_true", help="Entraîner le modèle final avec la meilleure configuration")
    parser.add_argument("--final_epochs", type=int, default=100, help="Nombre d'époques pour l'entraînement final")
    parser.add_argument("--patience", type=int, default=15, help="Patience pour l'arrêt précoce de l'entraînement final")
    
    # Paramètres système
    parser.add_argument("--num_cpus", type=int, default=4, help="Nombre total de CPUs à utiliser")
    parser.add_argument("--num_gpus", type=int, default=1, help="Nombre total de GPUs à utiliser")
    parser.add_argument("--cpu_per_trial", type=int, default=4, help="Nombre de CPUs par essai")
    parser.add_argument("--cuda", action="store_true", help="Utiliser CUDA si disponible")
    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire")
    
    args = parser.parse_args()
    
    # Fixer la graine aléatoire
    set_seed(args.seed)
    
    main(args) 