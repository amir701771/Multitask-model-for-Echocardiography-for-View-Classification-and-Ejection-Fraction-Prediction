import sys
import os
import logging
import argparse

import importlib
from multiprocessing import freeze_support
import json
import time
import random
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except ImportError:
    print("ERROR: DEAP library not found. Please install it: pip install deap")
    DEAP_AVAILABLE = False

from src.utils.helpers import load_config, setup_logging, set_seed, load_checkpoint
from src.models.cnn_multitask import MultiTaskCNN
from src.data_handling.dataset_echonet import EchoNetViewDataset as EchonetPseudoDataset, video_collate_fn
from src.training.losses import CombinedLoss
from src.utils.metrics import compute_metrics_multitask
from src.utils.visualize import plot_confusion_matrix, plot_ef_scatter

def run_evaluation(model, loader, device, config, split_name="EA_Eval", custom_alpha=None, custom_beta=None):
    """
    Runs evaluation on a given dataset loader.
    Can optionally calculate loss using provided alpha/beta.
    """
    model.eval()
    all_view_logits, all_ef_preds, all_view_labels, all_ef_labels = [], [], [], []
    processed_samples = 0
    running_loss, running_loss_c, running_loss_r = 0.0, 0.0, 0.0
    eval_criterion = None
    train_cfg = config.get('training', {})
    alpha = custom_alpha if custom_alpha is not None else train_cfg.get('loss_alpha', 0.5)
    beta = custom_beta if custom_beta is not None else train_cfg.get('loss_beta', 0.5)
    calculate_loss = True

    try:
        eval_criterion = CombinedLoss(alpha=alpha, beta=beta).to(device)
        logging.debug(f"Initialized CombinedLoss for eval (alpha={alpha:.3f}, beta={beta:.3f}).")
    except Exception as e:
        logging.error(f"Failed CombinedLoss init for eval: {e}. Loss won't be calculated.", exc_info=True)
        eval_criterion = None; calculate_loss = False

    eval_loop = tqdm(loader, desc=f"Evaluating {split_name}", leave=False, disable=(split_name=="EA_Eval"))
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loop):
            if not isinstance(batch, (list, tuple)) or len(batch) != 3: continue
            try: inputs, view_labels, ef_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            except Exception as e: logging.error(f"Eval Batch {batch_idx}: Error moving batch to device: {e}"); continue
            try:
                view_logits, ef_preds = model(inputs)
                if calculate_loss and eval_criterion:
                    try:
                        loss, loss_c, loss_r = eval_criterion(view_logits, ef_preds, view_labels, ef_labels)
                        batch_size = inputs.size(0)
                        running_loss += loss.item() * batch_size; running_loss_c += loss_c.item() * batch_size; running_loss_r += loss_r.item() * batch_size
                        if not (split_name=="EA_Eval"): eval_loop.set_postfix(loss=f"{loss.item():.4f}")
                    except Exception as loss_e: logging.warning(f"Eval Batch {batch_idx}: Could not calculate loss: {loss_e}"); loss = torch.tensor(float('nan'))
                all_view_logits.append(view_logits.cpu()); all_ef_preds.append(ef_preds.cpu())
                all_view_labels.append(view_labels.cpu()); all_ef_labels.append(ef_labels.cpu())
                processed_samples += inputs.size(0)
            except Exception as e: logging.error(f"Eval Batch {batch_idx}: Error during model forward pass: {e}.", exc_info=True); continue

    if processed_samples == 0: logging.warning(f"{split_name} eval skipped: No samples processed."); return {'metrics': {}}
    final_loss = running_loss / processed_samples if calculate_loss and processed_samples > 0 else float('nan')
    final_loss_c = running_loss_c / processed_samples if calculate_loss and processed_samples > 0 else float('nan')
    final_loss_r = running_loss_r / processed_samples if calculate_loss and processed_samples > 0 else float('nan')

    eval_metrics = {}; final_view_logits=None; final_ef_preds=None; final_view_labels=None; final_ef_labels=None
    try:
        final_view_logits = torch.cat(all_view_logits, dim=0) if all_view_logits else None; final_ef_preds = torch.cat(all_ef_preds, dim=0) if all_ef_preds else None
        final_view_labels = torch.cat(all_view_labels, dim=0) if all_view_labels else None; final_ef_labels = torch.cat(all_ef_labels, dim=0) if all_ef_labels else None
        eval_metrics = compute_metrics_multitask(final_view_logits, final_ef_preds, final_view_labels, final_ef_labels, config=config.get('data', {}))
    except Exception as e: logging.error(f"Error computing metrics for {split_name}: {e}", exc_info=True); eval_metrics = {}

    if split_name != "EA_Eval":
        logging.info(f"--- {split_name} Set Evaluation Results ---")
        if not np.isnan(final_loss): logging.info(f"{split_name} Loss: {final_loss:.4f} (C:{final_loss_c:.4f}, R:{final_loss_r:.4f})")
        if eval_metrics: logging.info(" | ".join([f"{split_name} {k.replace('_',' ').title()}: {v:.4f}" for k, v in eval_metrics.items()]))
        else: logging.warning(f"No metrics computed for {split_name} set.")

    results = { f'{split_name.lower()}_loss': final_loss, f'{split_name.lower()}_loss_classification': final_loss_c,
                f'{split_name.lower()}_loss_regression': final_loss_r,
                'metrics': {k: float(v) if isinstance(v, (np.floating, np.integer, np.number, torch.Tensor)) else v for k,v in eval_metrics.items()}, }

    if split_name != "EA_Eval":
        output_dir = config['training']['output_dir']; plots_dir = os.path.join(output_dir, "plots"); os.makedirs(plots_dir, exist_ok=True)
        view_map = config.get('data', {}).get('view_mapping', {}); view_class_names = [k for k, v in sorted(view_map.items(), key=lambda item: item[1])] if view_map else None
        try:
            can_plot_cm = 'accuracy' in eval_metrics and final_view_labels is not None and final_view_labels.numel() > 0 and final_view_logits is not None and final_view_logits.numel() > 0
            can_plot_scatter = 'mae' in eval_metrics and final_ef_labels is not None and final_ef_labels.numel() > 0 and final_ef_preds is not None and final_ef_preds.numel() > 0
            if can_plot_cm: plot_confusion_matrix(final_view_labels, final_view_logits, class_names=view_class_names, save_path=os.path.join(plots_dir, f"{split_name.lower()}_eval_confusion_matrix.png"))
            if can_plot_scatter: plot_ef_scatter(final_ef_labels, final_ef_preds, save_path=os.path.join(plots_dir, f"{split_name.lower()}_eval_ef_scatter.png"))
            if can_plot_cm or can_plot_scatter: logging.info(f"{split_name} set evaluation plots saved in {plots_dir}")
            else: logging.info(f"{split_name} set evaluation plots skipped.")
        except Exception as e: logging.error(f"Error generating evaluation plots for {split_name}: {e}", exc_info=True)

    return results

def main():


    if not DEAP_AVAILABLE:
         print("FATAL ERROR: DEAP library not found. Please run 'pip install deap'.") # Keep print for this early exit
         sys.exit(1)

    parser = argparse.ArgumentParser(description="Optimize Multi-Task Model using Evolutionary Algorithm.")
    parser.add_argument("--config", type=str, required=True, help="Path to the main multi-task config.")
    parser.add_argument("--ea_config", type=str, required=True, help="Path to the evolutionary algorithm config.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint to optimize.")
    args = parser.parse_args()

    try: config = load_config(args.config)
    except Exception as e: print(f"Failed loading main config {args.config}: {e}"); sys.exit(1)

    try: ea_config = load_config(args.ea_config)
    except Exception as e: print(f"Failed loading EA config {args.ea_config}: {e}"); sys.exit(1)

    train_cfg = config.get('training', {})
    output_dir = train_cfg.get('output_dir', 'results/multitask_default')
    ea_output_dir = os.path.join(output_dir, "optimization")
    os.makedirs(os.path.join(ea_output_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(ea_output_dir, 'checkpoints'), exist_ok=True)
    log_file = os.path.join(ea_output_dir, 'logs', 'optimize_multitask.log')

    setup_logging(log_file=log_file, level=logging.INFO)
    logging.info("--- Script 05 Main Execution Started ---")
    logging.info("--- Imports were completed prior to main() call ---")
    logging.info(f"Main Config: {args.config}; EA Config: {args.ea_config}")
    logging.info(f"Base Model Path: {args.model_path}")
    logging.info(f"Optimization results will be saved in: {ea_output_dir}")

    seed = train_cfg.get('seed', 42); set_seed(seed)
    device = torch.device(train_cfg.get('device', 'cuda') if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu': logging.warning("CUDA not available, running on CPU. EA evaluation will be slow.")
    logging.info(f"Using device: {device}")

    logging.info("Loading data for evaluation during optimization...")
    data_cfg = config.get('data', {})

    eval_split = ea_config.get('evaluation_split', 'VALIDATE')
    eval_loader = None
    try:
        eval_dataset = EchonetPseudoDataset(config=data_cfg, split=eval_split)
        if len(eval_dataset) == 0: raise RuntimeError(f"Evaluation dataset split '{eval_split}' is empty.")

        eval_batch_size = ea_config.get('evaluation_batch_size', train_cfg.get('batch_size', 8))
        eval_loader = DataLoader( eval_dataset, batch_size=eval_batch_size, shuffle=False,
            num_workers=train_cfg.get('num_workers', 0), pin_memory=(device.type=='cuda'), collate_fn=video_collate_fn )
        logging.info(f"Loaded evaluation data split '{eval_split}' with {len(eval_dataset)} samples. Eval Batch Size: {eval_batch_size}")
    except Exception as e: logging.error(f"FATAL: Failed loading evaluation data: {e}", exc_info=True); sys.exit(1)

    logging.info("Loading base multi-task model...")
    model_cfg = config.get('model', {})
    base_model = None
    try:
        num_views = data_cfg.get('num_views'); assert num_views is not None, "Config missing 'data.num_views'"
        base_model = MultiTaskCNN( backbone_name=model_cfg.get('backbone', 'r2plus1d_18'), pretrained=False,
            num_view_classes=num_views, dropout_rate=model_cfg.get('dropout_rate', 0.5) ).to(device)
        checkpoint = load_checkpoint(args.model_path, base_model, device=device)
        if not checkpoint: raise FileNotFoundError(f"Failed to load base model checkpoint from {args.model_path}")
        base_model.eval()
        base_state_dict = copy.deepcopy(base_model.state_dict())
        logging.info(f"Successfully loaded and cached base model weights from {args.model_path}")
    except Exception as e: logging.error(f"FATAL: Failed loading base model: {e}", exc_info=True); sys.exit(1)

    logging.info("Setting up Evolutionary Algorithm to optimize Loss Weights (alpha, beta)...")
    toolbox = None
    try:

        if not hasattr(creator, "FitnessMulti"): creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0)) # Min MAE, Max Kappa
        if not hasattr(creator, "Individual"): creator.create("Individual", list, fitness=creator.FitnessMulti)
        toolbox = base.Toolbox()
        BOUND_LOW, BOUND_UP = 0.0, 1.0
        toolbox.register("attr_alpha", random.uniform, BOUND_LOW, BOUND_UP)
        toolbox.register("attr_beta", random.uniform, BOUND_LOW, BOUND_UP)
        def create_individual(): return creator.Individual([toolbox.attr_alpha(), toolbox.attr_beta()])
        toolbox.register("individual", create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        def evaluate_individual(individual):
            alpha, beta = individual

            alpha = max(0.0, alpha); beta = max(0.0, beta)

            logging.debug(f"Evaluating Individual - Alpha={alpha:.4f}, Beta={beta:.4f}")
            try:
                base_model.eval()
                results = run_evaluation(base_model, eval_loader, device, config,
                                         split_name="EA_Eval", custom_alpha=alpha, custom_beta=beta)
                metrics = results.get('metrics', {})
                mae = metrics.get('mae', float('inf')); kappa = metrics.get('kappa', -float('inf'))
                if np.isnan(mae) or np.isinf(mae): mae = float('inf')
                if np.isnan(kappa) or np.isinf(kappa): kappa = -float('inf')
                logging.debug(f"Individual Fitness: MAE={mae:.4f}, Kappa={kappa:.4f}")
                return mae, kappa
            except Exception as eval_e: logging.error(f"Error during individual eval: {eval_e}", exc_info=True); return float('inf'), -float('inf')
        toolbox.register("evaluate", evaluate_individual)

        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2) # Sigma controls mutation intensity
        toolbox.register("select", tools.selNSGA2)
        logging.info("EA Toolbox setup complete for optimizing alpha/beta.")
    except Exception as e: logging.error(f"FATAL: Failed setting up EA: {e}", exc_info=True); sys.exit(1)

    logging.info("--- Starting main optimization logic ---")
    pop, logbook = None, None
    try:

        population_size = ea_config.get('population_size', 25)
        num_generations = ea_config.get('num_generations', 10)
        cxpb = ea_config.get('crossover_probability', 0.7)
        mutpb = ea_config.get('mutation_probability', 0.2)
        mu = ea_config.get('mu', population_size)
        lambda_ = ea_config.get('lambda_', population_size)
        logging.info(f"EA Parameters: Pop Size={population_size}, Generations={num_generations}, CXPB={cxpb}, MUTPB={mutpb}, Mu={mu}, Lambda={lambda_}")
        if toolbox is None: raise RuntimeError("EA Toolbox was not initialized.")

        pop = toolbox.population(n=population_size)
        hof = tools.ParetoFront()
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0); stats.register("min", np.min, axis=0); stats.register("max", np.max, axis=0)
        pop, logbook = algorithms.eaMuPlusLambda( population=pop, toolbox=toolbox, mu=mu, lambda_=lambda_,
            cxpb=cxpb, mutpb=mutpb, ngen=num_generations, stats=stats, halloffame=hof, verbose=True )

        logging.info(f"Optimization finished. Hall of Fame (Pareto Front) Size: {len(hof)}")
        logging.info("Individuals on Pareto Front:")
        best_results_list = []
        for i, ind in enumerate(hof):
            fitness_values = ind.fitness.values
            logging.info(f"  Individual {i}: Alpha={ind[0]:.4f}, Beta={ind[1]:.4f}")
            logging.info(f"  Fitness (MAE, Kappa): {[f'{x:.4f}' for x in fitness_values]}")
            best_results_list.append({'individual': list(ind), 'fitness': list(fitness_values)})

        results_file = os.path.join(ea_output_dir, "ea_optimization_results_alpha_beta.json")
        try:
             serializable_logbook = []
             if logbook:
                  for gen_data in logbook:
                       serializable_gen = { key: (val.tolist() if isinstance(val, (np.ndarray, np.generic)) else val) for key, val in gen_data.items() }
                       serializable_logbook.append(serializable_gen)
             final_results = { 'optimization_target': 'loss_weights_alpha_beta', 'pareto_front': best_results_list, 'logbook': serializable_logbook }
             with open(results_file, 'w') as f: json.dump(final_results, f, indent=4)
             logging.info(f"EA results saved to {results_file}")
        except Exception as e: logging.error(f"Failed to serialize or save EA results: {e}", exc_info=True)

    except NameError as ne: logging.error(f"Optimization loop failed: Missing EA component? {ne}", exc_info=True)
    except RuntimeError as rte: logging.error(f"Optimization loop failed: {rte}", exc_info=True)
    except Exception as e: logging.error(f"Optimization loop failed: {e}", exc_info=True)

    logging.info("--- Optimization logic finished ---")


if __name__ == "__main__":

    freeze_support()
    main()
