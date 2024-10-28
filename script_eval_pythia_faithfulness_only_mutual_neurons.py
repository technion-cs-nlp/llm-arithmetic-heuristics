import argparse
import os
import pickle
import random
import torch
import logging
from itertools import chain
from circuit import Circuit
from component import Component
from circuit_utils import topk_effective_components
from evaluation_utils import circuit_faithfulness_with_mean_ablation
from general_utils import generate_activations, get_neuron_importance_scores, set_deterministic, load_model
from heuristics_classification import load_heuristic_classes
from prompt_generation import OPERATORS, POSITIONS
from model_analysis_consts import PYTHIA_6_9B_CONSTS


device = 'cuda'
HEURISTIC_MATCH_THRESHOLD = 0.6
PYTHIA_PREFIX = "pythia-6.9b"


def load_mean_cache(model, model_name):
    """
    Load (or calculate) the mean activation for each component in the model, to be used for evaluation.
    """
    max_op = 300
    eval_mean_cache_path = f'./data/{model_name}/mean_cache_for_evaluation_all_arithmetic_prompts_max_op={max_op}.pt'
    if os.path.exists(eval_mean_cache_path):
        cached_activations = torch.load(eval_mean_cache_path)
        print('Loaded cached activations from file')
    else:
        all_heads = [(l, h) for h in range(model.cfg.n_heads) for l in range(model.cfg.n_layers)]
        all_mlps = list(range(model.cfg.n_layers))
        model.set_use_attn_result(True)
        all_components = [Component('z', layer=l, head=h) for (l, h) in all_heads] + \
                         [Component('mlp_post', layer=l) for l in all_mlps]
        all_prompts = [f"{x}{operator}{y}=" for operator in OPERATORS for x in range(0, max_op) for y in range(0, max_op)]
        cached_activations = generate_activations(model, all_prompts, all_components, pos=None, reduce_mean=True, batch_size=64)
        cached_activations = {c: a[None, ...].to(device='cpu') for c, a in zip(all_components, cached_activations)}
        torch.save(cached_activations, eval_mean_cache_path)
    return cached_activations


def build_circuit(model, mlp_neurons, operator_idx):
    # heads = [Component('z', layer=l, head=h) for l in range(0, model.cfg.n_layers) for h in range(0, model.cfg.n_heads)]
    ie_maps = torch.load(f'./data/{PYTHIA_PREFIX}/ie_maps_activation_patching.pt')
    summed_seed_ie_maps = {}
    seeds = set([])
    for op_idx, pos, seed in ie_maps.keys():
        seeds.add(seed)
        if (op_idx, pos) not in summed_seed_ie_maps:
            summed_seed_ie_maps[(op_idx, pos)] = ie_maps[(op_idx, pos, seed)]
        else:
            summed_seed_ie_maps[(op_idx, pos)] += ie_maps[(op_idx, pos, seed)]
    ie_maps = {k: v / len(seeds) for (k, v) in summed_seed_ie_maps.items()}
    ie_maps = torch.stack([ie_maps[(operator_idx, pos)] for pos in POSITIONS]).mean(dim=0)
    heads = topk_effective_components(model, ie_maps, k=50, heads_only=True).keys()

    partial_mlp_layers = list(range(PYTHIA_6_9B_CONSTS.first_heuristics_layer, model.cfg.n_layers))
    full_mlps = [Component('mlp_post', layer=l) for l in range(model.cfg.n_layers) if l not in partial_mlp_layers]
    partial_mlps = [Component('mlp_post', layer=l, neurons=mlp_neurons[l]) for l in partial_mlp_layers]

    full_circuit = Circuit(model.cfg)
    for c in list(set(heads + full_mlps + partial_mlps)):
        full_circuit.add_component(c)
    return full_circuit


def get_heuristic_neurons(model, model_name, operator_idx):
    heuristic_classes = load_heuristic_classes(f"./data/{model_name}", operator_idx, "HYBRID")

    # Filter by threshold
    heuristic_classes = {name: [(l, n, s) for (l, n, s) in layer_neuron_scores if s >= HEURISTIC_MATCH_THRESHOLD] for name, layer_neuron_scores in heuristic_classes.items()}
    heuristic_classes = {name: lns for name, lns in heuristic_classes.items() if len(lns) > 0}

    heuristic_neurons = {layer: [n for (l, n) in set([(v[0], v[1]) for v in chain.from_iterable(heuristic_classes.values())]) if l == layer] for layer in range(model.cfg.n_layers)}
    return heuristic_neurons


def get_intersection_neurons(model, model_name, operator_idx):
    """
    Get the neurons who are classified into the same heuristic both in the tested model (supplied by model_name) as well as the last checkpoint (step 143K) model.
    """
    # Generate the heuristic list in the last (GT) checkpoint
    step_to_compare_to = "143000"
    gt_model_name = f"{PYTHIA_PREFIX}-step{step_to_compare_to}"
    
    gt_heuristic_classes = load_heuristic_classes(f"./data/{gt_model_name}", operator_idx, "HYBRID")
    # Filter by threshold
    gt_heuristic_classes = {name: [(l, n, s) for (l, n, s) in layer_neuron_scores if s >= HEURISTIC_MATCH_THRESHOLD] for name, layer_neuron_scores in gt_heuristic_classes.items()}
    gt_heuristic_classes = {name: lns for name, lns in gt_heuristic_classes.items() if len(lns) > 0}    
    gt_heuristic_neuron_pairs = [(h_name, l, n) for h_name, lns in gt_heuristic_classes.items() for (l, n, s) in lns]

    # Generate the heuristic list in the current model
    heuristic_classes = load_heuristic_classes(f"./data/{model_name}", operator_idx, "HYBRID")
    # Filter by threshold
    heuristic_classes = {name: [(l, n, s) for (l, n, s) in layer_neuron_scores if s >= HEURISTIC_MATCH_THRESHOLD] for name, layer_neuron_scores in heuristic_classes.items()}
    heuristic_classes = {name: lns for name, lns in heuristic_classes.items() if len(lns) > 0}
    heuristic_neuron_pairs = [(h_name, l, n) for h_name, lns in heuristic_classes.items() for (l, n, s) in lns]

    # Get the intersection of the neurons
    mutual_neurons = list(set([(l, n) for (h_name, l, n) in set(gt_heuristic_neuron_pairs).intersection(set(heuristic_neuron_pairs))]))
    mutual_neurons = {layer: [n for (l, n) in mutual_neurons if l == layer] for layer in range(model.cfg.n_layers)}
    return mutual_neurons
    

def get_topk_neurons_per_layer(model, model_name, k=200, operator_idx=0, pos=-1):
    """
    Get a dictionary of {layer: [neurons]} where the neurons are the top-k neurons (sorted by indirect effect) in the given layer.
    """
    neurons_scores = get_neuron_importance_scores(model, model_name, operator_idx=operator_idx, pos=pos)
    neurons_scores = {layer: neurons.topk(k).indices.tolist() for layer, neurons in neurons_scores.items()}
    return neurons_scores

    
def calculate_faithfulness(model, model_name, mean_cache):
    """
    Calculate the faithfulness of the arithmetic circuit, in few settings: 
        1. With all top neurons in each layer in the given model. (Should be high faithfulness, this is used as a sanity check and not presented in the figures).
        2. With all neurons among the top neurons that implement heuristics.
        3. With all neurons among the top neurons that implement heuristics and also implement the same heuristic in the final checkpoint.
    The results are saved and later analyzed into the figures shown in the relevant section in the paper.
    """
    with open(fr'./data/{model_name}/large_prompts_and_answers_max_op=300.pkl', 'rb') as f:
        large_prompts_and_answers = pickle.load(f)

    for operator_idx in range(len(OPERATORS)):
        prompts_and_answers = random.sample(large_prompts_and_answers[operator_idx], k=50)

        # Sanity check faithfulness
        topk_neurons = get_topk_neurons_per_layer(model, model_name, k=200)
        full_circuit = build_circuit(model, topk_neurons, operator_idx)
        sanity_faithfulness = circuit_faithfulness_with_mean_ablation(model, full_circuit, prompts_and_answers, mean_cache, metric='nl')

        # Get the heuristic neurons in each layer and Calculate faithfulness based on them
        mlp_neurons = get_heuristic_neurons(model, model_name, operator_idx)
        full_circuit = build_circuit(model, mlp_neurons, operator_idx)
        baseline_faithfulness = circuit_faithfulness_with_mean_ablation(model, full_circuit, prompts_and_answers, mean_cache, metric='nl')

        # Get the neurons n who belong to a heuristic h where (h,n) also appears in the last checkpoint and calculate faithfulness based on them
        mutual_neurons_with_final_step = get_intersection_neurons(model, model_name, operator_idx)
        full_circuit = build_circuit(model, mutual_neurons_with_final_step, operator_idx)
        mutual_faithfulness = circuit_faithfulness_with_mean_ablation(model, full_circuit, prompts_and_answers, mean_cache, metric='nl')

        # Save the results
        logging.info(f"Operator {operator_idx}: Sanity faithfulness: {sanity_faithfulness}, Baseline faithfulness: {baseline_faithfulness}, Mutual faithfulness: {mutual_faithfulness}")
        results_file_path = f'./data/pythia-6.9b-step143000/mutual_faithfulness_results.pt'
        if os.path.exists(results_file_path):
            results = torch.load(results_file_path)
        else:
            results = {}
        results[(model_name, operator_idx)] = (sanity_faithfulness, baseline_faithfulness, mutual_faithfulness)
        torch.save(results, './data/pythia-6.9b-step143000/mutual_faithfulness_results.pt')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Name of the model to be loaded')
    parser.add_argument('--model_path', type=str, help='Path to the model to be loaded')
    args = parser.parse_args()
    return args


# Generate the heuristics intersection with a chosen training step (without categorization, weighted mean across all heuristics)
def main():
    """
    This script organizes the experiments done for the first part of section 5 (Analysis across Pythia-6.9B training checkpoints).
    """
    torch.set_grad_enabled(False)

    args = parse_args()
    model_name = args.model_name
    model_path = args.model_path

    logging.info("Loading model")
    model = load_model(model_name, model_path, device, False)

    logging.info("Calculating mean cache")
    mean_cache = load_mean_cache(model, model_name)
    mean_cache = {c: a.repeat(50, 1, 1) for c, a in mean_cache.items()}

    print("Calculating faithfulness")
    set_deterministic(42)
    calculate_faithfulness(model, args.model_name, mean_cache)


if __name__ == '__main__':
    logging.basicConfig(level = logging.INFO)
    main()



