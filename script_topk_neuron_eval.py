import argparse
import logging
import pickle
import random
import torch
import os
from circuit import Circuit
from circuit_utils import topk_effective_components
from evaluation_utils import circuit_faithfulness_with_mean_ablation
from component import Component
from prompt_generation import OPERATORS, POSITIONS
from general_utils import generate_activations, set_deterministic, load_model, get_model_consts, get_neuron_importance_scores


torch.set_grad_enabled(False)
max_op = 300


def calc_mean_cache(model, model_name):
    eval_mean_cache_path = f'./data/{model_name}/mean_cache_for_evaluation_all_arithmetic_prompts_max_op=300.pt'
    if os.path.exists(eval_mean_cache_path):
        print('Mean cache file found, skipping creation')
        return

    print("Calculating mean cache")
    all_heads = [(l, h) for h in range(model.cfg.n_heads) for l in range(model.cfg.n_layers)]
    all_mlps = list(range(model.cfg.n_layers))
    model.set_use_attn_result(True)
    all_components = [Component('z', layer=l, head=h) for (l, h) in all_heads] + [Component('result', layer=l, head=h) for (l, h) in all_heads] + \
                        [Component('mlp_post', layer=l) for l in all_mlps] + [Component('mlp_in', layer=l) for l in all_mlps]
    
    # Notice the prompts used for mean calculation are "illegal" prompts as well. This is to keep a balance between all operators.
    all_prompts = [f"{x}{operator}{y}=" for operator in OPERATORS for x in range(0, max_op) for y in range(0, max_op)]

    cached_activations = generate_activations(model, all_prompts, all_components, pos=None, reduce_mean=True)
    cached_activations = {c: a[None, ...].to(device='cpu') for c, a in zip(all_components, cached_activations)}
    torch.save(cached_activations, eval_mean_cache_path)


def build_circuit(model, model_name, operator_idx, mlp_top_neurons):
    heads = []

    ie_maps = torch.load(f'./data/{model_name}/ie_maps_activation_patching.pt')
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
    
    if model_name == 'llama3-8b':
        if operator_idx == 0:
            # Addition
            heads = [Component('z', layer=l, head=h) for (l, h) in [(2, 2), (5, 3), (5, 31), (14, 12), (15, 13), (16, 21)]]
        elif operator_idx == 1:
            # Subtraction
            heads = [Component('z', layer=l, head=h) for (l, h) in [(2, 2), (13, 21), (13, 22), (14, 12), (15, 13), (16, 21)]]
        elif operator_idx == 2:
            # Multiplication
            heads = [Component('z', layer=l, head=h) for (l, h) in [(2, 2), (5, 30), (8, 15), (9, 26), (13, 18), (13, 21), (13, 22), 
                                                                    (14, 12), (14, 13), (15, 8), (15, 13), (15, 14), (15, 15), (16, 3), 
                                                                    (16, 21), (17, 24), (17, 26), (18, 16), (20, 2), (22, 1)]]
        elif operator_idx == 3:
            # Division
            heads = [Component('z', layer=l, head=h) for (l, h) in [(2, 2), (5, 31), (15, 13), (15, 14), (16, 21), (18, 16)]]
    elif model_name == 'gptj' or 'pythia-6.9b' in model_name or model_name == 'llama3-70b':
        heads += topk_effective_components(model, ie_maps, k=100 if model_name == 'llama3-70b' else 50, heads_only=True).keys()
    else:
        raise ValueError(f"Unknown model {model_name}")
    
    partial_mlp_layers = list(range(get_model_consts(model_name).first_heuristics_layer, model.cfg.n_layers))
    full_mlps = [Component('mlp_post', layer=l) for l in range(model.cfg.n_layers) if l not in partial_mlp_layers]
    partial_mlps = [Component('mlp_post', layer=l, neurons=mlp_top_neurons[l]) for l in partial_mlp_layers]

    full_circuit = Circuit(model.cfg)
    for c in list(set(heads + full_mlps + partial_mlps)):
        full_circuit.add_component(c)
    return full_circuit


def evaluate_circuit_with_topk_neurons(model, model_name, evaluation_prompts_and_answers):
    PROMPT_COUNT_TO_USE = 50

    # Load mean cache for ablations
    mean_cache = torch.load(f'./data/{model_name}/mean_cache_for_evaluation_all_arithmetic_prompts_max_op=300.pt')
    if mean_cache[list(mean_cache.keys())[0]].shape[0] != PROMPT_COUNT_TO_USE:
        # Mean cache was saved as a single vector, we repeat it for the length of the evaluation prompts
        mean_cache = {c: a.repeat(PROMPT_COUNT_TO_USE, 1, 1) for c, a in mean_cache.items()}

    # Settings
    k_values = torch.tensor(sorted(list(range(0, 500, 10)) + list(range(500, model.cfg.d_mlp, 50)) + [model.cfg.d_mlp]))
    seeds = [42, 412, 32879]

    # Load existing results
    results_file_path = f'./data/{model_name}/topk_neuron_faithfulness_evaluation_results.pt'
    if os.path.exists(results_file_path):
        faithfulness_per_k = torch.load(results_file_path)
    else:
        faithfulness_per_k = {}

    # Calculate faithfulness of the model for each operator and seed (The seed affects the prompts chosen for evaluation)
    for seed in seeds:
        for operator_idx in range(len(OPERATORS)):
            logging.info(f"Starting {operator_idx=}, {seed=}")
            if (operator_idx, seed) in faithfulness_per_k:
                logging.info(f"Found results file for {operator_idx=}, {seed=}")
                continue
            set_deterministic(seed)
            prompts_and_answers = random.sample(evaluation_prompts_and_answers[operator_idx], k=PROMPT_COUNT_TO_USE)

            mlppost_neuron_scores = get_neuron_importance_scores(model, model_name, operator_idx=operator_idx, pos=-1) # Ranking neurons according to Attribution patching

            faithfulness_per_k[(operator_idx, seed)] = torch.zeros((len(k_values),))
            for i, k in enumerate(k_values):
                mlp_top_neurons = {}
                for mlp in range(1, model.cfg.n_layers):
                    mlp_top_neurons[mlp] = mlppost_neuron_scores[mlp].topk(k).indices.tolist()
                full_circuit = build_circuit(model, model_name, operator_idx, mlp_top_neurons)
                faithfulness_per_k[(operator_idx, seed)][i] = circuit_faithfulness_with_mean_ablation(model, full_circuit, prompts_and_answers, mean_cache, metric='nl')
                logging.info(f"{k=}, faithfulness={faithfulness_per_k[(operator_idx, seed)][i].item()}")
            torch.save(faithfulness_per_k, results_file_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Name of the model to be loaded')
    parser.add_argument('--model_path', type=str, help='Path to the model to be loaded')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model = load_model(args.model_name, args.model_path, "cuda")
    logging.info("Loaded model")

    # Pre-calculate mean cache, if doesn't exist
    calc_mean_cache(model, args.model_name)
    logging.info("Verified / Created mean cache")

    # Load prompts
    with open(fr'./data/{args.model_name}/large_prompts_and_answers_max_op=300.pkl', 'rb') as f:
        large_prompts_and_answers = pickle.load(f)
        large_prompts_and_answers = [[pa for pa in large_prompts_and_answers[op_idx] if pa[1] != '0'] for op_idx in range(len(OPERATORS))] # Drop prompts with zero for an answer due to bug (mean cache in Pythia leading to 0 logit, thus bad == good baseline in faithfulness function)
    
    # Evaluate the model's circuit 
    evaluate_circuit_with_topk_neurons(model, args.model_name, large_prompts_and_answers)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()