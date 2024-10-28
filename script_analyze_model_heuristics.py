import logging
import re
import argparse
import torch
import random
import pickle
import os
import plotly.express as px
from component import Component
from eap.attr_patching import node_attribution_patching
from evaluation_utils import model_accuracy
from general_utils import generate_activations, set_deterministic, safe_eval, load_model, get_model_consts, get_neuron_importance_scores
from prompt_generation import OPERATORS, OPERATOR_NAMES, _get_operand_range, _maximize_unique_answers, generate_all_prompts_for_operator, generate_prompts, separate_prompts_and_answers
from heuristics_classification import HeuristicAnalysisData, classify_heuristic_neurons, load_heuristic_classes
from heuristics_analysis import heuristic_class_knockout_experiment, prompt_knockout_experiment


# Save some globals to avoid passing them around between all functions
model_path = None
model_name = None
model = None
model_consts = None
seed = None
max_op = 300
op_ranges = {'+': (0, max_op), '-': (0, max_op), '*': (0, max_op), '/': (1, max_op)}
HEURISTIC_MATCH_THRESHOLD = 0.6
NEURONS_TO_VISUALIZE_PER_LAYER = 5


def _get_top_op1_op2_indices(prompts_activations, op1_op2_pairs, layer, neuron_idx, top_k=None, is_top=True):
    """
    Get the (op1, op2) pairs that cause the highest (or lowest) activations for a given neuron.
    """
    activation_map = prompts_activations[(layer, neuron_idx)]
    if top_k is None:
        top_k = len(activation_map)
    top_op1_op2_pairs = op1_op2_pairs[activation_map.topk(top_k, largest=is_top).indices.cpu().numpy()].tolist()
    return top_op1_op2_pairs


def load_prompts():
    """
    Load (or generate, if not already generated) the prompts for the analysis.
    """
    analysis_prompts_file_path = fr'./data/{model_name}/large_prompts_and_answers_max_op={max_op}.pkl'
    if os.path.exists(analysis_prompts_file_path):
        with open(analysis_prompts_file_path, 'rb') as f:
            large_prompts_and_answers = pickle.load(f)
    else:
        large_prompts_and_answers = generate_prompts(model, operand_ranges=op_ranges, correct_prompts=True, num_prompts_per_operator=None, 
                                                     single_token_number_range=(0, model_consts.max_single_token))
        with open(analysis_prompts_file_path, 'wb') as f:
            pickle.dump(large_prompts_and_answers, f)

    set_deterministic(seed)
    # for i in range(len(large_prompts_and_answers)):
        # random.shuffle(large_prompts_and_answers[i])
    wanted_size = 50
    filtered_prompts_and_answers = []
    for pa in large_prompts_and_answers:
        new_pa = []
        for p, a in pa:
            # Filter out simple prompts (x/0, x*1, etc)
            op1, op2 = tuple(map(int, re.findall(r'\d+', p)))[-2:]
            if op1 > 5 and op2 > 5 and int(a) > 2:
                new_pa.append((p, a))
        if len(new_pa) < wanted_size:
            print(len(new_pa), ' length of new_pa')
            new_pa = new_pa + random.sample(pa, k=wanted_size - len(new_pa))
        filtered_prompts_and_answers.append(new_pa)
    correct_prompts_and_answers = [_maximize_unique_answers(pa, k=wanted_size) for pa in filtered_prompts_and_answers]
    return large_prompts_and_answers, correct_prompts_and_answers


def save_model_accuracy():
    """
    Calculate the model accuracy on all arithmetic operators and save it to a file.
    """
    acc_path = f"./data/{model_name}/accuracy.pt"
    if os.path.exists(acc_path):
        acc_dict = torch.load(acc_path)
        remaining_operators = list(set(OPERATORS) - set(acc_dict.keys()))
    else:
        acc_dict = {}
        remaining_operators = OPERATORS
    
    for operator in remaining_operators:
        min_op = 1 if operator == '/' else 0
        prompts = generate_all_prompts_for_operator(operator, min_op, max_op)
        answers = [str(int(eval(p[:-1]))) for p in prompts]
        acc = model_accuracy(model, prompts, answers, None, verbose=False)
        logging.info(f"Model {model_name} accuracy on simple prompts with operator {operator} is: {acc :.3f}")
        acc_dict[operator] = acc
        torch.save(acc_dict, acc_path)


def calc_neuron_importance_scores(operator_idx, correct_prompts_and_answers, device='cuda'):
    """
    Calculate the neuron indirect effect for all middle- and late- MLP neurons.
    Uses attribution patching for fast approximation, if activation patching results are not pre-calculated.
    """
    set_deterministic(seed)
    
    neuron_importance_scores_path = f"./data/{model_name}/{OPERATOR_NAMES[operator_idx]}_node_attribution_scores.pt"
    if os.path.exists(neuron_importance_scores_path):
        logging.info("Skipping neuron importance scores calculation (Found results file)")
        return
    
    prompts_and_answers = correct_prompts_and_answers[operator_idx]
    corrupt_prompts_and_answers = random.sample(sum(correct_prompts_and_answers, []), k=len(prompts_and_answers))
    
    if device == 'cpu':
        model_cpu = load_model(model_name, model_path, 'cpu')
        model_cpu.reset_hooks()
        attribution_scores = node_attribution_patching(model_cpu, prompts_and_answers, corrupt_prompts_and_answers,
                                                    attributed_hook_names=['mlp.hook_post'],
                                                    metric='IE', batch_size=10)
    else:
        attribution_scores = node_attribution_patching(model, prompts_and_answers, corrupt_prompts_and_answers,
                                                    attributed_hook_names=['mlp.hook_post'],
                                                    metric='IE', batch_size=1)
    torch.save(attribution_scores, neuron_importance_scores_path)


def calc_heuristic_neuron_matching_scores(operator_idx, activation_map_type, first_heuristics_layer, neuron_importance_scores):
    """
    Calculate a dictionary of matching scores between each important (high effect) neuron and each heuristic.
    These scores are later thresholded to determine which neurons implement which heuristics.
    """
    logging.info(f"Analyzing heuristic neuron matching for {OPERATOR_NAMES[operator_idx]}, activation map type: {activation_map_type}")
    results_file_path = f"./data/{model_name}/{OPERATOR_NAMES[operator_idx]}_heuristic_matches_dict_{activation_map_type}_maps.pt"
    if os.path.exists(results_file_path):
        logging.info(f"Skipping heuristic neuron matching for {OPERATOR_NAMES[operator_idx]} (Found results file)")
        return

    min_op = 1 if operator_idx == 3 else 0
    op1_op2_pairs = torch.tensor(sorted([(op1, op2) for op1 in range(min_op, max_op) for op2 in _get_operand_range(OPERATORS[operator_idx], op1, min_op, max_op, model_consts.max_single_token)]))
    prompts = [f'{op1}{OPERATORS[operator_idx]}{op2}=' for (op1, op2) in op1_op2_pairs]
    
    logging.info("Create a list of heuristical neurons in the relevant layers")
    neuron_importance_scores = get_neuron_importance_scores(model, model_name, operator_idx=operator_idx, pos=-1)
    heuristic_neurons = []
    for layer in range(first_heuristics_layer, model.cfg.n_layers):
        heuristic_neurons += [(layer, neuron) for neuron in neuron_importance_scores[layer].topk(model_consts.topk_neurons_per_layer).indices.tolist()]
                          
    logging.info("Calculate K & KV activations maps for all (op1, op2) possible pairs")
    # Calculate k (key) prompt activations
    k_prompts_activations = generate_activations(model, prompts, [Component('mlp_post', layer=l) for l in range(model.cfg.n_layers)], pos=-1, batch_size=32)
    k_prompts_activations = {(layer, neuron): k_prompts_activations[layer][:, neuron] for (layer, neuron) in heuristic_neurons}

    # Calculate kv (key-value) prompt activations by multiplying the key activations with the value vector logits
    kv_prompts_activations = {}
    results_for_all_pairs = [str(safe_eval(f'{op1}{OPERATORS[operator_idx]}{op2}')) for (op1, op2) in op1_op2_pairs]
    results_for_all_pairs_labels = model.to_tokens(results_for_all_pairs, prepend_bos=False).view(-1)
    for (layer, neuron) in heuristic_neurons:
        v_vector_logits = (model.blocks[layer].mlp.W_out[neuron].to(model.cfg.device) @ model.W_U.to(model.cfg.device))
        logits_for_all_pairs = v_vector_logits[results_for_all_pairs_labels.to(model.cfg.device)].cpu()
        kv_prompts_activations[(layer, neuron)] = k_prompts_activations[(layer, neuron)] * logits_for_all_pairs

    logging.info("Save a visualization of top neurons in each layer")
    neuron_vis_path = f"./data/{model_name}/{OPERATOR_NAMES[operator_idx]}_neuron_visualizations"
    for layer in range(first_heuristics_layer, model.cfg.n_layers):
        top_neurons = neuron_importance_scores[layer].topk(NEURONS_TO_VISUALIZE_PER_LAYER).indices.tolist()
        for neuron in top_neurons:
            neuron_vis_path = f"./data/{model_name}/{OPERATOR_NAMES[operator_idx]}_neuron_visualizations/mlp{layer}_neuron{neuron}_{activation_map_type}_map.png"
            if not os.path.exists(neuron_vis_path):
                activations = k_prompts_activations if activation_map_type == "K" else kv_prompts_activations
                activation_img = torch.zeros((max_op - min_op, max_op - min_op))
                for i, (op1, op2) in enumerate(op1_op2_pairs):
                    activation_img[op1 - min_op, op2 - min_op] = activations[(layer, neuron)][i]
                fig = px.imshow(activation_img, x=list(range(min_op, max_op)), y=list(range(min_op, max_op)), 
                                labels={'x': 'Operand2', 'y': 'Operand1'}, width=600, 
                                title=f'MLP{layer}#{neuron} {activation_map_type} activation map as function of operands ({OPERATOR_NAMES[operator_idx]})',
                                color_continuous_midpoint=0.0, color_continuous_scale="RdBu")
                os.makedirs(os.path.dirname(neuron_vis_path), exist_ok=True)
                fig.write_image(neuron_vis_path)

    logging.info(f"Cache the top and bottom operand pairs and results for all heuristic neurons")
    prompts_activations = k_prompts_activations if activation_map_type == "K" else kv_prompts_activations
    top_op1_op2_indices = {(layer, neuron): _get_top_op1_op2_indices(prompts_activations, op1_op2_pairs, layer, neuron, is_top=True) for (layer, neuron) in heuristic_neurons}
    top_results = {}
    for (layer, neuron) in top_op1_op2_indices.keys():
        top_results[(layer, neuron)] = [safe_eval(f"{op1}{OPERATORS[operator_idx]}{op2}") for (op1, op2) in top_op1_op2_indices[(layer, neuron)]]
        assert all([0 <= result <= model_consts.max_single_token for result in top_results[(layer, neuron)]])

    heuristic_data = HeuristicAnalysisData()
    heuristic_data.also_check_bottom_results = model_consts.mlp_activations_also_negative
    heuristic_data.op1_op2_pairs = op1_op2_pairs
    heuristic_data.top_op1_op2_indices = top_op1_op2_indices
    heuristic_data.top_results = top_results
    heuristic_data.max_op = max_op
    heuristic_data.max_single_token = model_consts.max_single_token
    heuristic_data.operator_idx = operator_idx
    heuristic_data.k_per_heuristic_cache = {}

    heuristic_classes = classify_heuristic_neurons(heuristic_neurons, heuristic_data, verbose=False)
    torch.save(heuristic_classes, results_file_path)


def perform_heuristic_knockout_experiment(operator_idx, knockout_type):
    results_file_path = f'./data/{model_name}/{OPERATOR_NAMES[operator_idx]}_heuristic_ablation_results_thres={HEURISTIC_MATCH_THRESHOLD}_{knockout_type}_maps.pt'
    if os.path.exists(results_file_path):
        logging.info(f"Skipping heuristic knockout for {OPERATOR_NAMES[operator_idx]} (Found results file)")
        return
    
    set_deterministic(seed)
    heuristic_classes = load_heuristic_classes(f"./data/{model_name}", operator_idx, knockout_type)
    min_op = 1 if operator_idx == 3 else 0
    heuristics_knockout_results = heuristic_class_knockout_experiment(heuristic_classes, operator_idx, model, min_op, max_op, 
                                                                      model_consts.max_single_token,
                                                                      heuristic_neuron_match_threshold=HEURISTIC_MATCH_THRESHOLD,
                                                                      seed=seed, verbose=True)
    torch.save(heuristics_knockout_results, results_file_path)
    

def perform_prompt_knockout_experiment(prompts_and_answers, operator_idx, knockout_type):
    results_file_path = f'./data/{model_name}/{OPERATOR_NAMES[operator_idx]}_prompt_ablation_results_thres={HEURISTIC_MATCH_THRESHOLD}_{knockout_type}_maps.pt'
    if os.path.exists(results_file_path):
        logging.info(f"Skipping prompt knockout for {OPERATOR_NAMES[operator_idx]} (Found results file)")
        return
    
    logging.info(f"Starting prompt knockout experiment for {OPERATOR_NAMES[operator_idx]}")
    set_deterministic(seed)
    neuron_hard_limits = range(0, model_consts.topk_neurons_per_layer // 2 + 1, 5) 

    baseline_results = torch.zeros((len(neuron_hard_limits),))
    ablated_results = torch.zeros((len(neuron_hard_limits),))
    ablated_neuron_counts = torch.zeros((len(neuron_hard_limits),))
    control_results = torch.zeros((len(neuron_hard_limits),))

    for neuron_hard_limit_idx, neuron_hard_limit in enumerate(neuron_hard_limits):
        neuron_importance_scores = get_neuron_importance_scores(model, model_name, operator_idx=operator_idx, pos=-1)
        all_top_neurons = []
        for layer in range(model_consts.first_heuristics_layer, model.cfg.n_layers):
            all_top_neurons += [(layer, neuron) for neuron in neuron_importance_scores[layer].topk(model_consts.topk_neurons_per_layer).indices.tolist()]

        heuristic_classes = load_heuristic_classes(f'./data/{model_name}', operator_idx, knockout_type)
        heuristic_classes = {name: [(l, n, s) for (l, n, s) in layer_neuron_scores if s >= HEURISTIC_MATCH_THRESHOLD] for name, layer_neuron_scores in heuristic_classes.items()}
        baseline, ablated, ablated_neuron_avg_count, control_ablated = prompt_knockout_experiment(heuristic_classes, 
                                                                                                            model, prompts_and_answers, 
                                                                                                            neuron_count_hard_limit_per_layer=neuron_hard_limit,
                                                                                                            all_top_neurons=all_top_neurons,
                                                                                                            metric_fn=model_accuracy)
        baseline_results[neuron_hard_limit_idx] = baseline
        ablated_results[neuron_hard_limit_idx] = ablated
        ablated_neuron_counts[neuron_hard_limit_idx] = ablated_neuron_avg_count
        control_results[neuron_hard_limit_idx] = control_ablated

    torch.save((neuron_hard_limits, baseline_results, ablated_results, ablated_neuron_counts, control_results), results_file_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Name of the model to be loaded')
    parser.add_argument('--model_path', type=str, help='Path to the model to be loaded')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--heuristic_knockout_type', type=str, choices=['K', 'KV', 'HYBRID', 'ALL'], default='ALL')
    args = parser.parse_args()
    return args


def main():
    """
    This script pipelines the process of analyzing a given model's heuristics.
    It first calculates various pre-requisites for this (i.e. model accuracy, neuron importance scores, etc), and then proceeds to analyze the model's heuristic
    neurons and performs the two ablation experiments described in the paper.

    Most function implementations in this script are slightly-edited versions of the ones found in the llm-arithmetics-analysis-main.ipynb notebook.
    """
    global model_path
    global model_name
    global model_consts
    global model
    global seed
    args = parse_args()
    model_name = args.model_name
    model_path = args.model_path
    seed = args.seed
    logging.info(f"Starting analysis for model {model_name} ({seed=})")

    torch.set_grad_enabled(False)
    device = 'cuda'
    set_deterministic(seed)

    logging.info(f"Loading model {args.model_name}")
    model = load_model(args.model_name, args.model_path, device, extra_hooks=False)
    model_consts = get_model_consts(args.model_name)

    os.makedirs(f"./data/{model_name}", exist_ok=True)
    logging.info(f"Loading prompts)")
    _, correct_prompts_and_answers = load_prompts()
    logging.info(f"Using prompts: {correct_prompts_and_answers}")

    save_model_accuracy()

    for operator_idx in range(len(OPERATORS)):
        logging.info(f"Model {model_name}, Op {OPERATOR_NAMES[operator_idx]}, Starting analysis")
        logging.info(f"Model {model_name}, Op {OPERATOR_NAMES[operator_idx]}: First good enough layer is {model_consts.first_heuristics_layer}")

        # Find the top most effective neurons in mlps (starting from the layer where the answer can be extracted), and save them
        calc_neuron_importance_scores(operator_idx, correct_prompts_and_answers, device='cuda' if 'llama3-70b' in model_name else 'cpu')
        logging.info(f"Model {model_name}, Op {OPERATOR_NAMES[operator_idx]}: Neuron importance scores calculated")       
        neuron_importance_scores = get_neuron_importance_scores(model, model_name, operator_idx=operator_idx, pos=-1)
        for layer in range(model_consts.first_heuristics_layer, model.cfg.n_layers):
            topk_effect_percentage = neuron_importance_scores[layer].topk(model_consts.topk_neurons_per_layer).values.sum() / neuron_importance_scores[layer].sum()
            logging.info(f"Model {model_name}, Op {OPERATOR_NAMES[operator_idx]}: Top {model_consts.topk_neurons_per_layer} neurons in layer {layer} are responsible for {topk_effect_percentage:.3f} of the total effect")

        # Score each pair of (heuristic_class, neuron)
        calc_heuristic_neuron_matching_scores(operator_idx, "K", model_consts.first_heuristics_layer, neuron_importance_scores)
        calc_heuristic_neuron_matching_scores(operator_idx, "KV", model_consts.first_heuristics_layer, neuron_importance_scores)
        logging.info(f"Model {model_name}, Op {OPERATOR_NAMES[operator_idx]}: Heuristic neuron matching scores calculated")

        # Heuristic knockout experiment
        if args.heuristic_knockout_type != "ALL":
            perform_heuristic_knockout_experiment(operator_idx, knockout_type=args.heuristic_knockout_type)
            perform_prompt_knockout_experiment(correct_prompts_and_answers[operator_idx], operator_idx, knockout_type=args.heuristic_knockout_type)
        else:
            for knockout_type in ['K', 'KV', 'HYBRID']:
                perform_heuristic_knockout_experiment(operator_idx, knockout_type=knockout_type)
                perform_prompt_knockout_experiment(correct_prompts_and_answers[operator_idx], operator_idx, knockout_type=knockout_type)
        logging.info(f"Model {model_name}, Op {OPERATOR_NAMES[operator_idx]}: Heuristic knockout experiment completed")

    logging.info(f"Model {model_name}: Analysis done")


# srun -A nlp -p nlp -c 10 --gres=gpu:A40:1 python3 script_analyze_model_heuristics.py --model_name=pythia-6.9b-step143000 --model_path=/mnt/nlp/models/pythia-6.9b/step143000  --heuristic_knockout_type=ALL >> ~/projects/llm-arithmetic-analysis/output_logs/pythia_6_9_step_143000_analysis.log 2>&1 &    
if __name__ == '__main__':
    logging.basicConfig(level = logging.INFO)
    main()
