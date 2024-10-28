import argparse
import random
import pickle
import os
import torch
from functools import partial
from prompt_generation import OPERATORS
from general_utils import set_deterministic, load_model
from eap.attr_patching import node_attribution_patching
from activation_patching import activation_patching_experiment


torch.set_grad_enabled(False)
device = 'cuda'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Name of the model to be loaded')
    parser.add_argument('--model_path', type=str, help='Path to the model to be loaded')
    parser.add_argument('--do_attribution', action='store_true', help='Whether to perform node attribution instead of activation patching')
    args = parser.parse_args()
    return args


def manual_ap_localization(model_name, model_path):
    """
    Find the arithmetic circuit using activation patching on each component (head or MLP) of the model.
    """
    model = load_model(model_name, model_path, device)

    results_file_path = f'./data/{model_name}/ie_maps_activation_patching.pt'
    max_op = 300
    analysis_prompts_file_path = fr'./data/{model_name}/large_prompts_and_answers_max_op={max_op}.pkl'
    set_deterministic(42)

    # Load pre-calculated prompts and answers
    with open(analysis_prompts_file_path, 'rb') as f:
        large_prompts_and_answers = pickle.load(f)
    for i in range(len(large_prompts_and_answers)):
        random.shuffle(large_prompts_and_answers[i])
    correct_prompts_and_answers = [pa[:50] for pa in large_prompts_and_answers]

    seeds = [42, 412, 32879, 123, 436]
    if os.path.exists(results_file_path):
        ie_maps = torch.load(results_file_path)
    else:
        ie_maps = {}

    def head_hooking_func(value, hook, head_index, token_pos, cache):
        if token_pos is None:
            value[:, :, head_index, :] = cache[hook.name][:, :, head_index, :] # For z hooking
        else:
            value[:, token_pos, head_index, :] = cache[hook.name][:, token_pos, head_index, :] # For z hooking
        return value

    # Patch each component (MLP and attention head) at each token position, for each operator and each random seed.
    for token_pos in [4, 3, 2, 1]:
        for operator_idx in range(len(OPERATORS)):
            for seed in seeds:
                if (operator_idx, token_pos, seed) in ie_maps.keys():
                    continue
                print(f"{operator_idx=}, {token_pos=}, {seed=}")
                correct_pa = correct_prompts_and_answers[operator_idx]
                corrupt_pa = random.sample(sum(correct_prompts_and_answers, []), len(correct_pa))
                ie_maps[(operator_idx, token_pos, seed)] = torch.zeros((model.cfg.n_layers, model.cfg.n_heads + 1), dtype=torch.float32)

                # MLP
                ie_maps[(operator_idx, token_pos, seed)][:, -1] = activation_patching_experiment(model, correct_pa, 
                                                                                                corrupt_prompts_and_answers=corrupt_pa, hookpoint_name='mlp_post',
                                                                                                metric='IE', token_pos=token_pos, random_seed=seed).mean(dim=0)
                # Attention heads
                for head_idx in range(model.cfg.n_heads):
                    head_hook_fn = partial(head_hooking_func, head_index=head_idx)
                    ie_maps[(operator_idx, token_pos, seed)][:, head_idx] = activation_patching_experiment(model, correct_pa, 
                                                                                                        corrupt_prompts_and_answers=corrupt_pa, hookpoint_name='z',
                                                                                                        metric='IE', token_pos=token_pos, 
                                                                                                        hook_func_overload=head_hook_fn, random_seed=seed).mean(dim=0)
                # Save the results after each calculation to avoid losing them
                torch.save(ie_maps, results_file_path)


def node_attr_patching_localization(model_name, model_path):
    """
    Find the arithmetic circuit using node attribution patching on each component (head or MLP) of the model.
    Faster than activation patching, but less accurate.
    """
    # Load the model into CPU because backward pass in the GPU takes up too much memory
    model = load_model(model_name, model_path, 'cpu')
    max_op = 300
    analysis_prompts_file_path = fr'./data/{model_name}/large_prompts_and_answers_max_op={max_op}.pkl'
    set_deterministic(42)

    # Load the pre-calculated prompts and answers
    with open(analysis_prompts_file_path, 'rb') as f:
        large_prompts_and_answers = pickle.load(f)
    for i in range(len(large_prompts_and_answers)):
        random.shuffle(large_prompts_and_answers[i])
    correct_prompts_and_answers = [pa[:50] for pa in large_prompts_and_answers]

    results_file_path = f'./data/{model_name}/node_attribution_results.pt'
    if os.path.exists(results_file_path):
        attribution_results = torch.load(results_file_path)
    else:
        attribution_results = {}

    seeds = [42, 412, 32879, 123]
    for operator_idx in range(len(OPERATORS)):
        prompts_and_answers = correct_prompts_and_answers[operator_idx]
        for seed in seeds:
            if (operator_idx, seed) in attribution_results:
                print(f"Found results file for {operator_idx=}, {seed=}")
                continue
            print(f"{operator_idx=}, {seed=}")
            set_deterministic(seed)
            corrupt_prompts_and_answers = random.sample(sum(correct_prompts_and_answers, []), len(prompts_and_answers))
            attribution_scores = node_attribution_patching(model, prompts_and_answers, corrupt_prompts_and_answers, metric='IE', batch_size=10)
            scores_tensor = torch.zeros((5, model.cfg.n_layers, model.cfg.n_heads + 1), dtype=torch.float32)
            for layer in range(model.cfg.n_layers):
                for head in range(model.cfg.n_heads):
                    scores_tensor[:, layer, head] = attribution_scores[f'blocks.{layer}.attn.hook_z'].mean(dim=0)[:, head].sum(dim=-1)
                scores_tensor[:, layer, -1] = attribution_scores[f'blocks.{layer}.mlp.hook_post'].mean(dim=0).sum(dim=-1)
            attribution_results[(operator_idx, seed)] = scores_tensor

            # Save the results after each calculation to avoid losing them
            torch.save(attribution_results, results_file_path)


def main():
    args = parse_args()
    if args.do_attribution:
        node_attr_patching_localization(args.model_name, args.model_path)
    else:
        manual_ap_localization(args.model_name, args.model_path)


if __name__ == '__main__':
    main()