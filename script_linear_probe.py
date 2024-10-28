import argparse
import random
import pickle
import os
import torch
import logging
from prompt_generation import separate_prompts_and_answers, OPERATORS
from component import Component
from general_utils import generate_activations, load_model, get_model_consts
from linear_probing import linear_probe_across_layers


torch.set_grad_enabled(False)
device = 'cuda'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Name of the model to be loaded')
    parser.add_argument('--model_path', type=str, help='Path to the model to be loaded')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model_name, model_path = args.model_name, args.model_path
    logging.info("Loading model")
    model = load_model(model_name, model_path, device)

    max_op = 300
    model_consts = get_model_consts(model_name)
    max_answer_value = model_consts.max_single_token
    large_prompts_and_answers = pickle.load(open(fr'./data/{model_name}/large_prompts_and_answers_max_op={max_op}.pkl', 'rb'))

    results_path = f"./data/{model_name}/probe_accs.pt"
    if os.path.exists(results_path):
        probe_accs = torch.load(results_path)
    else:
        probe_accs = {}

    for operator_idx in range(len(OPERATORS)):
        activations = None
        for pos_to_probe in [4, 3, 2, 1]:
            if (operator_idx, pos_to_probe) in probe_accs:
                print(f"Found results file for {operator_idx=}, {pos_to_probe=}, ")
                continue
            print(f"{operator_idx=}, {pos_to_probe=}")
            components = [Component('resid_post', layer=i) for i in range(model.cfg.n_layers)]
            correct_prompts = separate_prompts_and_answers(large_prompts_and_answers[operator_idx])[0]
            random.shuffle(correct_prompts)
            if activations is None:
                activations = generate_activations(model, correct_prompts, components, pos=None)
            pos_activations = {i: activations[i][:, pos_to_probe] for i in range(model.cfg.n_layers)}
            answers = torch.tensor([int(eval(prompt[:-1])) for prompt in correct_prompts])
            probe_accs[(operator_idx, pos_to_probe)] = linear_probe_across_layers(model, pos_activations, answers, max_answer_value)[1] # [1] to get only test accs
            torch.save(probe_accs, results_path)


if __name__ == '__main__':
    main()