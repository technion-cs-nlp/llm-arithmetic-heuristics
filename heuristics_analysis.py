from functools import partial
from typing import Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import transformer_lens as lens
import random
import re

from component import Component
from evaluation_utils import model_accuracy
from general_utils import set_deterministic
from prompt_generation import OPERATORS, separate_prompts_and_answers


@dataclass
class HeuristicKnockoutData:
    heuristic_name: str
    ablated_neurons: List[Tuple[int, int, float]]
    baseline_related: float
    baseline_unrelated: float
    ablated_related: float
    ablated_unrelated: float
    ablated_neuron_matching_score: float


def is_associated_heuristic(heuristic_name: str, prompt: str):
    """
    Returns if the heuristic type is relevant for the prompt.
    Some examples:
    - is_relevant_heuristic("result_1mod2", "2+3=") -> True
    - is_relevant_heuristic("op1_region_1_5", "2+3=") -> True
    - is_relevant_heuristic("op1_region_1_5", "6-3=") -> False
    """
     # Find the operator from the prompt
    operator = re.findall(rf"[\+\-\*\/]", prompt)[0]
    # Get the operands
    op1, op2 = prompt.split(operator)
    op1, op2 = int(op1), int(op2[:-1])

    # Find the measured value (Which value is checked against the heuristic)
    if heuristic_name.startswith("result"):
        measured_value = eval(f"{op1}{operator}{op2}")
    elif heuristic_name.startswith("op1"):
        measured_value = op1
    elif heuristic_name.startswith("op2"):
        measured_value = op2
    else:
        if "both_operands" in heuristic_name:
            if "mod" in heuristic_name:
                m, n = map(int, re.findall(r"\d+", heuristic_name.split('operands_')[1]))
                return op1 % n == m and op2 % n == m
            elif "region" in heuristic_name:
                region = tuple(map(int, heuristic_name.split('_')[3:]))
                return region[0] <= op1 < region[1] and region[0] <= op2 < region[1]
        # Unique type of heuristic, for example same_operand
        elif heuristic_name == "same_operand":
            return op1 == op2
        else:
            raise ValueError(f"Unknown heuristic type {heuristic_name}")
        
    # Check the different heuristic types
    if "mod" in heuristic_name:
        m, n = map(int, re.findall(r"\d+", heuristic_name.split('_')[1]))
        return measured_value % n == m
    elif "region" in heuristic_name:
        region = tuple(map(int, heuristic_name.split('_')[2:]))
        return region[0] <= measured_value <= region[1]
    elif "multi_value" in heuristic_name:
        multi_values = list(map(int, heuristic_name.split('=')[1].strip('[]').split(',')))
        return measured_value in multi_values
    elif "value" in heuristic_name:
        value = int(heuristic_name.split('_')[2])
        return measured_value == value
    elif "pattern" in heuristic_name:
        pattern = heuristic_name.split('_')[2]
        return re.match(f"^{pattern}$", str(measured_value).zfill(3)) is not None
    else:
        raise ValueError(f"Unknown heuristic type {heuristic_name}")


def get_relevant_prompts(heuristic_name: str, operator_idx: int, min_op: int, max_op: int):
    """
    Get a list of prompts which are relevant for the heuristic.
    For example, if the heuristic is "result_1mod2", the relevant prompts are all prompts where the result of the operation is 1 mod 2.
    """
    op = OPERATORS[operator_idx]
    if "result" in heuristic_name:
        if "multi_value" in heuristic_name:
            multi_values = list(map(int, heuristic_name.split('=')[1].strip('[]').split(',')))
            return [f"{x}{op}{y}=" for x in range(min_op, max_op) for y in range(min_op, max_op) if eval(f"{x}{op}{y}") in multi_values]
        elif "value" in heuristic_name:
            value = int(heuristic_name.split('_')[2])
            return [f"{x}{op}{y}=" for x in range(min_op, max_op) for y in range(min_op, max_op) if eval(f"{x}{op}{y}") == value]
        elif "region" in heuristic_name:
            region = tuple(map(int, heuristic_name.split('_')[2:]))
            return [f"{x}{op}{y}=" for x in range(min_op, max_op) for y in range(min_op, max_op) if region[0] <= eval(f"{x}{op}{y}") <= region[1]]
        elif "mod" in heuristic_name:
            m, n = map(int, re.findall(r"\d+", heuristic_name.split('_')[1]))
            return [f"{x}{op}{y}=" for x in range(min_op, max_op) for y in range(min_op, max_op) if eval(f"{x}{op}{y}") % n == m]
        elif "result_pattern" in heuristic_name:
            pattern = heuristic_name.split('_')[2]
            return [f"{x}{op}{y}=" for x in range(min_op, max_op) for y in range(min_op, max_op) if re.match(f"^{pattern}$", str(eval(f"{x}{op}{y}")).zfill(3)) is not None]
    elif heuristic_name.startswith("both"):
        if "mod" in heuristic_name:
            m, n = map(int, re.findall(r"\d+", heuristic_name.split('_')[2]))
            return [f"{x}{op}{y}=" for x in range(min_op, max_op) for y in range(min_op, max_op) if x % n == m and y % n == m]
        elif "region" in heuristic_name:
            region = tuple(map(int, heuristic_name.split('_')[3:]))
            return [f"{x}{op}{y}=" for x in range(min_op, max_op) for y in range(min_op, max_op) if region[0] <= x < region[1] and region[0] <= y < region[1]]
    elif heuristic_name.startswith("op"):
        op_index = int(heuristic_name[2])
        if "value" in heuristic_name:
            value = int(heuristic_name.split('_')[2])
            return [f"{value}{op}{y}=" if op_index == 1 else f"{y}{op}{value}=" for y in range(min_op, max_op)]
        elif "region" in heuristic_name:
            region = tuple(map(int, heuristic_name.split('_')[2:]))
            return [f"{x}{op}{y}=" for x in range(min_op, max_op) for y in range(min_op, max_op) if region[0] <= (x if op_index == 1 else y) <= region[1]]
        elif "mod" in heuristic_name:
            m, n = map(int, re.findall(r"\d+", heuristic_name.split('_')[1]))
            return [f"{x}{op}{y}=" for x in range(min_op, max_op) for y in range(min_op, max_op) if (x if op_index == 1 else y) % n == m]
        elif "pattern" in heuristic_name:
            pattern = heuristic_name.split('_')[2]
            return [f"{x}{op}{y}=" for x in range(min_op, max_op) for y in range(min_op, max_op) if re.match(f"^{pattern}$", str(x if op_index == 1 else y).zfill(3)) is not None]
    elif "same_operand" in heuristic_name:
        return [f"{x}{op}{x}=" for x in range(min_op, max_op)]
    else:
        raise ValueError(f"Unknown heuristic type {heuristic_name}")


def get_neurons_associated_with_prompt(prompt: str, heuristic_classes: Dict[str, List[Tuple[int, int, float]]]):
    """
    For a given input prompt, get a dictionary of {(layer, neuron): List[heuristic_name, score]}.
    The (layer, neuron) keys are neurons that belong to the heuristics which are assosicated with the prompt.
    """
    # Get a list of relevant heuristics which should apply to the prompt
    relevant_heuristics = [heuristic_name for heuristic_name in heuristic_classes if is_associated_heuristic(heuristic_name, prompt)]

    # Create a list of all neurons in each of the heuristics
    relevant_neurons = []
    for heuristic_name in relevant_heuristics:
        relevant_neurons += [(heuristic_name, layer, neuron, score) for layer, neuron, score in heuristic_classes[heuristic_name]]

    # Group the neurons by layer and neuron index
    unified_relevant_neurons = {}
    for h, l, n, s in relevant_neurons:
        unified_relevant_neurons.setdefault((l, n), []).append((h, s))

    return unified_relevant_neurons


def heuristic_class_knockout_experiment(heuristic_classes: Dict[str, List[Tuple[int, int, float]]],
                                        operator_idx: int, 
                                        large_prompts_and_answers: List[Tuple[str, str]],
                                        model: lens.HookedTransformer,
                                        min_op: int,
                                        max_op: int,
                                        max_single_token: int, 
                                        prompt_prefix: str="",
                                        metric_fn: Callable=model_accuracy, 
                                        heuristic_neuron_match_threshold: float=0.55,
                                        seed: int=42,
                                        verbose: bool=True):
    """
    Runs a heuristic class knockout experiment on the model.
    For each heuristic class, the neurons which belong to the class are ablated, and the model's performance is measured across a subset of CORRECT prompts which
    should be affected by this heuristic class, as well as a control group of prompts unrelated to this heuristic.

    Returns:
        List[HeuristicKnockoutData]: A list of HeuristicKnockoutData objects, each containing the results of the knockout experiment for a specific heuristic class.
                                     For a full list of the fields, see the HeuristicKnockoutData class.
    """
    set_deterministic(seed)
    heuristic_classes = {name: [(l, n, s) for (l, n, s) in layer_neuron_scores if s >= heuristic_neuron_match_threshold] for name, layer_neuron_scores in heuristic_classes.items()}

    def filter_legal_prompts(prompts, operator_idx):
        if operator_idx == 0 or operator_idx == 2:
            return [p for p in prompts if int(eval(p[:-1])) < max_single_token]
        elif operator_idx == 1:
            return [p for p in prompts if int(p.split('-')[0]) >= int(p.split('-')[1].split('=')[0])]
        elif operator_idx == 3:
            return [p for p in prompts if int(p.split('/')[0]) >= int(p.split('/')[1].split('=')[0]) and int(p.split('/')[1].split('=')[0]) != 0]
    
    def filter_correct_prompts(prompts, operator_idx):
        correct_prompts = separate_prompts_and_answers(large_prompts_and_answers[operator_idx])[0]
        return list(set(prompts).intersection(correct_prompts))

    heuristics_knockout_results = []
    heuristic_names_to_test = [(k, v) for (k, v) in sorted(heuristic_classes.items(), key=lambda kv: len(kv[1]), reverse=True) if len(v) > 0]
    heuristic_iterator = (tqdm(heuristic_names_to_test) if verbose else heuristic_names_to_test)
    for ablated_heuristic_name, ablated_neurons in heuristic_iterator:
        if len(ablated_neurons) < 10:
            break

        # Find the relevant prompts for the tested heuristic    
        related_prompts = get_relevant_prompts(ablated_heuristic_name, operator_idx, min_op, max_op)
        related_prompts = filter_legal_prompts(related_prompts, operator_idx)
        related_prompts = filter_correct_prompts(related_prompts, operator_idx)
        if len(related_prompts) > 100:
            related_prompts = random.sample(related_prompts, k=100)
        elif len(related_prompts) <= 0:
            print("Continuing")
            continue
        print(len(related_prompts))
    
        answers = [str(int(eval(prompt[:-1]))) for prompt in related_prompts]
        related_prompts = [f"{prompt_prefix}{p}" for p in related_prompts]

        ablated_neuron_matching_score = sum([s for l, n, s in ablated_neurons])

        # Create a list of neurons to be mean ablated for the tested heuristic
        def hook_ablate_neurons(value, hook, hook_comp):
            value[:, :, hook_comp.neuron_indices] = 0
            return value

        hooked_mlp_components = []
        for layer in set([l for l, _, _ in ablated_neurons]):
            hooked_mlp_components.append(Component('mlp_post', layer=layer, neurons=[n for l, n, _ in ablated_neurons if l == layer]))
        hooks = [(comp.valid_hook_name(), partial(hook_ablate_neurons, hook_comp=comp)) for comp in hooked_mlp_components]

        baseline_related = 1.0#metric_fn(model, related_prompts, answers, None, verbose=False)
        ablated_related = metric_fn(model, related_prompts, answers, hooks, verbose=False)

        unrelated_prompts = list(set([f"{x}{OPERATORS[operator_idx]}{y}=" for x in range(max_op) for y in range(max_op)]) - set(related_prompts))
        unrelated_prompts = filter_legal_prompts(unrelated_prompts, operator_idx)
        unrelated_prompts = filter_correct_prompts(unrelated_prompts, operator_idx)
        unrelated_prompts = random.sample(unrelated_prompts, k=len(related_prompts))
        unrelated_answers = [str(int(eval(prompt[:-1]))) for prompt in unrelated_prompts]
        unrelated_prompts = [f"{prompt_prefix}{p}" for p in unrelated_prompts]
        baseline_unrelated = 1.0#metric_fn(model, unrelated_prompts, unrelated_answers, None, verbose=False)
        ablated_unrelated = metric_fn(model, unrelated_prompts, unrelated_answers, hooks, verbose=False)

        if verbose:
            print(f"Heuristic {ablated_heuristic_name} ({len(ablated_neurons)} neurons): Related prompts ({len(related_prompts)=})): Pre/Post:{baseline_related:.3f}/{ablated_related:.3f}; Unrelated prompts: Pre/Post:{baseline_unrelated:.3f}/{ablated_unrelated:.3f};")

        heuristic_knockout_data = HeuristicKnockoutData(ablated_heuristic_name, ablated_neurons, baseline_related, baseline_unrelated, ablated_related, ablated_unrelated, ablated_neuron_matching_score)
        heuristics_knockout_results.append(heuristic_knockout_data)

    return heuristics_knockout_results


def prompt_knockout_experiment(heuristic_classes: Dict[str, List[Tuple[int, int, float]]],
                               model: lens.HookedTransformer, 
                               prompts_and_answers: List[Tuple[str, str]],
                               all_top_neurons: List[Tuple[int, int]],
                               neuron_count_hard_limit_per_layer: int,
                               metric_fn: Callable = model_accuracy):
    """
    Runs a prompt knockout experiment on the model.
    In this experiment, we ablate a specific number of neurons (per layer) that have the highest matching score
    with heuristics that are associated with a specific prompt. A metric (usually accuracy) is then observed 
    for the model before and after the ablation. A control ablation is also done, where we ablate the same amount of random 
    heuristic neurons for each prompt.
    """
    prompts, answers = separate_prompts_and_answers(prompts_and_answers)
    baseline_metric = metric_fn(model, prompts, answers, verbose=False)
    ablated_metric = 0.0
    avg_relevant_neurons = 0.0
    control_metric = 0.0

    for i, (prompt, answer) in enumerate(prompts_and_answers):
        relevant_neurons = get_neurons_associated_with_prompt(prompt, heuristic_classes)

        neurons_to_scores = {(layer, neuron): len([s for _, s in scores_list]) for (layer, neuron), scores_list in relevant_neurons.items()}
        neurons_to_scores = sorted(neurons_to_scores.items(), key=lambda ln_s: ln_s[1], reverse=True)
        layers = set([l for l, _ in all_top_neurons])
        relevant_neurons = {l: [neuron for ((layer, neuron), _) in neurons_to_scores if l == layer][:neuron_count_hard_limit_per_layer] for l in layers}

        avg_relevant_neurons += sum([len(neurons) for neurons in relevant_neurons.values()])
        hooked_components = [Component('mlp_post', layer=layer, neurons=relevant_neurons[layer]) for layer in layers]

        def hook_ablate_neurons(value, hook, hook_comp: Component):
            value[:, :, hook_comp.neuron_indices] = 0
            return value
        hooks = [(comp.valid_hook_name(), partial(hook_ablate_neurons, hook_comp=comp)) for comp in hooked_components]
        ablated_metric += metric_fn(model, [prompt], [answer], hooks=hooks, verbose=False)


        control_hooked_components = []
        for comp in hooked_components:
            layer_neurons = set([n for (l, n) in all_top_neurons if l == comp.layer])
            # assert len(layer_neurons) == 200
            random_neurons = random.sample(list(layer_neurons - set(comp.neuron_indices)), k=len(comp.neuron_indices))
            control_hooked_components.append(Component('mlp_post', layer=comp.layer, neurons=random_neurons))        
        control_hooks = [(control_comp.valid_hook_name(), partial(hook_ablate_neurons, hook_comp=control_comp)) for control_comp in control_hooked_components]
        control_metric += metric_fn(model, [prompt], [answer], hooks=control_hooks, verbose=False)
    
    ablated_metric /= len(prompts)
    avg_relevant_neurons /= len(prompts)
    control_metric /= len(prompts)

    return baseline_metric, ablated_metric, avg_relevant_neurons, control_metric
