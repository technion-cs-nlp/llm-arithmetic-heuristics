import copy
import torch
import transformer_lens as lens
from typing import List, Tuple, Dict, Set
from tqdm import tqdm
from functools import partial
from transformer_lens.hook_points import HookPoint
from circuit import Circuit
from component import Component
from metrics import logit_diff
from general_utils import get_hook_dim
from prompt_generation import separate_prompts_and_answers, generate_all_prompts_for_operator


def circuit_faithfulness_with_mean_ablation(model: lens.HookedTransformer, 
                                            circuit: Circuit, 
                                            prompts_and_answers: List[Tuple[str, str]],
                                            mean_cache: Dict[Component, torch.Tensor],
                                            metric='nl'):
    """
    Calculate the faithfulness of the circuit w.r.t to the entire model. Non-circuit components are mean ablated.
    The faithfulness is normalized using two baselines - a "good" one where no components are mean ablated, and a "bad" one where all components are mean ablated.

    Args:
        model (lens.HookedTransformer): The model to evaluate.
        circuit (Circuit): The circuit to evaluate.
        prompts_and_answers (list(tuple(str, str))): A list of (prompt, answer) pairs to use for evaluation.
        mean_cache (Dict[Component, torch.Tensor]): The mean ablation cache to use.
        metric (str, optional): The metric to use for faithfulness. Can be one of 'logits', 'probs', 'nl', 'ce'. Defaults to 'nl'.
                    Each metric is calculated for the ablated pass as well as the two baselines passes.
                    The metrics are:
                    'logits' - The logit of the correct answer token.
                    'probs' - Probability of the correct answer token.
                    'nl' (normalized logit) - The logit of the correct answer token, divided by the maximal logit. 
                                              This is a continous proxy for the accuracy of the model. Used in the paper.
                    'ce' - Cross-entropy loss of the model. Used for debugging.

    """
    assert_valid_evaluation_hooks(circuit.components)
    prompts, answers = separate_prompts_and_answers(prompts_and_answers)
    answer_tokens = model.to_tokens(answers, prepend_bos=False).view(-1)

    good_baseline_logits = model(prompts)[:, -1] # For the "good" baseline, no components are mean ablated, so its a simple forward pass

    ablated_logits = run_with_mean_ablation(model, prompts, mean_cache, circuit.components, reverse_ablation=True) # All non-circuit componetns are mean ablated
    bad_baseline_logits = run_with_mean_ablation(model, prompts, mean_cache, [], reverse_ablation=True) # All components are mean ablated

    if metric == 'logits':
        good_baseline_correct_logits = good_baseline_logits.gather(1, answer_tokens.view(-1, 1))
        bad_baseline_correct_logits = bad_baseline_logits.gather(1, answer_tokens.view(-1, 1))
        ablated_correct_logits = ablated_logits.gather(1, answer_tokens.view(-1, 1))
        return ((ablated_correct_logits - bad_baseline_correct_logits) / (good_baseline_correct_logits - bad_baseline_correct_logits)).mean()
    elif metric == 'probs':
        good_baseline_correct_probs = good_baseline_logits.softmax(dim=-1).gather(1, answer_tokens.view(-1, 1))
        bad_baseline_correct_probs = bad_baseline_logits.softmax(dim=-1).gather(1, answer_tokens.view(-1, 1))
        ablated_correct_probs = ablated_logits.softmax(dim=-1).gather(1, answer_tokens.view(-1, 1))
        return ((ablated_correct_probs - bad_baseline_correct_probs) / (good_baseline_correct_probs - bad_baseline_correct_probs)).mean()
    elif metric == 'nl':
        max_val = good_baseline_logits.max(dim=-1).values.view(-1, 1)
        good_baseline_normalized_correct_logits = ((good_baseline_logits) / max_val).gather(1, answer_tokens.view(-1, 1).to(max_val.device))

        max_val = bad_baseline_logits.max(dim=-1).values.view(-1, 1)
        bad_baseline_normalized_correct_logits = ((bad_baseline_logits) / max_val).gather(1, answer_tokens.view(-1, 1).to(max_val.device))

        max_val = ablated_logits.max(dim=-1).values.view(-1, 1)
        ablated_normalized_correct_logits = ((ablated_logits) / max_val).gather(1, answer_tokens.view(-1, 1).to(max_val.device))
        return ((ablated_normalized_correct_logits - bad_baseline_normalized_correct_logits) / (good_baseline_normalized_correct_logits - bad_baseline_normalized_correct_logits)).mean()
    elif metric == 'ce':
        ce = torch.nn.CrossEntropyLoss()
        good_baseline_ce = -ce(good_baseline_logits, answer_tokens)
        bad_baseline_ce = -ce(bad_baseline_logits, answer_tokens)
        ablated_ce = -ce(ablated_logits, answer_tokens)
        return (ablated_ce - bad_baseline_ce) / (good_baseline_ce - bad_baseline_ce)
    else:
        raise ValueError(f"Unknown metric {metric}")


def circuit_faithfulness_with_corrupt_prompts(model: lens.HookedTransformer,
                     circuit: Circuit,
                     prompts_and_answers: List[Tuple[str, str]],
                     corrupt_prompts_and_answers: List[Tuple[str, str]],
                     metric='ld'):
    """
    Calculate the faithfulness of the circuit w.r.t to the entire model. 
    Non-circuit components are ablated using their activations for counterfactual (corrupt) prompts.
    The faithfulness is normalized using two baselines - a "good" one where no components are mean ablated, and a "bad" one where all components are mean ablated.

    Args:
        model (lens.HookedTransformer): The model to evaluate.
        circuit (Circuit): The circuit to evaluate.
        prompts_and_answers (list(tuple(str, str))): A list of (prompt, answer) pairs to use for evaluation.
        corrupt_prompts_and_answers (list(tuple(str, str))): A list of counterfactual (prompt, answer) pairs to use for evaluation.
        metric (str, optional): The metric to use for faithfulness. Can be one of 'ld', 'nld'. Defaults to 'ld'.
                    Each metric is calculated for the ablated pass as well as the two baselines passes.
                    The metrics are:
                    'ld' (logit difference) - The difference between the logit of the correct answer and the counterfactual answer.
                    'nl' (normalized logit) - The logit of the correct answer token, divided by the maximal logit.
    """
    assert_valid_evaluation_hooks(circuit.components)
    prompts, answers = separate_prompts_and_answers(prompts_and_answers)
    corrupt_prompts, corrupt_answers = separate_prompts_and_answers(corrupt_prompts_and_answers)

    corrupt_logits, corrupt_cache = model.run_with_cache(corrupt_prompts, return_type='logits')
    corrupt_logits = corrupt_logits[:, -1, :]
    clean_logits = model(prompts)[:, -1, :]
    clean_labels = model.to_tokens(answers, prepend_bos=False)
    corrupt_labels = model.to_tokens(corrupt_answers, prepend_bos=False)

    # Convert the clean_cache to a dict        
    real_cache = {}
    for layer in range(model.cfg.n_layers):
        for head_idx in [None] + list(range(model.cfg.n_heads)):
            hook_name = 'mlp_post' if head_idx is None else 'z'
            if head_idx is None:
                real_cache[Component(hook_name, layer)] = corrupt_cache[lens.utils.get_act_name(hook_name, layer)]
            else:
                real_cache[Component(hook_name, layer, head=head_idx)] = corrupt_cache[lens.utils.get_act_name(hook_name, layer)][:, :, head_idx]
    del corrupt_cache

    circuit_logits = run_with_mean_ablation(model, prompts, real_cache, circuit.components, reverse_ablation=True) # THIS ISNT REALLY MEAN ABLATION - THE ARGUMENTS MAKE IT AS DESCRIBED ABOVE

    if metric == 'nl':
        clean_baseline_nl = ((clean_logits) / (clean_logits.max(dim=-1).values.view(-1, 1))).gather(1, clean_labels.view(-1, 1))
        corrupt_baseline_nl = ((corrupt_logits) / (corrupt_logits.max(dim=-1).values.view(-1, 1))).gather(1, clean_labels.view(-1, 1))
        circuit_nl = ((circuit_logits) / (circuit_logits.max(dim=-1).values.view(-1, 1))).gather(1, clean_labels.view(-1, 1))
        return ((circuit_nl - corrupt_baseline_nl) / (clean_baseline_nl - corrupt_baseline_nl)).mean()
    elif metric == 'ld':
        clean_baseline_ld = torch.mean(logit_diff(clean_logits, clean_labels, corrupt_labels)) # just run on clean prompts for the entire model (nothing is patched out)
        corrupt_baseline_ld = torch.mean(logit_diff(corrupt_logits, clean_labels, corrupt_labels)) # All corrupt activations
        circuit_ld = torch.mean(logit_diff(circuit_logits, clean_labels, corrupt_labels))
        return ((circuit_ld - corrupt_baseline_ld) / (clean_baseline_ld - corrupt_baseline_ld)).mean()
    else:
        raise ValueError(f"Unknown metric {metric}")


def circuit_accuracy_with_mean_ablation(model: lens.HookedTransformer, 
                                        circuit: Circuit, 
                                        prompts_and_answers: List[Tuple[str, str]],
                                        mean_cache: Dict[Component, torch.Tensor]):
    """
    Calculate the accuracy of the model with a given circuit, using mean ablation to knock out non-circuit components.
    This is a harder (more difficult) metric to nail than faithfulness, as it requires the model to be able to produce perfect
    completions even when many components are out-of-distribution.

    Args:
        model (lens.HookedTransformer): The model.
        circuit (Circuit): The circuit to evaluate.
        prompts_and_answers (list(tuple(str, str))): A list of (prompt, answer) pairs.
        mean_cache (Dict[Component, torch.Tensor]): The mean ablation cache to use.
    """
    assert_valid_evaluation_hooks(circuit.components)
    prompts, answers = separate_prompts_and_answers(prompts_and_answers)
    answer_tokens = model.to_tokens(answers, prepend_bos=False).view(-1)

    good_baseline_logits = model(prompts)[:, -1] # No components are mean ablated
    ablated_logits = run_with_mean_ablation(model, prompts, mean_cache, circuit.components, reverse_ablation=True) # All non-circuit componetns are mean ablated
    bad_baseline_logits = run_with_mean_ablation(model, prompts, mean_cache, [], reverse_ablation=True) # All components are mean ablated

    good_baseline_accuracy = (good_baseline_logits.argmax(dim=-1) == answer_tokens).to(dtype=torch.float32).mean()
    bad_baseline_accuracy = (bad_baseline_logits.argmax(dim=-1) == answer_tokens).to(dtype=torch.float32).mean()
    ablated_model_accuracy = (ablated_logits.argmax(dim=-1) == answer_tokens).to(dtype=torch.float32).mean()

    return ((ablated_model_accuracy - bad_baseline_accuracy) / (good_baseline_accuracy - bad_baseline_accuracy))


def get_subgroup_for_minimality(model: lens.HookedTransformer,
                                circuit: Circuit,
                                component: Component, 
                                component_group: List[Component],
                                prompts_and_answers: List[Tuple[str, str]],
                                mean_cache: Dict[Component, torch.Tensor],
                                percentage: float = 0.3):
    """
    Find the top percentage of components from a component group, which are the most different 
    in terms of role from a given component.
    This is measured in terms of logit difference when the component is removed from the circuit,
    versus when the component and the other component in the group are removed from the circuit.

    For more information, see Appendix B in https://openreview.net/pdf?id=8sKcAWOf2D.

    Args:
        model (lens.HookedTransformer): The model to evaluate.
        circuit (Circuit): The circuit to evaluate.
        component (Component): The component to find the minimality subgrop for.
        component_group (List[Component]): The components to compare against.
        prompts_and_answers (list(tuple(str, str))): A list of (prompt, answer) pairs.
        mean_cache (Dict[Component, torch.Tensor]): The mean ablation cache to use.
        percentage (float, optional): The percentage of components to return.

    Returns:

    """
    # Following the algorithm presented in https://openreview.net/pdf?id=8sKcAWOf2D
    assert component in component_group, f'Component {component} should be a part of the component group from which a minimal set is to be found'

    prompts, answers = separate_prompts_and_answers(prompts_and_answers)
    answer_tokens = model.to_tokens(answers, prepend_bos=False)
    
    # Rank all other components in the group based on the difference of removing them only vs removing them and the component on circuit performance  
    other_comps_diff_ranking = {}
    for other_comp in list(set(component_group) - {component}):
        circuit_minus_other = list(set(circuit.components) - {other_comp})
        circuit_minus_other_and_comp = list(set(circuit.components) - {other_comp, component})
        other_removed_score = run_with_mean_ablation(model, prompts, mean_cache, circuit_minus_other).gather(1, answer_tokens.view(-1, 1))
        other_and_comp_removed_score = run_with_mean_ablation(model, prompts, mean_cache, circuit_minus_other_and_comp).gather(1, answer_tokens.view(-1, 1))
        other_comps_diff_ranking[other_comp] = (other_removed_score - other_and_comp_removed_score).abs().mean()

    other_comps_ranked_by_diff = sorted(other_comps_diff_ranking.items(), key=lambda x: x[1], reverse=True) 
    other_comps_diff_ranking = [c for (c, diff) in other_comps_ranked_by_diff]
    return other_comps_diff_ranking[:int(len(other_comps_diff_ranking) * percentage)]


def circuit_minimality_with_mean_ablation(model: lens.HookedTransformer, 
                                          circuit: Circuit,
                                          component_groups: List[List[Component]],
                                          prompts_and_answers: List[Tuple[str, str]],
                                          mean_cache: Dict[Component, torch.Tensor]):
    """
    Calculate the minimality score for each component in the circuit for a given set of prompts, 
    using mean ablation to knock out non-circuit components.
    The minimality score is calculated separately for each node, and is equal to - 
        (G(C\K) - G(C\{K u {v}})) / G(C\{K u {v}})
    where G(X) is the logit of the correct answer when the correct prompt is given to X and the rest of
    the components in the model are mean ablated.

    A higher score is better, and it indicates there is less redundancy between a component and the rest
    of the components in its component group.
    
    Args:
        model (lens.HookedTransformer): The model to evaluate.
        circuit (Circuit): The circuit to evaluate.
        component_groups (List[List[Component]]): The components to evaluate, separated into groups based on a shared role.
        prompts_and_answers (list(tuple(str, str))): A list of (prompt, answer) pairs.
        mean_cache (Dict[Component, torch.Tensor]): The mean ablation cache to use.
    Returns:
        Dict[Component, float]: A dictionary of minimality scores for each component in the circuit.
    """
    assert_valid_evaluation_hooks(circuit.components)
    prompts, answers = separate_prompts_and_answers(prompts_and_answers)
    answer_tokens = model.to_tokens(answers, prepend_bos=False)
    minimality_score_per_component = {}

    for comp_group in component_groups:
        for comp in comp_group:
            if len(comp_group) < 10: # Threshold magic number, was WIP, can be changed
                comp_subgroup = set(comp_group) - {comp}
            else:
                comp_subgroup = set(get_subgroup_for_minimality(model, circuit, comp, comp_group, prompts_and_answers, mean_cache))

            circuit_minus_subgroup = set(circuit.components) - comp_subgroup
            circuit_minus_subgroup_and_comp = circuit_minus_subgroup - {comp}
            score_circuit_exclude_subgroup = run_with_mean_ablation(model, prompts, mean_cache, circuit_minus_subgroup).gather(1, answer_tokens.view(-1, 1))
            score_circuit_exclude_subgroup_and_comp = run_with_mean_ablation(model, prompts, mean_cache, circuit_minus_subgroup_and_comp).gather(1, answer_tokens.view(-1, 1))
            
            minimality_score_per_component[comp] = ((score_circuit_exclude_subgroup - score_circuit_exclude_subgroup_and_comp) / score_circuit_exclude_subgroup_and_comp).mean().item()

    return minimality_score_per_component


def run_with_mean_ablation(model: lens.HookedTransformer,
                           prompts: List[Tuple[str, str]],
                           mean_cache: Dict[Component, torch.Tensor],
                           components: List[Component],
                           reverse_ablation: bool = True):
    """
    Run a forward pass through the model where a set of components is ablated with pre-calculated mean activations.

    Args:
        model (lens.HookedTransformer): The model to analyze.
        prompts (List[Tuple[str, str]]): The prompts to use for the forward pass.
        mean_cache (Dict[Component, torch.Tensor]): The mean activations cache. Each key is a component, and the value is the mean activation tensor.
        components (List[Component]): The components to mean ablate (or not mean ablate if reverse_ablation is True).
        reverse_ablation (bool, optional): If True, the components are the ones to keep and not mean ablate. Else, the components are mean ablated.
    Returns:
        torch.Tensor(batch, d_model): The output logits of the last position of the ablated model.
    """
    def hook_ablate_from_mean_cache(value, hook, hooked_comp=None):
        if hooked_comp.head_idx is None and hooked_comp.neuron_indices is None:
            # An entire MLP
            value = mean_cache[hooked_comp].to(value.device, value.dtype)
        elif hooked_comp.head_idx is None and hooked_comp.neuron_indices is not None:
            # MLP with specific neurons
            comp_key = Component(hooked_comp.hook_name, layer=hooked_comp.layer)
            value[:, :, hooked_comp.neuron_indices] = mean_cache[comp_key][:, :, hooked_comp.neuron_indices].to(value.device, value.dtype)
        elif hooked_comp.head_idx is not None and hooked_comp.neuron_indices is None:
            # Attention head with all neurons
            value[:, :, hooked_comp.head_idx, :] = mean_cache[hooked_comp].to(value.device, value.dtype)
        else:
            # Attention head with specific neurons
            comp_key = Component(hooked_comp.hook_name, layer=hooked_comp.layer, head=hooked_comp.head_idx)
            value[:, :, hooked_comp.head_idx, hooked_comp.neuron_indices] = mean_cache[comp_key][:, :, hooked_comp.neuron_indices].to(value.device, value.dtype)
        return value
    
    if not reverse_ablation:
        # Only the list of above components is ablated
        mean_ablation_hooks = [(comp.valid_hook_name(), partial(hook_ablate_from_mean_cache, hooked_comp=comp)) for comp in components]
    else:
        # All components EXCEPT the list of given components are ablated
        mean_ablation_hooks = []

        # For each layer, we check for each of its components (the full MLP and each attention head) if its in the circuit.
        # If the full version is in the circuit, we don't do anything (it should be mean ablated). Otherwise, we check if 
        # specific neurons are in the circuit, in which case we mean ablate the rest of the neurons. If the component is not 
        # in the circuit at all, we mean ablate the entire component.
        for layer in range(model.cfg.n_layers):
            full_mlp_comp = Component('mlp_post', layer=layer)
            if full_mlp_comp not in components:
                # The entire MLP isnt part of the circuit; Some specific neurons in it might still be
                mlp_layer_comps = [c for c in components if c.layer == layer and c.head_idx is None and c.valid_hook_name() == lens.utils.get_act_name('mlp_post', layer=c.layer)]
                if len(mlp_layer_comps) == 0:
                    # No component with specific MLP neurons are part of the circuit - The entire MLP should be mean ablated
                    mlp_comp = full_mlp_comp
                else:
                    # Specific neurons are part of the circuit; Combine all the supplied specific neurons for this MLP - and the rest are mean ablated
                    layer_neuron_indices = [c.neuron_indices for c in mlp_layer_comps]
                    circuit_neuron_indices = set(sum(layer_neuron_indices, start=()))
                    mean_ablated_neuron_indices = list(set(range(get_hook_dim(model, full_mlp_comp.hook_name))) - circuit_neuron_indices)
                    mlp_comp = Component(full_mlp_comp.hook_name, layer=layer, neurons=mean_ablated_neuron_indices)
                mean_ablation_hooks.append((mlp_comp.valid_hook_name(), partial(hook_ablate_from_mean_cache, hooked_comp=mlp_comp)))


            for head in range(model.cfg.n_heads):
                full_head_comp = Component('z', layer=layer, head=head)
                if full_head_comp not in components:
                    head_layer_comps = [c for c in components if c.layer == layer and c.head_idx == head and c.valid_hook_name() == lens.utils.get_act_name('z', layer=c.layer)]
                    if len(head_layer_comps) == 0:
                        # No head with specific neurons are part of the circuit - The entire MLP should be mean ablated
                        head_comp = full_head_comp
                    else:
                        # Specific neurons are part of the circuit; Combine all the supplied specific neurons for this attention head - and the rest are mean ablated
                        layer_neuron_indices = [c.neuron_indices for c in head_layer_comps]
                        circuit_neuron_indices = set(sum(layer_neuron_indices, start=()))
                        mean_ablated_neuron_indices = list(set(range(get_hook_dim(model, full_head_comp.hook_name))) - circuit_neuron_indices)
                        head_comp = Component(full_head_comp.hook_name, layer=layer, head=head, neurons=mean_ablated_neuron_indices)
                    mean_ablation_hooks.append((head_comp.valid_hook_name(), partial(hook_ablate_from_mean_cache, hooked_comp=head_comp)))
    
    # Get the logits after ablating the components
    ablated_logits = model.run_with_hooks(prompts, fwd_hooks=mean_ablation_hooks)[:, -1]
    return ablated_logits

    
def assert_valid_evaluation_hooks(components):
    for c in components:
        assert (c.head_idx is None and (c.hook_name == 'mlp_post' or c.hook_name == 'mlp_in')) or (c.head_idx is not None and c.hook_name == 'z'), \
            f"To evaluate a circuit, all components must hook mlp_post/mlp_in or z hooks only! Found component {c}"


def model_accuracy_on_simple_prompts(model: lens.HookedTransformer,
                                     min_op: int,
                                     max_op: int,
                                     single_token_number_range: Tuple[int, int],
                                     operators: List[str] = None,
                                     hooks: List[Tuple[str, HookPoint]] = None) -> float:
    """
    Measure the model's accuracy (NOT FAITHFULNESS! Simple accuracy) on simple prompts (2 operands with one operator).
    Args:
        model (lens.HookedTransformer): The model to measure the accuracy of.
        min_op (int): The minimal operand value to measure accuracy for.
        max_op (int): The maximum operand value to measure accuracy for.
        single_token_number_range (Tuple[int, int]): The range of single token numbers to limit the prompts on.
        operators (List[str], optional): The operators to measure the accuracy on. If None, all 4 operators are used.
        single_token_number_range
        hooks (list[(str, Callable)]): A list of hooks to use for the model. If None, no hooks are used. (Default is None).
    Returns:
        float: The model accuracy.
    """
    if operators is None:
        operators = ['+', '-', '*', '/']
    
    prompts = []
    for operator in operators:
        prompts.extend(generate_all_prompts_for_operator(operator, min_op, max_op, single_token_number_range))
    answers = [str(int(eval(p[:-1]))) for p in prompts]
    return model_accuracy(model, prompts, answers, hooks, verbose=False)
    

def model_accuracy(model: lens.HookedTransformer,
                   prompts: List[str],
                   answers: List[str],
                   hooks: List[Tuple[str, HookPoint]] = None,
                   verbose: bool = True) -> float:
    """
    Measure the model's accuracy on a set of prompts.

    Args:
        model (lens.HookedTransformer): The model to measure the accuracy of.
        prompts (List[str]): The prompts to measure the accuracy on.
        answers (List[str]): The answers to the prompts.
        hooks (list[(str, Callable)]): A list of hooks to use while calculating the accuracy on the prompts. If None, no hooks are used. (Default is None)
        verbose (bool): If True, print the wrong prompts as well as a progress bar. (Default is True)
    """
    count_correct = 0
    try:
        if hooks is not None:
            for h in hooks:
                model.add_hook(h[0], h[1])

        prompt_loader = torch.utils.data.DataLoader(list(zip(prompts, answers)), batch_size=32, shuffle=False)
        loader = tqdm(prompt_loader) if verbose else prompt_loader
        for batch in loader:
            prompts_batch = list(batch[0])
            answers_batch = model.to_tokens(list(batch[1]), prepend_bos=False).squeeze(1)
            logits = model(prompts_batch)
            preds = logits[:, -1, :].argmax(-1)
            count_correct += (preds.cpu() == answers_batch.cpu()).sum().item()
            if verbose:
                print(f"Wrong prompts: {[prompts_batch[i] for i in (preds.cpu() != answers_batch.cpu()).nonzero()]}:")
    finally:
        model.remove_all_hook_fns()

    return count_correct / len(prompts)
