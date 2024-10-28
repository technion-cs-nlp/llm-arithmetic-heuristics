import copy
from dataclasses import dataclass
import os
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from prompt_generation import OPERATOR_NAMES, OPERATORS
from general_utils import most_significant_wildcard_patterns, safe_eval
import torch


@dataclass
class HeuristicAnalysisData:
    """
    Various pre-calculated data structures that contain required data for heuristic analysis
    """
    also_check_bottom_results: bool = None
    op1_op2_pairs: List = None
    top_op1_op2_indices: List  = None
    top_results: List  = None
    max_op: int = None
    max_single_token: int = None
    operator_idx: int = None
    k_per_heuristic_cache: Dict = None

OPERAND_range_SIZES_BY_OPERATOR = {
    '+': [10, 30, 50, 100],
    '-': [10, 30, 50, 100],
    '*': [3, 5, 10],
    '/': [10, 30, 50, 100]
}

RESULT_range_SIZES_BY_OPERATOR = {
    '+': [10, 30, 50],
    '-': [10, 30, 50],
    '*': [10, 30, 50, 100],
    '/': [2, 10, 100]
}

###
### Heuristic Classification helper functions 
###
def insert(heuristic_matches_dict, name, layer_neuron_score):
    """
    Insert a new heuristic match into the heuristic_matches_dict.
    Initializes the list for the heuristic if it doesn't exist.
    """
    if name not in heuristic_matches_dict:
        heuristic_matches_dict[name] = []
    heuristic_matches_dict[name].append(layer_neuron_score)


def analyzed_operand_pairs_factor(heuristic_analysis_data: HeuristicAnalysisData):
    """
    Get the ratio of the number of analyzed operand pairs to the total number of possible operand pairs 
    (i.e. how many legal prompts are there for the current operator).
    For example, for addition it is 1.0 (all prompts are valid), for subtraction it is 0.5 (op1 has to be larger than op2).
    """
    return len(heuristic_analysis_data.op1_op2_pairs) / (heuristic_analysis_data.max_op ** 2)


def is_operand_m_mod_n_neuron(layer: int, neuron_idx: int, m: int, n: int, heuristic_analysis_data: HeuristicAnalysisData, op_index=1):
    """
    Checks if a specific neuron implements a mod heuristic on one of the operands.
    """
    k = int((heuristic_analysis_data.max_op // n) * heuristic_analysis_data.max_op * analyzed_operand_pairs_factor(heuristic_analysis_data))
    topk_op1_op2_indices = heuristic_analysis_data.top_op1_op2_indices[(layer, neuron_idx)][:k]
    m_mod_n_percentage = sum([1 for (op1, op2) in topk_op1_op2_indices if (op1 if op_index == 1 else op2) % n == m]) / k
    return m_mod_n_percentage


def is_both_operands_m_mod_n_neuron(layer: int, neuron_idx: int, m: int, n: int, heuristic_analysis_data: HeuristicAnalysisData):
    """
    Checks if a specific neuron implements the same mod heuristic on both of its operands.
    """
    k = int((heuristic_analysis_data.max_op // n) ** 2 * analyzed_operand_pairs_factor(heuristic_analysis_data))
    topk_op1_op2_indices = heuristic_analysis_data.top_op1_op2_indices[(layer, neuron_idx)][:k]
    both_operands_m_mod_n_percentage = sum([op1 % n == m and op2 % n == m for (op1, op2) in topk_op1_op2_indices]) / k
    return both_operands_m_mod_n_percentage


def is_result_m_mod_n_neuron(layer: int, neuron_idx: int, m: int, n: int, heuristic_analysis_data: HeuristicAnalysisData):
    """
    Checks if a specific neuron implements a mod heuristic on the prompts result.
    """
    k = int((heuristic_analysis_data.max_op // n) ** 2 * analyzed_operand_pairs_factor(heuristic_analysis_data))
    if heuristic_analysis_data.operator_idx == 2 or heuristic_analysis_data.operator_idx == 3:
        k = min(k, heuristic_analysis_data.max_op) # HACKY WAY TO SOLVE THE RESULT IMBALANCE IN MULT AND DIV
    topk_results = heuristic_analysis_data.top_results[(layer, neuron_idx)][:k]
    match_score = sum([result % n == m for result in topk_results]) / k
    return match_score
    

def is_operand_range_neuron(layer: int, neuron_idx: int, range: Tuple[int, int], heuristic_analysis_data: HeuristicAnalysisData, op_index: int = 1):
    """
    Checks if a specific neuron implements a range heuristic on one of the operands (e.g. op1 in [10, 20]).
    """
    cache_key = ('is_operand_range_neuron', range[0], range[1], heuristic_analysis_data.operator_idx)
    if cache_key not in heuristic_analysis_data.k_per_heuristic_cache:
        heuristic_analysis_data.k_per_heuristic_cache[cache_key] = len([(op1, op2) for (op1, op2) in heuristic_analysis_data.op1_op2_pairs if range[0] <= (op1 if op_index == 1 else op2) < range[1]])
    k = heuristic_analysis_data.k_per_heuristic_cache[cache_key]
    if k == 0:
        return 0.0
    topk_op1_op2_indices = heuristic_analysis_data.top_op1_op2_indices[(layer, neuron_idx)][:k]
    range_percentage = sum([range[0] <= (op1 if op_index == 1 else op2) < range[1] for (op1, op2) in topk_op1_op2_indices]) / k
    return range_percentage


def is_both_operands_range_neuron(layer: int, neuron_idx: int, range: Tuple[int, int], heuristic_analysis_data: HeuristicAnalysisData):
    """
    Checks if a specific neuron implements the same range heuristic on both of its operands (e.g. op1 in [10, 20] and op2 in [10, 20]).
    """
    cache_key = ('is_both_operands_range_neuron', range[0], range[1], heuristic_analysis_data.operator_idx)
    if cache_key not in heuristic_analysis_data.k_per_heuristic_cache:
        heuristic_analysis_data.k_per_heuristic_cache[cache_key] = len([(op1, op2) for (op1, op2) in heuristic_analysis_data.op1_op2_pairs if range[0] <= op1 < range[1] and range[0] <= op2 < range[1]])
    k = heuristic_analysis_data.k_per_heuristic_cache[cache_key]
    if k == 0:
        return 0.0
    topk_op1_op2_indices = heuristic_analysis_data.top_op1_op2_indices[(layer, neuron_idx)][:k]
    both_operands_range_percentage = sum([range[0] <= op1 < range[1] and range[0] <= op2 < range[1] for (op1, op2) in topk_op1_op2_indices]) / k
    return both_operands_range_percentage


def is_result_range_neuron(layer: int, neuron_idx: int, range: Tuple[int, int], heuristic_analysis_data: HeuristicAnalysisData):
    """
    Checks if the neuron implements a range heuristic on the result (e.g. result in [10, 20]).
    """
    cache_key = ('is_result_range_neuron', range[0], range[1], heuristic_analysis_data.operator_idx)
    if cache_key not in heuristic_analysis_data.k_per_heuristic_cache:
        heuristic_analysis_data.k_per_heuristic_cache[cache_key] = len([(op1, op2) for (op1, op2) in heuristic_analysis_data.op1_op2_pairs if range[0] <= safe_eval(f'{op1}{OPERATORS[heuristic_analysis_data.operator_idx]}{op2}') < range[1]])
    k = heuristic_analysis_data.k_per_heuristic_cache[cache_key]
    if k == 0:
        return 0.0
    if heuristic_analysis_data.operator_idx == 2 or heuristic_analysis_data.operator_idx == 3:
        k = min(k, heuristic_analysis_data.max_op) # HACKY WAY TO SOLVE THE RESULT IMBALANCE IN MULT AND DIV

    topk_results = heuristic_analysis_data.top_results[(layer, neuron_idx)][:k]

    if heuristic_analysis_data.operator_idx == 3 and set(topk_results) == {0}:
        # Special case for division, where the result is always zero. This can be ignored, it will be caught by "result_value".
        return 0.0

    result_range_percentage = sum([range[0] <= result < range[1] for result in topk_results]) / k
    return result_range_percentage
    

def is_same_operand_neuron(layer: int, neuron_idx: int, heuristic_analysis_data: HeuristicAnalysisData):
    """
    Checks if the neuron implements an "identical operands" heuristic (i.e. fires mostly when op1 == op2).
    """
    if heuristic_analysis_data.operator_idx == 2:
        # Multiplication is the only special case, in all other cases all op1-op2 pairs are valid.
        k = int(heuristic_analysis_data.max_single_token ** 0.5)
    else:
        k = heuristic_analysis_data.max_op
    topk_op1_op2_indices = heuristic_analysis_data.top_op1_op2_indices[(layer, neuron_idx)][:k]
    same_op_percentage = sum([op1 == op2 for (op1, op2) in topk_op1_op2_indices]) / k
    return same_op_percentage


def get_periodic_patterns(layer: int, neuron_idx: int, heuristic_analysis_data: HeuristicAnalysisData, op_index: Optional[int] = None):
    """
    Get the most significant patterns in one of the values (op1, op2 or result) in the prompts that activate the neuron the most.
    For example, if the neuron is mostly activated by prompts with op1=101, 102, 201, 202, etc., the pattern ".0." will be one of those returned.

    Args:
        layer: int - the layer of the neuron.
        neuron_idx: int - the index of the neuron.
        heuristic_analysis_data: HeuristicAnalysisData - the data structures that contain the required data for the analysis.
        op_index: Optional[int] - the index of the operand to analyze. If None, the result will be analyzed.
    """
    if op_index is None:
        k = int(heuristic_analysis_data.max_op * analyzed_operand_pairs_factor(heuristic_analysis_data)) 
        pattern_prefix = "result"
        topk_results = heuristic_analysis_data.top_results[(layer, neuron_idx)][:k]
        top_patterns_and_counts = most_significant_wildcard_patterns(topk_results, min_occurrences=k // 10, k_patterns=5)
    else:
        k = int(heuristic_analysis_data.max_op * analyzed_operand_pairs_factor(heuristic_analysis_data))
        pattern_prefix = f"op{op_index}"
        topk_op1_op2_pairs = heuristic_analysis_data.top_op1_op2_indices[(layer, neuron_idx)][:k]
        top_patterns_and_counts = most_significant_wildcard_patterns([op1 if op_index == 1 else op2 for (op1, op2) in topk_op1_op2_pairs], min_occurrences=k // 10, k_patterns=5)

    relevant_patterns = {}
    for pattern_name, count in top_patterns_and_counts:
        # Pattern name is a wildcard pattern of "123" (e.g. "1.3", ".23", "1..").
        if pattern_name.count('.') == 0:
            # Complete match ("123") - a size-1 range pattern
            pattern_name = f"{pattern_prefix}_value_{pattern_name}"
        elif (pattern_name[-1] == '.' and pattern_name[0] != '.'):
            # Handles two cases ([1.., 12.]). In these cases the pattern is a range pattern, and not periodic.
            # In this case this will be handled by the range_size_10 heuristic. No need to handle this.
            # pattern_name = f"{pattern_prefix}_range_{pattern_name}"
            continue
        else:
            # Periodic pattern, one of [.2., ..3, .23, 1.3]
            pattern_name = f"{pattern_prefix}_pattern_{pattern_name}"
            
        relevant_patterns[pattern_name] = max(relevant_patterns.get(pattern_name, 0), count / k)

    relevant_patterns = list(relevant_patterns.items())
    return relevant_patterns

def is_multi_result_neuron(layer, neuron_idx, multi_n: int, heuristic_analysis_data: HeuristicAnalysisData):
    k = 1000
    threshold = 0.95
    topk_results = heuristic_analysis_data.top_results[(layer, neuron_idx)][:k]
    # Create a histogram of the results (count how many times each result appears)
    res_counts = torch.zeros(heuristic_analysis_data.max_op,)
    for result in topk_results:
        res_counts[result] += 1
    res_counts = res_counts / res_counts.sum()
    vals, indices = res_counts.sort(descending=True)
    less_than_multi_n_sum = vals[:multi_n - 1].sum()
    multi_n_sum = less_than_multi_n_sum + vals[multi_n - 1]
    if less_than_multi_n_sum < threshold and multi_n_sum >= threshold:
        return sorted(indices[:multi_n].tolist()), multi_n_sum
    else:
        return [], 0.0


def classify_heuristic_neurons(heuristic_neurons: List[Tuple[int, int]], 
                               heuristic_analysis_data: HeuristicAnalysisData,
                               verbose: bool = True):
    """
    The implementation of the heuristic classification algorithm for a list of neurons.
    Each neuron is checked against all pre-defined heuristics, for each possible parameter value.

    Args:
        heuristic_neurons: List[Tuple[int, int]] - the list of (layer, neuron) tuples to classify.
        heuristic_analysis_data: HeuristicAnalysisData - the data structures that contain the required data for the analysis.
        verbose: bool - whether to print progress information.
    """
    heuristic_matches_dict = {}

    max_op, max_single_token = heuristic_analysis_data.max_op, heuristic_analysis_data.max_single_token

    # Check if also the lowest activations should be classified for heuristics.
    # This happens when the activation patterns can also receive large negative resutls (for example, in Llama3 K activation maps).
    use_bottom_topk = heuristic_analysis_data.also_check_bottom_results
    if use_bottom_topk:
        if verbose:
            print("Preparing bottom activations for heuristics...")
        bottom_heuristic_data = HeuristicAnalysisData()
        bottom_heuristic_data.also_check_bottom_results = False # Doesn't really matter
        bottom_heuristic_data.op1_op2_pairs = heuristic_analysis_data.op1_op2_pairs
        bottom_heuristic_data.top_op1_op2_indices = {(layer, neuron): heuristic_analysis_data.top_op1_op2_indices[(layer, neuron)][::-1] for (layer, neuron) in heuristic_analysis_data.top_op1_op2_indices}
        bottom_heuristic_data.top_results = {(layer, neuron): heuristic_analysis_data.top_results[(layer, neuron)][::-1] for (layer, neuron) in heuristic_analysis_data.top_results}
        bottom_heuristic_data.max_op = max_op
        bottom_heuristic_data.max_single_token = max_single_token
        bottom_heuristic_data.operator_idx = heuristic_analysis_data.operator_idx
        bottom_heuristic_data.k_per_heuristic_cache = heuristic_analysis_data.k_per_heuristic_cache

    max_result = [max_op + max_op, max_op, max_single_token + 1, max_op][heuristic_analysis_data.operator_idx]
    neuron_iteration = tqdm(heuristic_neurons) if verbose else heuristic_neurons
    for (layer, neuron) in neuron_iteration:
        for op_index in [1, 2]:
            # m mod n
            for n in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
                for m in range(0, n):
                    score = is_operand_m_mod_n_neuron(layer, neuron, m, n, heuristic_analysis_data, op_index)
                    if use_bottom_topk:
                        score_bottom = is_operand_m_mod_n_neuron(layer, neuron, m, n, bottom_heuristic_data, op_index)
                        score = max(score, score_bottom)
                    insert(heuristic_matches_dict, f"op{op_index}_{m}mod{n}", (layer, neuron, score))

            # operand periodic pattern
            pattern_names = get_periodic_patterns(layer, neuron, heuristic_analysis_data, op_index)
            if use_bottom_topk:
                pattern_names_bottom = get_periodic_patterns(layer, neuron, bottom_heuristic_data, op_index)
                pattern_names = pattern_names + pattern_names_bottom
            for pattern_name, pattern_score in pattern_names:
                insert(heuristic_matches_dict, pattern_name, (layer, neuron, pattern_score))

            # operand range
            for range_size in OPERAND_range_SIZES_BY_OPERATOR[OPERATORS[heuristic_analysis_data.operator_idx]]:
                for range_start in range(0, max_op, range_size):
                    range = (range_start, range_start + range_size)
                    range_score = is_operand_range_neuron(layer, neuron, range, heuristic_analysis_data, op_index)
                    if use_bottom_topk:
                        range_score_bottom = is_operand_range_neuron(layer, neuron, range, bottom_heuristic_data, op_index)
                        range_score = max(range_score, range_score_bottom)
                    insert(heuristic_matches_dict, f"op{op_index}_range_{range[0]}_{range[1]}", (layer, neuron, range_score))

        # Both operands m mod n
        for n in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
            for m in range(0, n):
                score = is_both_operands_m_mod_n_neuron(layer, neuron, m, n, heuristic_analysis_data)
                if use_bottom_topk:
                    score_bottom = is_both_operands_m_mod_n_neuron(layer, neuron, m, n, bottom_heuristic_data)
                    score = max(score, score_bottom)
                insert(heuristic_matches_dict, f"both_operands_{m}mod{n}", (layer, neuron, score))
        
        # Both operands range
        for range_size in OPERAND_range_SIZES_BY_OPERATOR[OPERATORS[heuristic_analysis_data.operator_idx]]:
            for range_start in range(0, max_op, range_size):
                range = (range_start, range_start + range_size)
                range_score = is_both_operands_range_neuron(layer, neuron, range, heuristic_analysis_data)
                if use_bottom_topk:
                    range_score_bottom = is_both_operands_range_neuron(layer, neuron, range, bottom_heuristic_data)
                    range_score = max(range_score, range_score_bottom)
                insert(heuristic_matches_dict, f"both_operands_range_{range[0]}_{range[1]}", (layer, neuron, range_score))

        # result m mod n
        if not (heuristic_analysis_data.operator_idx == 3):
            for n in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
                for m in range(0, n):
                    score = is_result_m_mod_n_neuron(layer, neuron, m, n, heuristic_analysis_data)
                    if use_bottom_topk:
                        score_bottom = is_result_m_mod_n_neuron(layer, neuron, m, n, bottom_heuristic_data)
                        score = max(score, score_bottom)
                    insert(heuristic_matches_dict, f"result_{m}mod{n}", (layer, neuron, score))
        
        # result periodic pattern
        pattern_names = get_periodic_patterns(layer, neuron, heuristic_analysis_data, op_index=None)
        for pattern_name, pattern_score in pattern_names:
            if (heuristic_analysis_data.operator_idx == 3 and 'pattern' in pattern_name):
                # Cancel pattern heuristics in division, not enough possible results for this to be helpful
                continue
            insert(heuristic_matches_dict, pattern_name, (layer, neuron, pattern_score))

        # result range
        for range_size in RESULT_range_SIZES_BY_OPERATOR[OPERATORS[heuristic_analysis_data.operator_idx]]:
            for range_start in range(0, max_result, max(range_size // 3, 10)):
                range = (range_start, range_start + range_size)
                range_score = is_result_range_neuron(layer, neuron, range, heuristic_analysis_data)
                insert(heuristic_matches_dict, f"result_range_{range[0]}_{range[1]}", (layer, neuron, range_score))

        # Same operand neuron
        same_operand_score = is_same_operand_neuron(layer, neuron, heuristic_analysis_data)
        insert(heuristic_matches_dict, f"same_operand", (layer, neuron, same_operand_score))           

        # Mult / Div special heuristics
        if heuristic_analysis_data.operator_idx == 3:
            # Multi result neuron
            for multi_n in range(2, 6):
                multi_results, multi_result_score = is_multi_result_neuron(layer, neuron, multi_n, heuristic_analysis_data)
                if multi_results != []:
                    insert(heuristic_matches_dict, f"result_multi_value_{multi_n}={multi_results}", (layer, neuron, multi_result_score.item()))

    # Some post processing
    if heuristic_analysis_data.operator_idx == 3:
        # Remove any heuristic based on result=0
        for removal_heuristic in ['result_range_0_100', 'result_range_0_10', 'result_range_0_2',
                                'result_multi_value_2=[0, 1]', 'result_multi_value_2=[0, 2]', 
                                'result_multi_value_3=[0, 1, 2]', 
                                'result_multi_value_4=[0, 1, 2, 3]', 
                                'result_multi_value_5=[0, 1, 2, 3, 4]']:
            if removal_heuristic in heuristic_matches_dict:
                del heuristic_matches_dict[removal_heuristic]

    if heuristic_analysis_data.operator_idx == 2:
        for (layer, neuron, score) in heuristic_matches_dict['result_value_000']:
            if score == 1.0:
                for zero_heuristic_to_cancel in [f'result_0mod{n}' for n in range(2, 11)] + [f'{v}_pattern_{p}' for v in ['op1', 'op2', 'result'] for p in ['..0', '.00', '0.0', '00.']]:
                    try:
                        i = [(l, n) for (l, n, s) in heuristic_matches_dict[zero_heuristic_to_cancel]].index((layer, neuron))
                        heuristic_matches_dict[zero_heuristic_to_cancel][i] = (layer, neuron, 0.0)
                    except Exception as e:
                        pass

    return heuristic_matches_dict

def load_heuristic_classes(data_dir: str, operator_idx: int, neuron_activations_type: str, override_fileprefix: Optional[str] = None):
    """
    Load the pre-calculated heuristic-to-neurons dictionary.

    Args:
        data_dir: str - the directory to load the data from.
        operator_idx: int - the index of the operator to load the data for.
        neuron_activations_type: str - Either "K" (key), "KV", (key-value), or "HYBRID".
                                    For HYBRID, direct heuristic matches are taken from KV-based matches, and indirect heuristics are taken from K-based matches.
                                    This is the measure presented in the paper.
        override_fileprefix: str - override the default fileprefix for the data files.
    """
    file_prefix = override_fileprefix or f'{OPERATOR_NAMES[operator_idx]}_heuristic_matches_dict'
    if neuron_activations_type == "KV":
        return torch.load(os.path.join(data_dir, file_prefix + '_KV_maps.pt'))
    elif neuron_activations_type == "K":
        return torch.load(os.path.join(data_dir, file_prefix + '_K_maps.pt'))
    elif neuron_activations_type == "HYBRID":
        kv_based_matches = torch.load(os.path.join(data_dir, file_prefix + '_KV_maps.pt'))
        k_based_matches = torch.load(os.path.join(data_dir, file_prefix + '_K_maps.pt'))

        # Result heuristics from KV-based matches, Operand heuristics from K-based matches
        result_heuristic_matches_from_kv = {name: layer_neuron_scores for (name, layer_neuron_scores) in kv_based_matches.items() if name.startswith('result')}
        operand_heuristic_matches_from_k = {name: layer_neuron_scores for (name, layer_neuron_scores) in k_based_matches.items() if not name.startswith('result')}
        result_heuristic_matches_from_kv.update(operand_heuristic_matches_from_k)
        return result_heuristic_matches_from_kv
    else:
        raise ValueError(f"Unknown neuron_activations_type: {neuron_activations_type}")