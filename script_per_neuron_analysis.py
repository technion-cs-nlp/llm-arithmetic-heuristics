import pickle
import transformer_lens as lens
import torch
import random
import os
from functools import partial
from tqdm import tqdm
from general_utils import set_deterministic, get_hook_dim
from prompt_generation import separate_prompts_and_answers
from typing import List, Tuple, Dict
from metrics import indirect_effect
from component import Component


def per_neuron_ap_experiment(model: lens.HookedTransformer, 
                             prompts_and_answers: List[Tuple[str, str]], 
                             hook_component: Component,
                             corrupt_prompts_and_answers: List[Tuple[str, str]] = None, 
                             metric: str = 'IE', 
                             token_pos: int = -1,
                             random_seed: int = None):
    """
    Conducts a activation patching experiment on individual neurons of a given component.
    Args:
        model (lens.HookedTransformer): The transformer model to be analyzed.
        prompts_and_answers (List[Tuple[str, str]]): A list of tuples containing prompts and their corresponding answers.
        hook_component (Component): The component of the model to hook into for patching.
        corrupt_prompts_and_answers (List[Tuple[str, str]], optional): A list of tuples containing corrupt prompts and their corresponding answers. 
            If not provided, random corrupt prompts will be chosen. Defaults to None.
        metric (str, optional): The metric to be used for evaluation. Currently, only 'IE' (Indirect Effect) is supported. Defaults to 'IE'.
        token_pos (int, optional): The position of the token to be patched. If -1, the last token is used. Defaults to -1.
        random_seed (int, optional): The random seed for reproducibility. If None, no seed is set. Defaults to None.
    Returns:
        torch.Tensor: A tensor containing the metric results for each neuron.
    """
    if random_seed is not None:
        # Set random seed for reproducibility
        set_deterministic(seed=random_seed)

    embed_dim = get_hook_dim(model, hook_component.hook_name)
    metric_results = torch.zeros((len(prompts_and_answers), embed_dim), dtype=torch.float32)

    # Define a default hooking function, which works for patching MLP / full attention output activations
    # For specific head 
    def hook_fn(value, hook, cache, neuron_idx, token_pos):
        if token_pos is None:
            if hook_component.head_idx is None:
                value[:, :, neuron_idx] = cache[:, :, neuron_idx]
            else:
                value[:, :, hook_component.head_idx, neuron_idx] = cache[:, :, hook_component.head_idx, neuron_idx]
        else:
            if hook_component.head_idx is None:
                value[:, token_pos, neuron_idx] = cache[:, token_pos, neuron_idx]
            else:
                value[:, token_pos, hook_component.head_idx, neuron_idx] = cache[:, token_pos, hook_component.head_idx, neuron_idx]
        return value

    # Choose a random corrupt prompt for each prompt, if not given
    if corrupt_prompts_and_answers is None:
        corrupt_prompts_and_answers = []
        for prompt_idx in range(len(prompts_and_answers)):
            corrupt_prompt_idx = random.choice(list(set(range(len(prompts_and_answers))) - {prompt_idx}))
            corrupt_prompts_and_answers.append(prompts_and_answers[corrupt_prompt_idx])

    clean_prompts, clean_answers = separate_prompts_and_answers(prompts_and_answers)
    corrupt_prompts, corrupt_answers = separate_prompts_and_answers(corrupt_prompts_and_answers)
    clean_labels = model.to_tokens(clean_answers, prepend_bos=False)
    corrupt_labels = model.to_tokens(corrupt_answers, prepend_bos=False)

    # Run both prompt batches to get the logits and activation cache
    clean_logits = model(clean_prompts, return_type='logits')
    _, corrupt_cache = model.run_with_cache(corrupt_prompts, return_type='logits')
    specific_hook_cache = corrupt_cache[hook_component.valid_hook_name()].detach().clone()
    del corrupt_cache
    torch.cuda.empty_cache()

    # Patch each neuron and measure the effect
    for neuron_idx in tqdm(range(embed_dim)):
        hook_fn_with_cache = partial(hook_fn, cache=specific_hook_cache, neuron_idx=neuron_idx, token_pos=token_pos)
        patched_logits = model.run_with_hooks(clean_prompts, 
                                            fwd_hooks=[(hook_component.valid_hook_name(), hook_fn_with_cache)],
                                            return_type='logits')
        if metric == 'IE':
            metric_results[:, neuron_idx] = indirect_effect(clean_logits[:, -1].softmax(dim=-1), patched_logits[:, -1].softmax(dim=-1), clean_labels, corrupt_labels) 
        else:
            raise ValueError(f"Unknown metric {metric}")
            
    return metric_results


if __name__ == '__main__':
    # Code to run per_neuron_analysis as a background script because it takes too long to run in notebook.
    print('Loading model, prompts, etc')
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    device = 'cuda:0'
    torch.set_grad_enabled(False)
    gptj = lens.HookedTransformer.from_pretrained("EleutherAI/gpt-j-6b", fold_ln=True, center_unembed=True, center_writing_weights=True, device=device)
    gptj.eval()
    max_op = 100
    pos = -1

    results_output_path = fr'./data/addition/per_neuron_analysis_max_op={max_op}_operand_and_operator_corruptions.pkl'
    if os.path.exists(results_output_path):
        results_dict = pickle.load(open(results_output_path, 'rb'))
    else:
        results_dict = {}

    with open(fr'./data/gptj/correct_prompts_and_answers_max_op={max_op}.pkl', 'rb') as f:
        correct_prompts_and_answers = pickle.load(f)
        corrupt_prompts_and_answers = random.sample(correct_prompts_and_answers[0] + 
                                                    correct_prompts_and_answers[1] + 
                                                    correct_prompts_and_answers[2] + 
                                                    correct_prompts_and_answers[3], 
                                                    k=50)
        correct_prompts_and_answers = correct_prompts_and_answers[0]

                        

    print(f'Correct prompts: {correct_prompts_and_answers}')
    print(f'Corrupt prompts: {corrupt_prompts_and_answers}')

    print('Running AP experiments')
    for mlp in range(23, -1, -1):
        print(f'Running mlp_post={mlp}')	
        after_relu_ie = per_neuron_ap_experiment(gptj, correct_prompts_and_answers,
                                                 hook_component=Component('mlp_post', layer=mlp),
                                                 corrupt_prompts_and_answers=corrupt_prompts_and_answers,
                                                 token_pos=pos, random_seed=42)
        results_dict[f'mlp_{mlp}_pos_{pos}_max_op_{max_op}_mlp_post'] = after_relu_ie.mean(dim=0).cpu()

        with open(results_output_path, 'wb') as f:
            pickle.dump(results_dict, f)