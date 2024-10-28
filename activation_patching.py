from general_utils import set_deterministic
from functools import partial
import transformer_lens as lens 
import torch
import random
from metrics import indirect_effect
from prompt_generation import separate_prompts_and_answers
from typing import List, Tuple

def activation_patching_experiment(model: lens.HookedTransformer,
                                  prompts_and_answers: List[Tuple[str, str]], 
                                  metric: str='IE',
                                  hookpoint_name: str='mlp_post',
                                  n_shots: int=0,
                                  token_pos: int=-1, 
                                  hook_func_overload=None,
                                  corrupt_prompts_and_answers: List[Tuple[str, str]]=None,
                                  random_seed: int=None):
    """
    Performs an activation patching experiment.
    Each prompt is passed through the model, and at each layer, the activations are patched with the activations from another prompt.
    The effect of this patching is measured by the metric and averaged over all prompts.

    Args:
        model (lens.HookedTransformer): The model to patch.
        prompts_and_answers (List[Tuple[str, str]]): A list of (prompt, answer) tuples.
        metric (str): The metric to use. Can either be 'IE' (indirect effect) or 'IE-logits' (indirect effect on logits).
        hookpoint_name (str): The name of the hookpoint to patch. Defaults to patching MLP output (mlp_post). See transformer_lens for more details.
        n_shots (int): The number of pre-prompt examples to use. For example, for n_shots=1, the prompt '5+4=' might pass through the model as '13+27=40;5+4='.
                       The shots are chosen randomly from the prompts_and_answers list (excluding the current clean and corrupt prompt).
        token_pos (int): The token position to patch. Defaults to -1 (last token). If None, all token positions are patched.
        hook_func_overload (Callable): A function to overload the hook function with. If None, the default hook function is used.
                                       If this is not None, other hook-related arguments are ignored.
        corrupt_prompts_and_answers (List[Tuple[str, str]]): A list of (prompt, answer) tuples to use as the corrupt prompts. 
                                       If None, the corrupt prompts are chosen randomly from the prompts_and_answers list (such that no prompt 
                                       is used as its own corrupt prompt).
        random_seed (int): The random seed to use for the experiment. If None, the seed is not set.
    Returns:
        torch.Tensor (n_prompts, n_layers): The metric results for each prompt and layer.
    """
    if random_seed is not None:
        # Set random seed for reproducibility
        set_deterministic(seed=random_seed)

    metric_results = torch.zeros((len(prompts_and_answers), model.cfg.n_layers, ), dtype=torch.float32)

    # Define a default hooking function, which works for patching MLP / full attention output activations
    def default_patching_hook(value, hook, cache, token_pos):
        """
        A hook that works for some of the more common modules (MLP outputs, Attention outputs).
        """
        if token_pos is None:
            value = cache[hook.name]
        else:
            value[:, token_pos, :] = cache[hook.name][:, token_pos, :]

        return value
    
    hook_func = default_patching_hook if hook_func_overload is None else hook_func_overload

    # Choose a random corrupt prompt for each prompt, if not given
    if corrupt_prompts_and_answers is None:
        corrupt_prompts_and_answers = []
        for prompt_idx in range(len(prompts_and_answers)):
            # Choose a random prompt to corrupt with, without any limitations other than choosing a different prompt
            corrupt_prompt_idx = random.choice(list(set(range(len(prompts_and_answers))) - {prompt_idx}))
            corrupt_prompts_and_answers.append(prompts_and_answers[corrupt_prompt_idx])

    clean_prompts, clean_answers = separate_prompts_and_answers(prompts_and_answers)
    corrupt_prompts, corrupt_answers = separate_prompts_and_answers(corrupt_prompts_and_answers)
    clean_labels = model.to_tokens(clean_answers, prepend_bos=False)
    corrupt_labels = model.to_tokens(corrupt_answers, prepend_bos=False)

    # Add pre-prompt examples for each prompt, according to number of shots
    for i in range(n_shots):
        for prompt_idx in range(len(prompts_and_answers)):
            shot_prompt_idx = random.choice(list(set(range(len(prompts_and_answers))) - {prompt_idx, corrupt_prompt_idx}))
            shot_prompt = f'{clean_prompts[shot_prompt_idx]}={clean_answers[shot_prompt_idx]}'
            clean_prompts[prompt_idx] = shot_prompt + '\n' + clean_prompts[prompt_idx]
            corrupt_prompts[prompt_idx] = shot_prompt + '\n' + corrupt_prompts[prompt_idx]

    # Run both prompt batches to get the logits and activation cache
    clean_logits, clean_cache = model.run_with_cache(clean_prompts, return_type='logits')
    corrupt_logits, corrupt_cache = model.run_with_cache(corrupt_prompts, return_type='logits')

    # Patch each layer and measure the effect metric
    hook_fn_with_cache = partial(hook_func, cache=corrupt_cache, token_pos=token_pos)
    for layer in range(model.cfg.n_layers):
        patched_logits = model.run_with_hooks(clean_prompts, 
                                              fwd_hooks=[(lens.utils.get_act_name(name=hookpoint_name, layer=layer), hook_fn_with_cache)], 
                                              return_type='logits')
        if metric == 'IE':
            metric_results[:, layer] = indirect_effect(clean_logits[:, -1].softmax(dim=-1).to(model.cfg.device), 
                                                       patched_logits[:, -1].softmax(dim=-1).to(model.cfg.device), 
                                                       clean_labels.to(model.cfg.device), 
                                                       corrupt_labels.to(model.cfg.device))
        elif metric == 'IE-Logits':
            metric_results[:, layer] = indirect_effect(clean_logits[:, -1].to(model.cfg.device), 
                                                       patched_logits[:, -1].to(model.cfg.device), 
                                                       clean_labels.to(model.cfg.device), 
                                                       corrupt_labels.to(model.cfg.device)) 
        else:
            raise ValueError(f"Unknown metric {metric}")
            
    return metric_results
