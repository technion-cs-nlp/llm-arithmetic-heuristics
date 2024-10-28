from functools import partial
import random
import torch
import transformer_lens as lens
from tqdm import tqdm
from typing import List, Tuple, Dict
from general_utils import get_hook_dim
from metrics import indirect_effect, logit_diff
from component import Component

from prompt_generation import separate_prompts_and_answers

# TODO CHANGE TO RECEIVE A LIST OF COMPONENT OBJECTS
def node_attribution_patching(model: lens.HookedTransformer,
                              prompts_and_answers: List[Tuple[str, str]],
                              corrupt_prompts_and_answers: List[Tuple[str, str]] = None,
                              mean_cache: Dict[Component, torch.Tensor] = None,
                              attributed_hook_names: List[str] = ['mlp.hook_post', 'hook_z'],
                              metric: str = 'IE',
                              batch_size: int = 1,
                              verbose: bool = True):
    """
    Get a cache of attribution patching scores for all components of the model, 
    estimating the ground truth activation patching result of each component.

    Args:
        TODO COMPLETE

    Returns:
        TODO COMPLETE
    """
    model.requires_grad_(True)

    # Node filter function
    should_measure_hook_filter = partial(should_measure_hook, measurable_hooks=attributed_hook_names)

    # Choose a random corrupt prompt for each prompt, if not given
    assert corrupt_prompts_and_answers or mean_cache, "Either corrupt prompts or mean cache must be provided for ablation attribution"
    use_counterfactual_ablation = corrupt_prompts_and_answers is not None

    if use_counterfactual_ablation:
        corrupt_prompts, corrupt_answers = separate_prompts_and_answers(corrupt_prompts_and_answers)
        corrupt_labels = model.to_tokens(corrupt_answers, prepend_bos=False)

    prompts, answers = separate_prompts_and_answers(prompts_and_answers)
    labels = model.to_tokens(answers, prepend_bos=False)

    attr_patching_scores = {}

    it = tqdm(range(0, len(prompts), batch_size)) if verbose else range(0, len(prompts), batch_size)
    for idx in it:
        prompt_batch = prompts[idx : idx + batch_size]
        label_batch = labels[idx : idx + batch_size].to(device=model.cfg.device)

        if use_counterfactual_ablation:
            corrupt_prompt_batch = corrupt_prompts[idx : idx + batch_size]
            corrupt_label_batch = corrupt_labels[idx : idx + batch_size].to(device=model.cfg.device)

            # Forward pass to get corrupt cache
            _, corrupt_cache = model.run_with_cache(corrupt_prompt_batch)
            corrupt_cache = {k: v for (k, v) in corrupt_cache.cache_dict.items() if should_measure_hook_filter(k)}
        else:
            corrupt_cache = mean_cache

        # Forward pass to get corrupt cache
        clean_logits_orig, clean_cache = model.run_with_cache(prompt_batch)
        clean_cache = {k: v for (k, v) in clean_cache.cache_dict.items() if should_measure_hook_filter(k)}

        # Calculate the difference between every two parallel activation cache elements
        diff_cache = {}
        for k in clean_cache:
            diff_cache[k] = corrupt_cache[k] - clean_cache[k]
        
        def backward_hook_fn(grad, hook, attr_patching_scores):
            # Gradient is multiplicated with the activation difference (between corrupt and clean prompts).
            if hook.name not in attr_patching_scores:
                attr_patching_scores[hook.name] = torch.zeros((len(prompts),) + clean_cache[hook.name].shape[1:], device='cpu')
            attr_patching_scores[hook.name][idx : idx + batch_size] = (diff_cache[hook.name] * grad).cpu()

        model.reset_hooks()
        model.add_hook(name=should_measure_hook_filter, hook=partial(backward_hook_fn, attr_patching_scores=attr_patching_scores), dir="bwd")
        with torch.set_grad_enabled(True):   
            clean_logits = model(prompt_batch, return_type="logits")
            if metric == 'IE':
                if use_counterfactual_ablation:
                    value = indirect_effect(clean_logits_orig[:, -1].softmax(dim=-1).to(device=model.cfg.device), 
                                            clean_logits[:, -1].softmax(dim=-1).to(device=model.cfg.device), 
                                            label_batch, corrupt_label_batch).mean(dim=0)
                else:
                    # When using mean ablation in attribution, the IE metric becomes only the "clean" part in the original IE
                    pre_ablation_probs = clean_logits_orig[:, -1].softmax(dim=-1)
                    post_ablation_probs = clean_logits[:, -1].softmax(dim=-1)
                    value = (pre_ablation_probs.gather(1, label_batch) - post_ablation_probs.gather(1, label_batch)) / post_ablation_probs.gather(1, label_batch)
                    value = value.nan_to_num(0)
                    value = value.squeeze(1).mean(dim=0)
            elif metric == 'KL':
                kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=False)
                target = clean_logits_orig[:, -1].softmax(dim=-1)
                value = kl_loss(clean_logits[:, -1].softmax(dim=-1).log(), target)
            else:
                raise ValueError(f"Unknown metric {metric}")
            value.backward()
            model.zero_grad()
    
        del diff_cache

    model.reset_hooks()
    model.requires_grad_(False)
    torch.cuda.empty_cache()

    return attr_patching_scores


def should_measure_hook(hook_name, measurable_hooks):
    if any([h in hook_name for h in measurable_hooks]):
        return True