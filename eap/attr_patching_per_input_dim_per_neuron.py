from functools import partial
import random
import torch
import transformer_lens as lens
from tqdm import tqdm
from typing import List, Tuple
from metrics import indirect_effect

from prompt_generation import separate_prompts_and_answers


# THIS IS A HACKEY VERSION OF "full_attribution_patching" as it exists
# in attr_patching.py.
# This is used only for finding the gradient of each input dimension at each mlp_in neuron input.


def full_attribution_patching_per_input_dim_per_neuron(
    model: lens.HookedTransformer,
    prompts_and_answers: List[Tuple[str, str]],
    corrupt_prompts_and_answers: List[Tuple[str, str]] = None,
    metric: str = "IE",
    pos=-1,
    batch_size: int = 1,
):
    model.requires_grad_(True)

    forward_hook_names = ["ln2.hook_normalized"]  # ['hook_mlp_in']
    backward_hook_names = ["mlp.hook_pre"]

    forward_hook_filter = partial(
        should_measure_hook, measurable_hooks=forward_hook_names
    )
    backward_hook_filter = partial(
        should_measure_hook, measurable_hooks=backward_hook_names
    )

    # Choose a random corrupt prompt for each prompt, if not given
    if corrupt_prompts_and_answers is None:
        corrupt_prompts_and_answers = []
        for prompt_idx in range(len(prompts_and_answers)):
            # Choose a random prompt to corrupt with, without any limitations other than choosing a different prompt
            corrupt_prompt_idx = random.choice(
                list(set(range(len(prompts_and_answers))) - {prompt_idx})
            )
            corrupt_prompts_and_answers.append(prompts_and_answers[corrupt_prompt_idx])

    prompts, answers = separate_prompts_and_answers(prompts_and_answers)
    corrupt_prompts, corrupt_answers = separate_prompts_and_answers(
        corrupt_prompts_and_answers
    )
    labels, corrupt_labels = model.to_tokens(
        answers, prepend_bos=False
    ), model.to_tokens(corrupt_answers, prepend_bos=False)

    attr_patching_scores = {}

    for idx in tqdm(range(0, len(prompts), batch_size)):
        prompt_batch, corrupt_prompt_batch = (
            prompts[idx : idx + batch_size],
            corrupt_prompts[idx : idx + batch_size],
        )
        label_batch, corrupt_label_batch = (
            labels[idx : idx + batch_size],
            corrupt_labels[idx : idx + batch_size],
        )

        # First forward pass to get corrupt cache
        _, corrupt_cache = model.run_with_cache(corrupt_prompt_batch)
        corrupt_cache = {
            k: v
            for (k, v) in corrupt_cache.cache_dict.items()
            if forward_hook_filter(k)
        }

        # Second forward pass to get corrupt cache
        clean_logits_orig, clean_cache = model.run_with_cache(prompt_batch)
        clean_cache = {
            k: v for (k, v) in clean_cache.cache_dict.items() if forward_hook_filter(k)
        }

        # Calculate the difference between every two parallel activation cache elements
        diff_cache = {}
        for k in clean_cache:
            diff_cache[k] = (corrupt_cache[k] - clean_cache[k])[:, pos, :].cpu()

        def backward_hook_fn(grad, hook, attr_patching_scores):
            matching_hook_name_key = f"blocks.{hook.layer()}.{forward_hook_names[0]}"
            if matching_hook_name_key not in attr_patching_scores:
                # attr_patching_scores[matching_hook_name_key] = torch.zeros((model.cfg.d_model,), device='cpu') # for full attribution (not per neuron per dim)
                # attr_patching_scores[matching_hook_name_key] = torch.zeros(model.cfg.d_model, model.cfg.d_mlp, device='cpu')  # for per neuron per input dim attribution
                attr_patching_scores[matching_hook_name_key] = torch.zeros(
                    len(prompts), model.cfg.d_model, model.cfg.d_mlp, device="cpu"
                )  # for per neuron per input dim attribution

            # Full attribution - not per neuron per dim, but should work (on dimension level) - THIS IS FOR DEBUGGING
            # grad_a_pre_act = model.blocks[hook.layer()].mlp.W_in # The gradient of the pre_act output w.r.t the input - (d_model, d_mlp)
            # grad_l = grad[:, pos, :] # The gradient of the loss metric w.r.t. the pre_act activation (d_mlp, )
            # attr_patching_scores[matching_hook_name_key] += (diff_cache[matching_hook_name_key] * (grad_l @ grad_a_pre_act.transpose(0, 1))).sum(dim=0).cpu()
            # attr_patching_scores[matching_hook_name_key] += (diff_cache[matching_hook_name_key] * grad[:, pos, :]).sum(dim=0).cpu()

            # Per neuron per input dim attribution
            grad_L_wrt_e = (model.blocks[hook.layer()].mlp.W_in * grad[:, pos, :]).cpu()
            # attr_patching_scores[matching_hook_name_key] += (diff_cache[matching_hook_name_key].unsqueeze(-1) * grad_L_wrt_e).sum(dim=0)
            attr_patching_scores[matching_hook_name_key][idx : idx + batch_size] = (
                diff_cache[matching_hook_name_key].unsqueeze(-1) * grad_L_wrt_e
            )

        model.reset_hooks()
        model.add_hook(
            name=backward_hook_filter,
            hook=partial(backward_hook_fn, attr_patching_scores=attr_patching_scores),
            dir="bwd",
        )
        with torch.set_grad_enabled(True):
            clean_logits = model(prompt_batch, return_type="logits")
            if metric == "IE":
                value = indirect_effect(
                    clean_logits_orig[:, -1].softmax(dim=-1),
                    clean_logits[:, -1].softmax(dim=-1),
                    label_batch,
                    corrupt_label_batch,
                ).mean(dim=0)
            else:
                raise ValueError(f"Unknown metric {metric}")
            value.backward()
            model.zero_grad()

        del diff_cache

    # for hook_name in attr_patching_scores.keys():
    #     attr_patching_scores[hook_name] = (attr_patching_scores[hook_name] / len(prompts)).cpu()

    model.reset_hooks()
    model.requires_grad_(False)
    torch.cuda.empty_cache()

    return attr_patching_scores


def should_measure_hook(hook_name, measurable_hooks):
    if any([h in hook_name for h in measurable_hooks]):
        return True
