import torch
import random
from functools import partial
from fancy_einsum import einsum
from component import Component
from general_utils import set_deterministic
from circuit_utils import is_valid_path, is_earlier_component
from metrics import indirect_effect
from prompt_generation import separate_prompts_and_answers


def path_patching_experiment(model, 
                             late_component,
                             early_components,
                             prompts_and_answers,
                             metric='IE',
                             token_pos=None,
                             corrupt_prompts_and_answers=None,
                             random_seed=None):
    """
    Perform a path patching experiment, patching paths between many early components and a single late component.
    In each forward pass, the clean prompts are passed through the model, and the output from an early component 
    which goes (residually) into the late component is patched using the corrupt activations. The difference in
    logits is measured via IE (indirect effect).

    Args:
        model (lens.HookedTransformer): The model to patch.
        late_component (Component): The late component (C_l, receiver) that the path ends in.
        early_components (list[Components]): A list of early components (C_e, sender) that the paths start in.
        prompts_and_answers (List[Tuple[str, str]]): A list of (prompt, answer) tuples.
        token_pos (int): The position of the token to patch. If None, all positions are patched.
        corrupt_prompts_and_answers (List[Tuple[str, str]]): A list of (prompt, answer) tuples to use as the corrupt prompts.
                                                                If None, the corrupt prompts are chosen randomly from the prompts_and_answers list 
                                                                (such that no prompt is used as its own corrupt prompt).
        random_seed (int): The random seed to use for the experiment. If None, the seed is not set.

    Returns:
        torch.Tensor (n_prompts, n_layers, n_early_components): The metric results for each prompt, layer and early component.
    """
    if random_seed is not None:
        # Set random seed for reproducibility
        set_deterministic(seed=random_seed)
        
    metric_results = torch.zeros((len(prompts_and_answers), model.cfg.n_layers, len(early_components)), dtype=torch.float32)

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

    # Run both prompt batches to get the logits and activation cache
    clean_logits, clean_cache = model.run_with_cache(clean_prompts, return_type='logits')
    corrupt_logits, corrupt_cache = model.run_with_cache(corrupt_prompts, return_type='logits')

    # Patch the path between each component in each layer and the late component, and measure the effect metric
    max_layer_to_check = late_component.layer + (1 if not model.cfg.parallel_attn_mlp else 0) # If the model isn't parallel, we should also check 
                                                                                              # components in the late component layer
    try:
        for layer in range(0, max_layer_to_check):
            for component_idx in range(len(early_components)):
                early_comp = Component(hook_name=early_components[component_idx].hook_name,  
                                    head=early_components[component_idx].head_idx, 
                                    layer=layer)
                if not is_earlier_component(model.cfg, early_comp, late_component):
                    # Skip the case where the early and late components are in the same layer but not one-before-another
                    continue
                patched_logits = single_path_patch(model, late_component, early_comp, 
                                                    clean_prompts=clean_prompts, corrupt_prompts=corrupt_prompts,
                                                    clean_cache=clean_cache, corrupt_cache=corrupt_cache, token_pos=token_pos)
                clean_last_token_logits = clean_logits[:, -1]
                
                if metric == 'IE':
                    metric_results[:, layer, component_idx] = indirect_effect(clean_last_token_logits.softmax(dim=-1), patched_logits.softmax(dim=-1), clean_labels, corrupt_labels)
                else:
                    raise NotImplementedError
    finally:
        del clean_cache
        del corrupt_cache
            
    return metric_results


def hook_single_direct_path_func(value, hook, early_component, late_component, clean_cache, corrupt_cache, model=None, token_pos=None):
    """
    Patch a single direct path between two components.
    The patching logic is separated to several cases -
        1. hook_z -> attn_out/mlp_out/resid_pre/resid_post:
                The early component must be projected using its W_O matrix, and then the patching is done from the specific head in the early component to 
                the late component. 
        2. hook_z -> hook_q_in/hook_k_in/hook_v_in:
                The early component must be projected using its W_O matrix, and then the patching is done from the specific head in the early component to
                the specific head in the late component.
        3. attn_out/mlp_out/resid_pre/resid_post -> hook_q_in/hook_k_in/hook_v_in: 
                The patching is done directly to the specific head in the late component.
        3. attn_out/mlp_out/resid_pre/resid_post -> attn_out/mlp_out/resid_pre/resid_post: 
                The simplest case in which the patching is done directly (no projections).

    Note: Essentially, it is also possible to patch a path from hook_z -> hook_q/k/v by also projecting the diff in the early component using W_K/W_Q/W_V,
          but this is equivalent to patching hook_z -> hook_q_in/k_in/v_in and then (implicitly) applying the projection to the patched value, as done in case 2.
        
    Args:
        value (torch.Tensor): The value of the current hookpoint output.
        hook (lens.Hook): The hook object.
        early_component (Component): The early component in the hooked path.
        late_component (Component): The late (current) component in the hooked path.
        clean_cache (dict): The cache of clean activations.
        corrupt_cache (dict): The cache of corrupt activations.
        model (lens.HookedTransformer): The model in inference. Used to get the weight matrices for projecting head-specific outputs.
        token_pos (int): The position of the token to patch. If None, all positions are patched.
    Returns:
        (torch.Tensor): The patched value of the current hookpoint output.
    """
    early_component_name = early_component.valid_hook_name()
    token_pos = slice(token_pos) if token_pos is None else token_pos

    if early_component.head_idx is not None and 'hook_z' in early_component_name:
        # Case 1 or 2 - early component is a specific attention head (z) and requires projection using its W_O matrix 
        diff = (corrupt_cache[early_component_name][:, :, early_component.head_idx, :] - clean_cache[early_component_name][:, :, early_component.head_idx, :])
        diff = einsum('batch pos d_head, d_head d_model -> batch pos d_model', 
                        diff, model.W_O[early_component.layer, early_component.head_idx]) # No need to add bias because it cancels out (+b-b)
        diff = diff[:, token_pos]
        if late_component.head_idx is not None:
            # Case 1 - late component is also a specific attention head (q_in/k_in/v_in/q/k/v)

            if late_component.is_qkv:
                # Case 1.5 - Late component is q/k/v (after projection). This should be used when the model has GroupedQueryAttention.
                qkv_letter = late_component.hook_name.split('hook_')[-1][0].upper()
                proj_weight = getattr(model.blocks[late_component.layer].attn, f'W_{qkv_letter}')[late_component.head_idx]
                diff = einsum('batch d_model, d_model d_head -> batch d_head', diff, proj_weight) # No need to add bias because it cancels out here as well

            value[:, token_pos, late_component.head_idx, :] = value[:, token_pos, late_component.head_idx, :] + diff
        else:
            # Case 2 - late component is a layer-wide output (attn_out/mlp_out/resid_pre/resid_post)
            value[:, token_pos] = value[:, token_pos] + diff
    else:
        # Case 3/4 - early component is a layer-wide output (attn_out/mlp_out/resid_pre/resid_post)
        diff = (corrupt_cache[early_component_name] - clean_cache[early_component_name])[:, token_pos]
        if late_component.head_idx is not None:
            # Case 3 - late component is a specific attention head (q_in/k_in/v_in/q/k/v)
            if late_component.is_qkv:
                # Case 3.5 - Late component is q/k/v (after projection). This should be used when the model has GroupedQueryAttention.
                qkv_letter = late_component.hook_name.split('hook_')[-1][0].upper()
                proj_weight = getattr(model.blocks[late_component.layer].attn, f'W_{qkv_letter}')[late_component.head_idx]
                diff = einsum('batch d_model, d_model d_head -> batch d_head', diff, proj_weight) # No need to add bias because it cancels out here as well
            value[:, token_pos, late_component.head_idx, :] = value[:, token_pos, late_component.head_idx, :] + diff
        else:
            # Case 4 - late component is a layer-wide output (attn_out/mlp_out/resid_pre/resid_post)
            value[:, token_pos] = value[:, token_pos] + diff

    return value


def single_path_patch(model, late_component, early_component, clean_prompts, corrupt_prompts, clean_cache=None, corrupt_cache=None, token_pos=None):
    """
    Patch the paths from C_e (early_component) to C_l (late_component) in the model.

    Args:
        model (lens.HookedTransformer): The model to patch.
                                        NOTICE - This function currently assumes that the model is a GPT-J model, due to the attention-MLP parallelism. 
        late_component_name (str): The name of the late component (C_l, receiver) that the path ends in.
        early_component_name (str): The name of the early component (C_e, sender) that the path starts in.
        clean_prompts (List[str]): A list of clean prompts.
        corrupt_prompts (List[str]): A list of corrupt prompts.
        clean_cache (dict): The cache of clean activations (activations of the model on the clean prompts).
        corrupt_cache (dict): The cache of corrupt activations (activations of the model on the corrupt prompts).
        token_pos (int): The position of the token to patch. If None, all positions are patched.
    Returns:
        The logits resulting from running the patched model on the clean prompts.
    """
    model.reset_hooks()
    assert is_valid_path(early_component, late_component), f'Invalid path: {early_component.hook_name} -> {late_component.hook_name}'
    
    if clean_cache is None:
        _, clean_cache = model.run_with_cache(clean_prompts, return_type='logits')
    if corrupt_cache is None:
        _, corrupt_cache = model.run_with_cache(corrupt_prompts, return_type='logits')

    # Noising ablation
    hook_fn = partial(hook_single_direct_path_func, 
                        early_component=early_component, late_component=late_component,
                        clean_cache=clean_cache, corrupt_cache=corrupt_cache, model=model,
                        token_pos=token_pos)
    # Patch the and get its outputs
    patched_logits = model.run_with_hooks(clean_prompts, fwd_hooks=[(late_component.valid_hook_name(), hook_fn)], return_type='logits')

    # Denoising ablation
    # hook_fn = partial(hook_single_direct_path_func,
    #                     early_component=early_component, late_component=late_component,
    #                     clean_cache=corrupt_cache, corrupt_cache=clean_cache, model=model, # NOTICE THE OPPOSITE CACHES
    #                     token_pos=token_pos)
    # # Patch the and get its outputs
    # patched_logits = model.run_with_hooks(corrupt_prompts, fwd_hooks=[(late_component.valid_hook_name(), hook_fn)], return_type='logits')
    
    # Get the logits for the last token only
    patched_logits = patched_logits[:, -1, :]
    return patched_logits


def multiple_path_patch(model, late_component, early_components, clean_prompts, corrupt_prompts, clean_cache=None, corrupt_cache=None, token_pos=None):
    """
    Patch multiple paths during a single forward pass. Useful when trying to find the effect of a 
    group of early components on a single late component.

    Args:
        model (lens.HookedTransformer): The model to patch.
        late_component (Component): The late component (C_l, receiver) that the path ends in.
        early_components (list[Components]): A list of early components (C_e, sender) that the paths start in.
        clean_prompts (List[str]): A list of clean prompts.
        corrupt_prompts (List[str]): A list of corrupt prompts.
        clean_cache (dict): The cache of clean activations (activations of the model on the clean prompts).
        corrupt_cache (dict): The cache of corrupt activations (activations of the model on the corrupt prompts).

    Returns:
        The logits resulting from running the patched model on the clean prompts.
    """
    model.reset_hooks()
    for early_component in early_components:
        assert is_valid_path(early_component, late_component), f'Invalid path: {early_component.hook_name} -> {late_component.hook_name}'

    if clean_cache is None:
        _, clean_cache = model.run_with_cache(clean_prompts, return_type='logits')
    if corrupt_cache is None:
        _, corrupt_cache = model.run_with_cache(corrupt_prompts, return_type='logits')

    # Hook all paths from all early_components to the late_component,
    # and run a single forward pass with all hooks enabled
    hook_fns = [partial(hook_single_direct_path_func,
                        early_component=ec, late_component=late_component,
                        clean_cache=clean_cache, corrupt_cache=corrupt_cache, model=model) 
                        for ec in early_components]
    fwd_hooks = [(late_component.valid_hook_name(), hook_fn) for hook_fn in hook_fns] # Registering multiple hooks to the same hookpoint is allowed
    patched_logits = model.run_with_hooks(clean_prompts, fwd_hooks=fwd_hooks, return_type='logits')

    # Get the logits for the last token only
    patched_logits = patched_logits[:, -1, :]
    return patched_logits
        