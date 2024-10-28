import torch
import transformer_lens as lens
from component import Component
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from typing import List

# The valid early and late components in a residual path.
# Valid early components write to the residual stream, and valid late components read from the residual stream.
VALID_EARLY_COMPONENTS = ['hook_z', 'hook_attn_out', 'hook_mlp_out', 'hook_resid_pre', 'hook_resid_post']
VALID_LATE_COMPONENTS = ['hook_q_input', 'hook_k_input', 'hook_v_input', 'hook_mlp_in', 'hook_resid_pre', 'hook_resid_post']


def is_valid_path(early_component: Component, late_component: Component) -> bool:
    """
    Check if a path from the early component to another late component is supported.
    A path is considered valid if the current (early) component writes to the residual stream,
    and the late component reads from the residual stream.
    The components can process information in different shapes (For example, hook_z -> hook_resid_post),
    but in this case a projection (using W_O matrix) is performed.

    NOTE: This function is used for path patching experiments.

    Args:
        early_component (Component): The early component to check the path from.
        late_component (Component): The late component to check the path to.
    """
    # hook_name can be either like "z" or like "hook_z", thus we check if its contained in any valid name
    return any(early_component.hook_name in valid_name for valid_name in VALID_EARLY_COMPONENTS) and \
            any(late_component.hook_name in valid_name for valid_name in VALID_LATE_COMPONENTS)


def is_earlier_component(model_cfg: HookedTransformerConfig, 
                         early_component: Component,
                         late_component: Component) -> bool:
    """
    Check if the early component is earlier than the late component in the model.
    """
    if early_component.layer < late_component.layer:
        return True
    elif early_component.layer == late_component.layer:
        # If both components are in the same layer, we say the early component is earlier only in several cases - 
        # 1. Attention -> MLP when they are not parallel
        # 2. Attention/MLP/resid_pre -> resid_post
        # 3. resid_pre -> Attention/MLP/resid_post
        if early_component.is_attn and late_component.is_mlp and not model_cfg.parallel_attn_mlp:
            return True
        elif 'resid_post' in late_component.valid_hook_name() or 'resid_pre' in early_component.valid_hook_name():
            return True
        else:
            return False
    else:
        return False


def topk_effective_components(model: lens.HookedTransformer, 
                              effect_map: torch.Tensor,
                              k: int = 3,
                              effect_threshold: float = None,
                              heads_only: bool = False):
        """
        Get the most effective components in the effect map.
        
        Args:
            effect_map (torch.Tensor): The effect map to get the most effective components from.
                Should be of shape (c, l), where c is the number of components in each layer (first heads, then MLP),
                and l is the number of layers.
            k (int): The number of components to return.
            effect_threshold (float): The threshold to filter the components by. 
                If None, no filtering is applied.
            heads_only (bool): If true, only attention heads are considered and MLPs are ignored.
        Returns:
            dict (Component -> float): A dictionary mapping the most effective components to their effect.
        """
        if heads_only:
            effect_map = effect_map[:, :-1] # Ignore last column, where MLP information should be

        # Make up a list of the most effective components for each C_1 components (to create "C_2")
        most_effective_components = {}
        indices = torch.topk(effect_map.flatten(), k=k, dim=0).indices
        layers, heads = indices // effect_map.shape[1], indices % effect_map.shape[1]
        for layer, head in zip(layers, heads):
            layer, head = layer.item(), head.item()
            if head == model.cfg.n_heads:
                # is mlp
                most_effective_components[Component('mlp_out', layer=layer)] = effect_map[layer, -1]
            else:
                # is head
                most_effective_components[Component('z', layer=layer, head=head)] = effect_map[layer, head]
        
        if effect_threshold is not None:
            most_effective_components = {c:e for c, e in most_effective_components.items() if e.abs() > effect_threshold}
        
        return most_effective_components


def convert_late_to_early(components: List[Component]):
    """
    Convert late components (q_input, k_input, v_input, mlp_in) to early components (z, mlp_out).
    
    Args:
        components (list[Component]): The components to convert.

    Return:
        list[Component]: The converted components.
    """
    converted = []
    for comp in components:
        if 'q_input' in comp.hook_name or 'k_input' in comp.hook_name or 'v_input' in comp.hook_name:
            converted.append(Component('z', layer=comp.layer, head=comp.head_idx))
        elif 'mlp_in' in comp.hook_name:
            converted.append(Component('mlp_out', layer=comp.layer))
        else:
            # Component is already early
            converted.append(comp)
    return converted


def convert_early_to_late(components: List[Component]):
    """
    Convert early components (z, mlp_out) to late components (q_input, k_input, v_input, mlp_in).
    
    Args:
        components (list[Component]): The components to convert.

    Return:
        list[Component]: The converted components.
    """
    converted = []
    for comp in components:
        if 'z' in comp.hook_name:
            converted.append(Component('q_input', layer=comp.layer, head=comp.head_idx))
            converted.append(Component('k_input', layer=comp.layer, head=comp.head_idx))
            converted.append(Component('v_input', layer=comp.layer, head=comp.head_idx))
        elif 'mlp_out' in comp.hook_name:
            converted.append(Component('mlp_in', layer=comp.layer))
        else:
            # Component is already late
            converted.append(comp)
    return converted