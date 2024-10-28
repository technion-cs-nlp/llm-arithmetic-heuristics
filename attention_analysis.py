import torch
import transformer_lens as lens
from tqdm import tqdm
from typing import List

def two_operands_arithmetic_qk_heatmap(model: lens.HookedTransformer, 
                                       operator: str = '+',
                                       maximal_operand_value: int = 100,
                                       dst_token_position: int = -1, 
                                       show_progress: bool = True):
    """
    Visualize the attention maps of a model, based on all possible combinations of 
    input tokens at two operand positions (y axis for the first operand, x axis for 
    the second operand).
    The resulting visualization is a heatmap (one for each layer, head index and src position),
    wehre the value at (x, y) represents the attention value at the (layer, head) from the last dst token
    to the src position.

    Args:
        model (lens.HookedTransformer): The model to visualize.
        operator (str): The operator to use for the calculation.
        maximal_operand_value (int): The maximal value for each of the two operands.
        show_progress (bool): Whether to show a progress bar.
    Returns:
        torch.Tensor (n_layers, n_heads, len(tokens), len(tokens), positions_per_prompt) -
            A matrix of attention maps where a cell at (l,h,x,y,p) represents the attention
            value at head h at layer l, from the dst token at the given position to token at
            position p, where the first operand is x and the second operand is y.
    """
    positions_per_prompt = 5 # BOS, op1, operator, op2, =
    attention_pattern_values = torch.zeros((model.cfg.n_layers, model.cfg.n_heads, 
                                            maximal_operand_value, maximal_operand_value, 
                                            positions_per_prompt), dtype=torch.float16)

    progress = lambda x: tqdm(x) if show_progress else x
    for operand1 in progress(range(maximal_operand_value)):
        prompts = [f'{operand1}{operator}{operand2}=' for operand2 in range(0, maximal_operand_value)]
        dataloader = torch.utils.data.DataLoader(prompts, batch_size=32, shuffle=False)
        cur_idx = 0
        for batch in dataloader:
            _, cache = model.run_with_cache(batch)
            for layer in range(model.cfg.n_layers):
                for head_idx in range(model.cfg.n_heads):
                    attention_pattern_values[layer, head_idx, operand1, cur_idx:cur_idx+len(batch), :] = \
                        cache[f'blocks.{layer}.attn.hook_pattern'][:, head_idx, dst_token_position, :]
            cur_idx += len(batch)
            del cache
        torch.cuda.empty_cache()

    return attention_pattern_values


def ov_transition_analysis(model: lens.HookedTransformer,
                           layer: int,
                           head: int,
                           words: List[str]):
    """
    Visualize the OV transition matrix, defined as - 
        W_Transition = W_U^T @ W_V @ W_O @ W_U

    Which defines how an OV circuit connects pairs of (input_token, output_token).
    This is taken from the paper "Analyzing Transformers in Embedding Space" (https://arxiv.org/abs/2209.02535).

    Args:
        model (lens.HookedTransformer): The model to visualize.
        layer (int): The layer of the attention head to look at.
        head (int): The head index of the attention head to look at.
        words (List[str]): A list of possible tokens to be considered for the (input, output) pairs.
    """
    tokens = model.to_tokens(words, prepend_bos=False).view(-1) # T
    W_U = model.unembed.W_U[:, tokens] # d_model, T
    W_O = model.blocks[layer].attn.W_O[head] # d_head, d_model
    W_V = model.blocks[layer].attn.W_V[head] # d_model, d_head
    transition_matrix = W_U.T @ W_V @ W_O @ W_U # T, T
    return transition_matrix