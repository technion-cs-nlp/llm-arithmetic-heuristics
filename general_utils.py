import transformer_lens as lens
import umap
import numpy as np
import random
import pickle
import torch
import os

from model_analysis_consts import GPTJ_CONSTS, LLAMA3_70B_CONSTS, LLAMA3_8B_CONSTS, PYTHIA_6_9B_CONSTS
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import List
from tqdm import tqdm
from circuit import Circuit
from component import Component
from transformers import AutoModelForCausalLM, GPTNeoXForCausalLM


class Metric(object):
  def __init__(self):
    self.lst = 0.
    self.sum = 0.
    self.cnt = 0
    self.avg = 0.

  def update(self, val, cnt=1):
    self.lst = val
    self.sum += val * cnt
    self.cnt += cnt
    self.avg = self.sum / self.cnt


def load_model(model_name, model_path, device, extra_hooks=True):
    if 'pythia' in model_name:
        name, step = model_name.split('-step')
        model = lens.HookedTransformer.from_pretrained(model_name=name, hf_model=GPTNeoXForCausalLM.from_pretrained(f"EleutherAI/{name}", revision=f"step{step}", cache_dir=model_path), fold_ln=True, center_unembed=True, center_writing_weights=True, device=device)
    elif 'gptj' in model_name:
        model = lens.HookedTransformer.from_pretrained("EleutherAI/gpt-j-6b", fold_ln=True, center_unembed=True, center_writing_weights=True, device=device)
    elif 'llama3' in model_name:
        if 'llama3-8b' in model_name:
            name = "meta-llama/Meta-Llama-3-8B"
            inner_model = AutoModelForCausalLM.from_pretrained(model_path)
            model = lens.HookedTransformer.from_pretrained(model_name=name, hf_model=inner_model, fold_ln=True, center_unembed=True, center_writing_weights=True, device=device)
        elif 'llama3-70b' in model_name:
            name = "meta-llama/Meta-Llama-3-70B"
            inner_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
            model = lens.HookedTransformer.from_pretrained_no_processing(model_name=name, hf_model=inner_model, fold_ln=True, device=device, 
                                                                         n_devices=4, dtype=torch.float16)
    else:
        raise ValueError(f"Unsupported model ({model_name}) for model loading")

    model.set_use_split_qkv_input(extra_hooks)
    model.set_use_hook_mlp_in(extra_hooks)
    model.eval()
    return model


def get_model_consts(model_name):
    """
    Return a ModelAnalysisConsts instance for a given model name.
    """
    if 'pythia' in model_name and '6.9' in model_name:
        return PYTHIA_6_9B_CONSTS
    elif 'llama3-8b' in model_name:
        return LLAMA3_8B_CONSTS
    elif 'llama3-70b' in model_name:
        return LLAMA3_70B_CONSTS
    elif 'gptj' in model_name:
        return GPTJ_CONSTS
    else:
        raise ValueError(f"Unsupported model ({model_name}) for model consts")
        

def set_deterministic(seed=1337):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_cuda_device(device_idx):
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_idx)


def predict_answer(model, prompts):
    """
    Wrapper for model forward pass to predict answer tokens for a list of prompts.
    """
    tokens = model.to_tokens(prompts, prepend_bos=True)
    logits = model(tokens, return_type='logits')
    return model.to_str_tokens(logits[:, -1, :].argmax(dim=-1))


def generate_random_strings(model, num_tokens, count=1, batch_size=1, initial_token=None):
    """
    Generate a random string of tokens from the model.
    """
    result_strings = []
    for idx in range(0, count, batch_size):
        real_bs = min(count - idx, batch_size)
        if initial_token is None:
            initial_token = model.to_tokens("")
        else:
            initial_token = model.to_tokens(initial_token, prepend_bos=False)
        tokens = model.generate(initial_token.repeat(real_bs, 1), num_tokens - 1, prepend_bos=False, temperature=1.0) # -1 because BOS is already included
        result_strings += model.to_string(tokens[:, 1:]) # skip BOS
    return result_strings


def reduce_dimensionality(vectors, type='tsne'):
    if type == 'tsne':
        tsne = TSNE(n_components=2, random_state=0)
        tsne_vectors = tsne.fit_transform(vectors.detach().numpy())
        return tsne_vectors[:, 0], tsne_vectors[:, 1]
    elif type == 'pca':
        pca = PCA(n_components=2, random_state=0)
        pca_vectors = pca.fit_transform(vectors.detach().numpy())
        return pca_vectors[:, 0], pca_vectors[:, 1]
    elif type == 'umap':
        reducer = umap.UMAP()
        umap_vectors = reducer.fit_transform(vectors.detach().numpy())
        return umap_vectors[:, 0], umap_vectors[:, 1]
    else:
        raise NotImplementedError


def generate_activations(model: lens.HookedTransformer, 
                         prompts: List[str], 
                         components: List[Component], 
                         pos: int=-1, 
                         batch_size: int=32, 
                         reduce_mean=False, 
                         total_positions=5): 
    """
    Generate activations for a list of given components in a model, by passing a list of prompts through the model.

    Args:
        model (torch.nn.Module): The model to generate activations from.
        prompts (List[str]): The prompts to generate activations for.
        components (List[Component]): The components to extract activations from.
        pos (int): The position of the component to extract activations from. Default is -1.
        batch_size (int): The batch size for generating activations. Default is 32.
        reduce_mean (bool): If True, only calculate the mean of each components activation across prompts. Default is False. Use to save memory.
        total_positions (int): The total number of positions in the prompt template. Default is 5 for arithmetic prompts.
    Returns:
        List[torch.Tensor]: A list of activations, one for each components, each of shape (len(prompts), optional_pos_dim, embed_dim).
    """
    # Initialize the activations tensor as zeros for each component
    if pos is None:
        pos = slice(0, total_positions)
        if reduce_mean:
            activations = [torch.zeros(total_positions, get_hook_dim(model, component.hook_name)) for component in components]
        else:
            activations = [torch.zeros(len(prompts), total_positions, get_hook_dim(model, component.hook_name)) for component in components]
    else:
        if reduce_mean:
            activations = [torch.zeros(get_hook_dim(model, component.hook_name)) for component in components]
        else:
            activations = [torch.zeros(len(prompts), get_hook_dim(model, component.hook_name)) for component in components]

    # Run batched forward passes through the model, saving the activations of the requested components for each prompt (or sum across prompts in case of mean reduction)
    dataloader = torch.utils.data.DataLoader(prompts, batch_size=batch_size, shuffle=False)
    for i, batch in enumerate(tqdm(dataloader)):
        _, cache = model.run_with_cache(batch)
        for j, component in enumerate(components):
            if component.head_idx is None:
                # Component is an MLP
                if reduce_mean:
                    # Sum up the previous sum with the current activations
                    activations[j] += cache[component.valid_hook_name()][:, pos, :].sum(dim=0).to(activations[j].device)
                else:
                    # Save all activations as is
                    activations[j][i*batch_size:(i+1)*batch_size] = cache[component.valid_hook_name()][:, pos, :]
            else:
                # Component is an attention head
                if 'pattern' in component.valid_hook_name():
                    # Pattern component has a different shape
                    act = cache[component.valid_hook_name()][:, component.head_idx, pos, :]
                else:
                    act = cache[component.valid_hook_name()][:, pos, component.head_idx, :]
                
                if reduce_mean:
                    activations[j] += act.sum(dim=0).to(activations[j].device)
                else:
                    activations[j][i*batch_size:(i+1)*batch_size] = act
        
        # Make sure to avoid GPU memory overflow
        del cache
        torch.cuda.empty_cache()

    if reduce_mean:
        # Divide by the number of prompts to get the mean activation
        for k in range(len(activations)):
            activations[k] = activations[k] / len(prompts)

    return activations


def generate_random_circuit(model: lens.HookedTransformer, mlp_count: int, head_count: int, seed: int = 42):
    """
    Generate a random circuit with a given number of MLPs and heads.
    Args:
        mlp_count (int): The number of MLPs to include in the circuit.
        head_count (int): The number of heads to include in the circuit.
        seed (int): The seed to use for the random number generator.
    Returns:
        Circuit (lens.HookedTransformer): The random circuit.
    """
    circuit = Circuit(model.cfg)
    random.seed(seed)
    mlps = random.sample(range(model.cfg.n_layers), mlp_count)
    for mlp in mlps:
        circuit.add_component(Component('mlp_post', layer=mlp))
    
    heads = random.sample([(l, h) for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)], head_count)
    for (l,h) in heads:
        circuit.add_component(Component('z', layer=l, head=h))
    
    return circuit


def get_hook_dim(model, hook_name):
    """
    Get the size of the output of a specific hook in the model's calculation.
    """
    return {
        lens.utils.get_act_name('hook_embed', 0): model.cfg.d_model,
        
        lens.utils.get_act_name('v_input', 0): model.cfg.d_model,
        lens.utils.get_act_name('k_input', 0): model.cfg.d_model,
        lens.utils.get_act_name('q_input', 0): model.cfg.d_model,
        lens.utils.get_act_name('pattern', 0): 5, # Works in the our arithmetic prompt context, where each prompt contains 5 tokens (including BoS)
        lens.utils.get_act_name('z', 0): model.cfg.d_head,
        lens.utils.get_act_name('result', 0): model.cfg.d_model,
        lens.utils.get_act_name('attn_out', 0): model.cfg.d_model,
        
        lens.utils.get_act_name('mlp_post', 0): model.cfg.d_mlp,
        lens.utils.get_act_name('mlp_in', 0): model.cfg.d_model,
        lens.utils.get_act_name('mlp_out', 0): model.cfg.d_model,

        lens.utils.get_act_name('resid_pre', 0): model.cfg.d_model,
        lens.utils.get_act_name('resid_post', 0): model.cfg.d_model,        
    }[lens.utils.get_act_name(hook_name, 0)]


def safe_eval(prompt):
    """
    Wrapper for eval function to avoid throwing exceptions where dividing by zero.
    """
    try:
        return int(eval(prompt))
    except ZeroDivisionError as e:
        return torch.nan


def most_significant_wildcard_patterns(numbers, min_occurrences, k_patterns=3):
    """
    Find the most significant wildcard patterns in a list of numbers.
    For example: [100, 200, 300, 400,...] -> Pattern: [.00]

    Args:
        numbers: List of numbers to analyze
        min_occurrences: Minimum number of occurrences for a pattern to be considered significant
        k_patterns: Number of top patterns to return

    Returns:
        List of tuples (pattern, count) of the most significant wildcard patterns
    """
    str_numbers = [str(num).zfill(3) for num in numbers]
    pattern_counts = defaultdict(int)
    
    def generate_wildcards(s):
        """
        Generate all wildcard patterns (except complete wildcard) for a given string.
        For example: "123" -> ['1..', '.2.', '12.', '..3', '1.3', '.23', '123']
        """
        n = len(s)
        result = []
        # Generate all possible combinations
        for i in range(1, 2**n):
            wildcard = ''
            for j in range(n):
                if i & (1 << j):
                    wildcard += s[j]
                else:
                    wildcard += '.'
            result.append(wildcard)
        return result

    # Count patterns
    for num in str_numbers:
        for pattern in generate_wildcards(num):
            pattern_counts[pattern] += 1
    
    # Filter and sort patterns
    significant_patterns = {pattern: count for pattern, count in pattern_counts.items() if count >= min_occurrences}
    most_significant_patterns = sorted(significant_patterns.items(), key=lambda x: (-x[1], -len(x[0].replace('.', '')))) # Prioritize patterns with higher occurrence counts and more specific patterns (fewer wildcards)
    return most_significant_patterns[:k_patterns]


def get_neuron_importance_scores(model: lens.HookedTransformer, 
                                 model_name: str, 
                                 reduce_type: str = "mean+std", 
                                 operator_idx: int = 0,
                                 pos: int = -1,
                                 is_attn: bool = False):
    """
    Get a measure of importance of each neuron in the model based on the neuron indirect effect attribution scores.

    Args:
        model (lens.HookedTransformer): The model to get the neuron scores for.
        model_name (str): The name of the model.
        ranking_method (str): How to reduce the neuron effects across prompts.
        operator_idx (int): The index of the operator to get the neuron scores for.
        pos (int): The position to get the neuron scores for. Default is -1.
        is_attn (bool): If true, get the neuron scores for the attention heads, prior to multiplication with W_O. Used to check the KV hypothesis for attention heads.
    """
    def ranking_func(attribution_scores, pos):
        if reduce_type == "mean":
            return attribution_scores[:, pos].nan_to_num(0).mean(dim=0)
        elif reduce_type == "mean+std":
            return attribution_scores[:, pos].nan_to_num(0).mean(dim=0) + attribution_scores[:, pos].nan_to_num(0).std(dim=0)
        else:
            raise ValueError(f"Unknown ranking method: {reduce_type}")

    def head_ranking_func(attribution_scores, pos, head):
        return attribution_scores[:, pos, head].nan_to_num(0).mean(dim=0) + attribution_scores[:, pos, head].nan_to_num(0).std(dim=0)
        
    operator_names = ['addition', 'subtraction', 'multiplication', 'division']
    neuron_attribution_scores = torch.load(f"./data/{model_name}/{operator_names[operator_idx]}_{'attn_' if is_attn else ''}node_attribution_scores.pt")
    if is_attn:
        neurons_scores = {(layer, head): head_ranking_func(neuron_attribution_scores[f'blocks.{layer}.attn.hook_z'], -1, head) for layer in range(0, model.cfg.n_layers) for head in range(0, model.cfg.n_heads)}
    else:
        neurons_scores = {layer: ranking_func(neuron_attribution_scores[f'blocks.{layer}.mlp.hook_post'], pos) for layer in range(0, model.cfg.n_layers)}
    return neurons_scores



#### Memory leak debugging util functions ####
def enable_memory_snapshot():
    # keep a maximum 100,000 alloc/free events from before the snapshot
    torch.cuda.memory._record_memory_history(True, trace_alloc_max_entries=100_000)
    

def gpu_memory_snapshot(output_file):
    snapshot = torch.cuda.memory._snapshot()
    with open(output_file, 'wb') as f:
        pickle.dump(snapshot, f)


def monitor_out_of_memory():
    """
    Register a monitor to save a GPU memory snapshot right after an Out-Of-Memory error occurs.
    """
    enable_memory_snapshot()
    def oom_observer(device, alloc, device_alloc, device_free):
        gpu_memory_snapshot('oom_snapshot.pkl')
    torch._C._cuda_attach_out_of_memory_observer(oom_observer)


def monitor_memory(func):
    """
    Decorator wrapper to wrap a function / code block with a memory snapshot before and after it
    """
    def wrapper(*args, **kwargs):
        enable_memory_snapshot()
        func(*args, **kwargs)
        gpu_memory_snapshot('snapshot.pkl')
    return wrapper


