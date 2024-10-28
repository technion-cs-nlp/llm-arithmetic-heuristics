from dataclasses import dataclass

@dataclass
class ModelAnalysisConsts:
    max_single_token: int # The highest number n for which all numbers in [0, n] are represented by a single token
    first_heuristics_layer: int # The earliest layer where the model begins promoting the correct answer in the final position
    topk_neurons_per_layer: int # How many neurons in each middle- and late-layer MLP are required for high faithfulness?
    mlp_activations_also_negative: bool # Can mlp_post activations be negative? In models with GatedMLPs, this is True

PYTHIA_6_9B_CONSTS = ModelAnalysisConsts(max_single_token=530, first_heuristics_layer=14, topk_neurons_per_layer=200, mlp_activations_also_negative=False)
LLAMA3_70B_CONSTS = ModelAnalysisConsts(max_single_token=999, first_heuristics_layer=39, topk_neurons_per_layer=400, mlp_activations_also_negative=True)
LLAMA3_8B_CONSTS = ModelAnalysisConsts(max_single_token=999, first_heuristics_layer=16, topk_neurons_per_layer=200, mlp_activations_also_negative=True)
GPTJ_CONSTS = ModelAnalysisConsts(max_single_token=520, first_heuristics_layer=17, topk_neurons_per_layer=200, mlp_activations_also_negative=False)