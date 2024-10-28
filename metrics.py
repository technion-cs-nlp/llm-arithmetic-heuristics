import torch


def logit_diff(logits: torch.Tensor,
                clean_labels: torch.Tensor,
                corrupt_labels: torch.Tensor):
    return logits.gather(1, clean_labels) - logits.gather(1, corrupt_labels)


def indirect_effect(pre_patch_probs: torch.Tensor,
                    post_patch_probs: torch.Tensor,
                    clean_labels: torch.Tensor, 
                    corrupt_labels: torch.Tensor):
    """
    Measure indirect effect of a patch on probabilities, as described in Eq. 2 of "Understanding Arithmetic 
    Reasoning in Language Models using Causal Mediation Analysis".
    
    Args:
        pre_patch_probs (torch.Tensor (batch, vocab_size)): The probabilities before patching.
        post_patch_probs (torch.Tensor (batch, vocab_size)): The probabilities after patching.
        clean_labels (torch.Tensor(batch, 1)): The labels of the clean answers.
        corrupt_labels (torch.Tensor(batch, 1)): The label of the corrupt answers.

    Returns:
        torch.Tensor((batch,), dtype=torch.float32): The indirect effects for each prompt in the batch. The IE is not limited in magnitude.
    """
    a = (post_patch_probs.gather(1, corrupt_labels) - pre_patch_probs.gather(1, corrupt_labels)) / pre_patch_probs.gather(1, corrupt_labels)
    b = (pre_patch_probs.gather(1, clean_labels) - post_patch_probs.gather(1, clean_labels)) / post_patch_probs.gather(1, clean_labels)
    return (a + b).squeeze(1) / 2
