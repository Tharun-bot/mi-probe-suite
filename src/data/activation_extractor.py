# import torch
# import numpy as np
# from transformer_lens import HookedTransformer
# from typing import List, Optional


# def extract_residual_stream(
#     model: HookedTransformer,
#     prompt: str,
#     layers: Optional[List[int]] = None,
#     token_pos: int = -1
# ) -> np.ndarray:
#     """
#     Extract residual stream activations for a given prompt.

#     Args:
#         model: Loaded HookedTransformer model
#         prompt: Input text
#         layers: List of layer indices to extract. None = all layers.
#         token_pos: Token position to extract. -1 = last token.

#     Returns:
#         activations: np.ndarray of shape (n_layers, d_model)
#     """
#     if layers is None:
#         layers = list(range(model.cfg.n_layers))

#     with torch.no_grad():
#         _, cache = model.run_with_cache(prompt)

#     activations = []
#     for layer in layers:
#         resid = cache[f"blocks.{layer}.hook_resid_post"]
#         # resid shape: (batch, seq_len, d_model)
#         activation = resid[0, token_pos, :].cpu().numpy()
#         activations.append(activation)

#     return np.array(activations)  # shape: (n_layers, d_model)


# def extract_batch(
#     model: HookedTransformer,
#     prompts: List[str],
#     layers: Optional[List[int]] = None,
#     token_pos: int = -1
# ) -> np.ndarray:
#     """
#     Extract residual stream for a list of prompts.

#     Returns:
#         activations: np.ndarray of shape (n_prompts, n_layers, d_model)
#     """
#     all_activations = []
#     for prompt in prompts:
#         act = extract_residual_stream(model, prompt, layers, token_pos)
#         all_activations.append(act)

#     return np.array(all_activations)  # shape: (n_prompts, n_layers, d_model)

#Updated version - 1

import torch
import numpy as np
from transformer_lens import HookedTransformer
from typing import List, Optional, Dict
from tqdm import tqdm


def extract_residual_stream(
    model: HookedTransformer,
    prompt: str,
    layers: Optional[List[int]] = None,
    token_pos: int = -1
) -> np.ndarray:
    """
    Extract residual stream activations for a single prompt.

    Args:
        model: Loaded HookedTransformer model
        prompt: Input text
        layers: Layer indices to extract. None = all layers.
        token_pos: Token position. -1 = last token.

    Returns:
        np.ndarray of shape (n_layers, d_model)
    """
    if layers is None:
        layers = list(range(model.cfg.n_layers))

    # Only cache what we need — saves memory
    names_filter = [f"blocks.{l}.hook_resid_post" for l in layers]

    with torch.no_grad():
        _, cache = model.run_with_cache(
            prompt,
            names_filter=names_filter  # KEY: only cache specified layers
        )

    activations = []
    for layer in layers:
        resid = cache[f"blocks.{layer}.hook_resid_post"]
        activation = resid[0, token_pos, :].cpu().numpy()
        activations.append(activation)

    return np.array(activations)  # (n_layers, d_model)


def extract_batch(
    model: HookedTransformer,
    prompts: List[str],
    layers: Optional[List[int]] = None,
    token_pos: int = -1,
    show_progress: bool = True
) -> np.ndarray:
    """
    Extract residual stream for a list of prompts.

    Returns:
        np.ndarray of shape (n_prompts, n_layers, d_model)
    """
    all_activations = []
    iterator = tqdm(prompts) if show_progress else prompts

    for prompt in iterator:
        act = extract_residual_stream(model, prompt, layers, token_pos)
        all_activations.append(act)

    return np.array(all_activations)  # (n_prompts, n_layers, d_model)

#for the confidence rate value, didnt get the expected prob.

def get_relative_confidence(model, question, correct_answer, wrong_answer):
    with torch.no_grad():
        logits = model(question)
    
    probs = torch.softmax(logits[0, -1, :], dim=-1)
    
    correct_token = model.to_single_token(correct_answer)
    wrong_token = model.to_single_token(wrong_answer)
    
    correct_prob = probs[correct_token].item()
    wrong_prob = probs[wrong_token].item()
    
    # Label = 1 if model ranks correct answer higher
    label = 1 if correct_prob > wrong_prob else 0
    confidence = correct_prob - wrong_prob
    
    return label, confidence


def get_next_token_probs(
    model: HookedTransformer,
    prompt: str
) -> Dict[str, float]:
    """
    Get top-10 next token probabilities for a prompt.
    Useful for measuring model confidence.

    Returns:
        dict of {token_string: probability}
    """
    with torch.no_grad():
        logits = model(prompt)

    probs = torch.softmax(logits[0, -1, :], dim=-1)
    top_probs, top_indices = torch.topk(probs, 10)

    return {
        model.to_single_str_token(idx.item()): prob.item()
        for idx, prob in zip(top_indices, top_probs)
    }