import torch
import numpy as np
from transformer_lens import HookedTransformer
from typing import List, Optional


def extract_residual_stream(
    model: HookedTransformer,
    prompt: str,
    layers: Optional[List[int]] = None,
    token_pos: int = -1
) -> np.ndarray:
    """
    Extract residual stream activations for a given prompt.

    Args:
        model: Loaded HookedTransformer model
        prompt: Input text
        layers: List of layer indices to extract. None = all layers.
        token_pos: Token position to extract. -1 = last token.

    Returns:
        activations: np.ndarray of shape (n_layers, d_model)
    """
    if layers is None:
        layers = list(range(model.cfg.n_layers))

    with torch.no_grad():
        _, cache = model.run_with_cache(prompt)

    activations = []
    for layer in layers:
        resid = cache[f"blocks.{layer}.hook_resid_post"]
        # resid shape: (batch, seq_len, d_model)
        activation = resid[0, token_pos, :].cpu().numpy()
        activations.append(activation)

    return np.array(activations)  # shape: (n_layers, d_model)


def extract_batch(
    model: HookedTransformer,
    prompts: List[str],
    layers: Optional[List[int]] = None,
    token_pos: int = -1
) -> np.ndarray:
    """
    Extract residual stream for a list of prompts.

    Returns:
        activations: np.ndarray of shape (n_prompts, n_layers, d_model)
    """
    all_activations = []
    for prompt in prompts:
        act = extract_residual_stream(model, prompt, layers, token_pos)
        all_activations.append(act)

    return np.array(all_activations)  # shape: (n_prompts, n_layers, d_model)