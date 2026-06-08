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

def get_token_position(
    model: HookedTransformer,
    prompt: str,
    target_str: str
) -> int:
    """
    Find the position of a target string in the tokenized prompt.

    Args:
        model: HookedTransformer model
        prompt: Full input prompt
        target_str: String to find (e.g., "France")

    Returns:
        Token position index

    Raises:
        ValueError if target not found
    """
    str_tokens = model.to_str_tokens(prompt)

    for i, token in enumerate(str_tokens):
        if token.strip() == target_str.strip():
            return i

    raise ValueError(
        f"Target '{target_str}' not found in tokens: {str_tokens}"
    )

def extract_at_position(
    model: HookedTransformer,
    prompt: str,
    token_pos: int,
    layers: Optional[List[int]] = None
) -> np.ndarray:
    """
    Extract residual stream at a specific token position across layers.

    Args:
        model: HookedTransformer model
        prompt: Input text
        token_pos: Token position to extract
        layers: Layer indices. None = all layers.

    Returns:
        np.ndarray of shape (n_layers, d_model)
    """
    return extract_residual_stream(
        model, prompt, layers, token_pos=token_pos
    )


def extract_all_positions(
    model: HookedTransformer,
    prompt: str,
    layer: int
) -> np.ndarray:
    """
    Extract residual stream at ALL token positions for a given layer.

    Args:
        model: HookedTransformer model
        prompt: Input text
        layer: Which layer to extract from

    Returns:
        np.ndarray of shape (seq_len, d_model)
    """
    names_filter = [f"blocks.{layer}.hook_resid_post"]

    with torch.no_grad():
        _, cache = model.run_with_cache(
            prompt,
            names_filter=names_filter
        )

    resid = cache[f"blocks.{layer}.hook_resid_post"]
    return resid[0, :, :].cpu().numpy()  # (seq_len, d_model)