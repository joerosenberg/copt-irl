import torch
from torch import Tensor

DEVICE = torch.device('cuda')

def generate_square_subsequent_mask(size: int, device=DEVICE) -> Tensor:
    """
    Generates a mask that prevents actions attending to subsequent actions in the transformer decoder.
    (code taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
    Args:
        size: Size of the mask (i.e. length of the solution so far.)
        device: torch.device to perform computations and store tensors on.

    Returns:
        Mask for transformer decoder.

    """
    mask = torch.eq(torch.triu(torch.ones(size, size, device=device)), 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def generate_sorted_element_mask(previous_actions, input_length: int, device=DEVICE) -> Tensor:
    """
    Generates a mask that prevents actions from attending to elements of the unordered set that have already been
    placed into the ordered sequence.
    Args:
        previous_actions: List of previous actions (in order) that we need to mask
        input_length: Number of elements in the unordered sequence
        device: torch.device to perform computations and store tensors on.

    Returns:
        Memory mask of shape (nb of previous actions + 1, input sequence length) suitable for use in transformer

    """
    # Generate lower triangular matrix (creates columns for masked input elements)
    # i_th column of masked_cols is equal to the {a_i}'th column of the mask
    masked_cols = torch.tril(
        torch.ones(len(previous_actions) + 1, len(previous_actions), device=device) * float('-inf'), diagonal=-1)
    # Create empty mask
    mask = torch.zeros(len(previous_actions) + 1, input_length, device=device)
    # For each previous action, prevent further actions from attending to its corresponding input element
    mask[:, previous_actions] = masked_cols
    return mask


def generate_batch_of_sorted_element_masks(prev_actions_batch: Tensor, input_sequence_length: int, nb_heads: int,
                                           device=DEVICE) -> Tensor:
    """
    Generates a batch of masks that prevents actions from attending to elements of the encoder set that have already
    been placed into the ordered sequence. A mask is produced for each attention head and batch entry.

    Args:
        prev_actions_batch: Batch of previous actions as a tensor of shape (batch_size, nb actions taken so far)
        input_sequence_length: Length of the input to the encoder.
        device: torch.device to perform computations and store tensors on.

    Returns: Tensor of shape (nb_heads * batch_size, decoder input length, encoder input length) that is meant to
    be used as a memory mask for the transformer.

    Note that decoder input length = nb. of previous actions + 1, due to the beginning-of-sequence token.

    """
    nb_actions_taken = prev_actions_batch.shape[1]
    batch_size = prev_actions_batch.shape[0]
    # Get mask columns
    mask_cols = torch.tril(
        torch.ones(nb_actions_taken + 1, nb_actions_taken, device=device) * float('-inf'), diagonal=-1)
    # Create empty mask
    mask = torch.zeros((batch_size, nb_actions_taken + 1, input_sequence_length), device=device)
    # For each previous action, prevent further actions from attending to its corresponding input element
    # Unvectorised prototype:
    # TODO: Replace this with faster vectorised version using torch tensor operations. Probably no significant gains.
    for i in range(batch_size):
        mask[i, :, prev_actions_batch[i, :]] = mask_cols

    # Need to repeat each 2D mask nb_heads times - repeats like abcd -> aaabbbcccddd, since we want to use the same mask
    # across all heads for each sequence
    mask = torch.repeat_interleave(mask, nb_heads, dim=0)
    return mask