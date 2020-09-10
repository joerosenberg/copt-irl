"""
Implementation of multi-head attention from 'scratch' (using base pytorch functions.)
"""
import torch
from torch.nn import Module, Parameter
from torch.nn.init import xavier_uniform
from torch.nn.functional import linear

class MultiheadAttention(Module):
    """
    Implementation of multihead attention (for the easier case where the query, key and value dimension are the same).
    """
    def __init__(self, embed_dim: int, nb_heads: int):
        super(MultiheadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.k_dim = embed_dim
        self.v_dim = embed_dim

        self.nb_heads = nb_heads
        self.head_dim = embed_dim / nb_heads

        # The embedding dimension must be divisible by the number of heads.
        assert self.head_dim * nb_heads == embed_dim

        # Weights for projections
        self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.q_proj_weight = Parameter(torch.empty(embed_dim, embed_dim))
        self.k_proj_weight = Parameter(torch.empty(embed_dim, self.k_dim))
        self.v_proj_weight = Parameter(torch.empty(embed_dim, self.v_dim))

        self.in_proj

        # Initialise parameters
        self._reset_parameters()

    def _reset_parameters(self):
        r"""
        Initialises/resets parameters using Xavier (aka Glorot) uniform initialisation:
        Xavier initialisation samples values for each layer's parameters from U(-a, a), where

        .. math::
            a = \text{gain} \times \sqrt{\frac{6}{\text{fan_in} + \text{fan_out}}}.

        fan_in is the number of connections going into that layer, fan_out is the number of connections leaving the
        layer.
        :return:
        """
        self.in_proj_weight = xavier_uniform(self.in_proj_weight)
        self.q_proj_weight = xavier_uniform(self.q_proj_weight)
        self.k_proj_weight = xavier_uniform(self.k_proj_weight)
        self.v_proj_weight = xavier_uniform(self.v_proj_weight)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        r"""

        :param query:
        :param key:
        :param value:
        :param key_padding_mask:
        :param attn_mask:
        :return:
        """

        target_len, batch_size, _ = query.size()

        # Scale dot product by inverse square root of the key dimension, as in Scaled Dot-Product Attention from
        # Attention is All You Need, Vaswani et al. 2017
        scaling = float(self.head_dim) ** -0.5

