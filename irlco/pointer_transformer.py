import torch
import torch.nn as nn
from torch.nn import TransformerDecoderLayer, TransformerDecoder, Transformer, TransformerEncoder, TransformerEncoderLayer
from torch import Tensor
from typing import Optional, Callable


class PointerTransformerDecoderLayer(nn.Module):
    """
    Final decoder layer in the pointer transformer. Differs from a regular transformer decoder layer only slightly -
    instead of 
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu"):
        super(PointerTransformerDecoderLayer, self).__init__()

        # First multihead attention block: uses self attention on outputs from previous layer.
        # Masked so that outputs do not attend to other outputs corresponding to future actions.
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Second multihead attention block: attends
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout)
        #self.logit = nn.Linear(d_model, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        self.nhead = nhead

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        # Instead of returning the attention-weighted sum, return the weights over the source sequence for the last
        # element of the target sequence
        weights = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[1][:,-1,:]

        # Clip weights into [-10, 10] using tanh to get logits
        logits = self.tanh(weights) * 10.0

        # Get probabilities by taking softmax of logits (with masking for actions that have already been selected)
        bsize = tgt.shape[1]
        idx = torch.arange(0, bsize * self.nhead, self.nhead, dtype=torch.int64)
        probs = self.softmax(logits + memory_mask[idx, -1, :])

        return probs


class PointerTransformerDecoder(nn.TransformerDecoder):
    def __init__(self, decoder_layer: nn.TransformerDecoderLayer, pointer_layer: PointerTransformerDecoderLayer,
                 num_layers: int, norm = None):
        super(PointerTransformerDecoder, self).__init__(decoder_layer, num_layers, norm=norm)
        self.layers.append(pointer_layer)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        return output


class PointerTransformer(Transformer):
    """
    Transformer model with modified output. Instead of outputting a vector that has the same dimension regardless of the
    input set, it outputs a probability distribution over the elements of the input set (so the output dimension will be
    the same as the size of the input set).

    This is just a convenient way of modifying the transformer for tasks where we want to output some ordering of
    the input set. Normally, the transformer has a fixed output range (e.g. at each step we output a probability
    distribution over a predetermined dictionary of words, so every output will be a vector in R^n for some fixed n).

    It's convenient because of how the attention mechanism in the transformer works - in each decoder layer of the
    transformer, we 'attend' over the input set anyway (i.e. compute attention scores for each element of the set),
    using the outputs of the encoder stack as values V and keys K.

    $$ A(Q, K, V) = \text{softmax}( QK^T / \sqrt{d_k} ) V $$

    All we have to do to get the desired output is modify the last decoder layer to output these scores directly,
    rather than
    """

    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1, activation: str = "relu") -> None:
        super(Transformer, self).__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)

        # Original Transformer model uses batch normalisation for encoder, but the 'Attention Solves Your TSP' paper
        # found that they had better results using layer normalisation instead
        # encoder_norm = nn.BatchNorm1d(d_model)
        encoder_norm = nn.LayerNorm(d_model)

        self.encoder: Callable[[...], Tensor] = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        pointer_layer = PointerTransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)

        # Same deal with the batch normalisation for the decoder layer as above - use layer normalisation instead as
        # suggested by 'Attention Solves Your TSP'.
        # decoder_norm = nn.BatchNorm1d(d_model)
        decoder_norm = nn.LayerNorm(d_model)

        self.decoder: Callable[[...], Tensor] = PointerTransformerDecoder(decoder_layer, pointer_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
