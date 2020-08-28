import torch
import torch.nn as nn
from torch.nn import TransformerDecoderLayer, TransformerDecoder, Transformer, TransformerEncoder, TransformerEncoderLayer
from torch import Tensor
from typing import Optional


class PointerTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu"):
        super(PointerTransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout)
        self.logit = nn.Linear(d_model, 1)
        self.softmax = nn.Softmax(dim=0)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        logits = self.logit(tgt2)
        probs = self.softmax(logits)

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
    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1, activation: str = "relu") -> None:
        super(Transformer, self).__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        #encoder_norm = nn.BatchNorm1d(d_model)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        pointer_layer = PointerTransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        #decoder_norm = nn.BatchNorm1d(d_model)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = PointerTransformerDecoder(decoder_layer, pointer_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
