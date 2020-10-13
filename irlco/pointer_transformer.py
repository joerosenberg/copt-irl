import torch
import torch.nn as nn
from torch.nn import TransformerDecoderLayer, TransformerDecoder, Transformer, TransformerEncoder, \
    TransformerEncoderLayer, Linear
from torch import Tensor
from typing import Optional, Callable, Tuple


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
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=2)

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
        weights = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                      key_padding_mask=memory_key_padding_mask)[1]

        # Clip weights into [-10, 10] using tanh to get scores for actions
        scores = self.tanh(weights) * 10.0

        # Mask actions and calculate probabilities (masking for actions that have already been selected)
        bsize = tgt.shape[1]
        idx = torch.arange(0, bsize * self.nhead, self.nhead, dtype=torch.int64)
        scores = scores + memory_mask[idx, :, :]
        # Take softmax over last index, since that is the index corresponding to the source sequence and we want a
        # probability distribution over the source sequence.
        probs = self.softmax(scores)

        # We don't take softmax here because the log probabilities are used when calculating the policy objective
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

    All we have to do to get the desired output is modify the last decoder layer to output the scores QK^T / \sqrt{d_k}
    directly, rather than using them as weights.
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


class TwinDecoderPointerTransformer(Transformer):
    """
    Transformer with two separate decoders (outputs) - one outputs action log probabilities, the other outputs learned
    rewards.
    """
    def __init__(self, d_input: int, d_model: int, nhead: int, num_encoder_layers: int, num_decoder_layers: int, dim_feedforward: int,
                 dropout: float, activation: str = "relu", shared_encoder=True):
        super(Transformer, self).__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        encoder_norm = nn.LayerNorm(d_model)

        # Input embedding
        self.input_embedding = Linear(d_input, d_model)

        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        if shared_encoder:
            self.reward_encoder = self.encoder
        else:
            self.reward_encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        pointer_layer = PointerTransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        decoder_norm = nn.LayerNorm(d_model)

        # We have two decoders: one for producing action log-probs, the other for producing rewards.
        self.action_decoder = PointerTransformerDecoder(decoder_layer, pointer_layer, num_decoder_layers, decoder_norm)
        self.reward_decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        self.reward_feedforward = nn.Sequential(
            Linear(d_model, dim_feedforward),
            nn.ReLU(),
            Linear(dim_feedforward, dim_feedforward),
            nn.ReLU(),
            Linear(dim_feedforward, 1)
        )

        self._reset_parameters()

        self.d_model = d_model
        self.d_input = d_input
        self.nhead = nhead

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        if src.size(1) != tgt.size(1):
            raise RuntimeError("the batch number of src and tgt must be equal")

        if src.size(2) != self.d_input or tgt.size(2) != self.d_input:
            raise RuntimeError("the feature number of src and tgt must be equal to d_input")

        embedded_src = self.input_embedding(src)
        embedded_tgt = self.input_embedding(tgt)

        memory = self.encoder(embedded_src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        action_probs = self.action_decoder(embedded_tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                           tgt_key_padding_mask=tgt_key_padding_mask,
                                           memory_key_padding_mask=memory_key_padding_mask)
        return action_probs

    def shaping_terms(self, src: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                      memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                      tgt_key_padding_mask: Optional[Tensor] = None,
                      memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        embedded_src = self.input_embedding(src)
        embedded_tgt = self.input_embedding(tgt)
        memory = self.reward_encoder(embedded_src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        decoder_output = self.reward_decoder(embedded_tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                             tgt_key_padding_mask=tgt_key_padding_mask,
                                             memory_key_padding_mask=memory_key_padding_mask)
        shaping_terms = self.reward_feedforward(decoder_output)
        return shaping_terms


class KoolModel(torch.nn.Module):
    def __init__(self):
        super(torch.nn.Module, self).__init__()
        # d_h = 256
        # d_k = 256 / nheads = 256 / 8 = 32
        encoder_layer = TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=1024, dropout=0, activation='relu')
        encoder_norm = nn.BatchNorm1d(num_features=256)

        # Encoder input embedding
        self.encoder_input_embedding = Linear(in_features=4, out_features=256)
        self.encoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=3, norm=encoder_norm)

        # Decoder
        # Placeholder parameters for first decoding step
        self.initial_decoding_state1 = torch.nn.Parameter(data=torch.empty(256))
        self.initial_decoding_state2 = torch.nn.Parameter(data=torch.empty(256))

        # Query, key and value projections for the first decoder layer (project for all heads simultaneously)
        # in = 3*d_h = 3*256 for query, in = d_h = 256 for key and value,
        # out = d_h = nb_heads * d_k = 256.
        # We have nb_heads = 8 for the first decoder layer
        self.query_proj1 = Linear(in_features=3 * 256, out_features=256, bias=False)
        self.key_proj1 = Linear(in_features=256, out_features=256, bias=False)
        self.value_proj1 = Linear(in_features=256, out_features=256, bias=False)
        # Output projections for first layer - have one projection for each head
        self.output_projs1 = [Linear(in_features=256 // 8, out_features=256, bias=False) for _ in range(8)]

        # Query and key projections for the second decoder layer, which has only one head.
        self.query_proj2 = Linear(in_features=256, out_features=256, bias=False)
        self.key_proj2 = Linear(in_features=256, out_features=256, bias=False)

        self._reset_parameters()

    def encode(self, base_pairs):
        # base_pairs shape: (episode_length, batch_size, 4)
        embedded_input = self.encoder_input_embedding(base_pairs)
        return self.encoder(embedded_input)

    def forward(self, embedded_base_pairs, prev_actions):
        # embedded_base_pairs shape: (episode_length, batch_size, d_h=256)
        # prev_actions shape: (batch_size, t)
        batch_size, t = prev_actions.shape
        episode_length = embedded_base_pairs.shape[0]

        # Compute context nodes
        hbar = torch.mean(embedded_base_pairs, dim=0) # shape (batch_size, d_h=256)
        if t == 0:
            context = torch.cat((hbar, self.initial_decoding_state1.unsqueeze(0).repeat(batch_size, 1),
                                 self.initial_decoding_state2.unsqueeze(0).repeat(batch_size, 1)), dim=1)
        else:
            prev_action_embeddings = embedded_base_pairs[prev_actions[:, -1], torch.arange(0, batch_size), :]
            first_action_embeddings = embedded_base_pairs[prev_actions[:, 0], torch.arange(0, batch_size), :]
            # shapes of above are (batch_size, d_h=256)

            context = torch.cat((hbar, prev_action_embeddings, first_action_embeddings), dim=1)
        # shape of context is (batch_size, 3*256)

        # First decoder layer:
        # Project queries, keys and values
        queries1 = self.query_proj1(context) # shape is (batch_size, 256)
        keys1 = self.key_proj1(embedded_base_pairs) # shape is (episode_length, batch_size, 256)
        values1 = self.value_proj1(embedded_base_pairs) # shape is (episode_length, batch_size, 256)

        # Create mask for attention scores: has shape (episode_length, batch_size)
        # entry i, j = -infty if action i has been taken in episode j
        mask = torch.zeros((episode_length, batch_size))
        for j in range(batch_size):
            mask[prev_actions[j, :], j] = float('-inf')

        context2 = torch.zeros((batch_size, 256))
        # Compute attention scores u and outputs for each head
        for i in range(8):
            # attention_scores has shape (episode_length, batch_size)
            attention_scores = torch.sum(queries1[:, i*32:(i+1)*32] * keys1[:, :, i*32:(i+1)*32], dim=2) / (32**0.5)
            # mask attention scores according to previously taken actions
            attention_scores += mask
            attention_weights = torch.softmax(attention_scores, dim=0) # shape is still (episode_length, batch_size)
            # use weights to weight values: shape of weighted_sum is (batch_size, 32)
            weighted_sum = torch.sum(values1[:, :, i*32:(i+1)*32] * attention_weights.unsqueeze(2), dim=0)
            head_output = self.output_projs1[i](weighted_sum) # shape is (batch_size, 256)
            context2 += head_output

        # Second decoder layer:
        queries2 = self.query_proj2(context2) # shape is (batch_size, 256)
        keys2 = self.key_proj2(embedded_base_pairs) # shape is (episode_length, batch_size, 256)

        # shape of final attention scores is (episode_length, batch_size)
        final_attention_scores = torch.sum(queries2 * keys2, dim=2) / (256**0.5)
        # clip and mask to get log-probabilities
        log_probabilities = torch.tanh(final_attention_scores) * 10 + mask # shape still (episode_length, batch_size)
        # softmax to get probabilities
        action_probabilities = torch.softmax(log_probabilities, dim=0) # shape still (episode_length, batch_size)
        return action_probabilities
