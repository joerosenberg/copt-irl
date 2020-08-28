import torch
import torch.nn as nn


transformer = nn.Transformer(d_model=4, nhead=4, num_decoder_layers=6, num_encoder_layers=6)

# src has shape (source seq length, batch size, nb features)
# for batch of one sequence of length 10 of vectors in R4:
src = torch.randn(10, 1, 4)

# tgt has shape (tgt seq length, batch size, nb features)
tgt = torch.randn(8, 1, 4)

print(transformer.forward(src, tgt))

# Now try masking:
mask = transformer.generate_square_subsequent_mask(8)
print(transformer.forward(src, tgt, tgt_mask=mask))