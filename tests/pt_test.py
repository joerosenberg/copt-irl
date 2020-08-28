from irlco.pointer_transformer import PointerTransformer
import torch
import torch.nn as nn
import numpy as np

transformer = PointerTransformer(4, 4, 6, 6, 1024)
mask = transformer.generate_square_subsequent_mask(10)

# src has shape (source seq length, batch size, nb features)
# for batch of one sequence of length 10 of vectors in R4:
src = torch.randn(10, 1, 4)

# tgt has shape (tgt seq length, batch size, nb features)
tgt = torch.randn(10, 1, 4)

memory_mask = torch.ones(10, 10, dtype=torch.bool)
memory_mask[0] = False

seq = []

for i in range(9):
    probs = torch.flatten(transformer.forward(src, tgt, memory_mask=memory_mask, tgt_mask=mask)).detach().numpy()
    choice = np.random.choice(10, p=probs)
    seq.append(choice)
    memory_mask[i+1] = memory_mask[i]
    memory_mask[i+1, choice] = True

print(seq)
