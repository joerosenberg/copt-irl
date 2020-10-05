import torch
from irlco.masking import generate_batch_of_sorted_element_masks

actions = torch.LongTensor([[0, 1, 2, 3],
                            [4, 3, 2, 1]])
source_sequence_length = 5
target_sequence_length = actions.shape[1] + 1
batch_size = actions.shape[0]
nb_heads = 2
masks = generate_batch_of_sorted_element_masks(actions, source_sequence_length, nb_heads).cpu()
print(masks)

scores = torch.randn((batch_size, target_sequence_length, source_sequence_length))
idx = torch.arange(0, batch_size * nb_heads, nb_heads, dtype=torch.long)
scores = scores + masks[idx, :, :]
probs = torch.softmax(scores, dim=2)
print(scores)
print(probs)