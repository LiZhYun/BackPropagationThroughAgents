import torch
import numpy as np

# def mysoftmax(values, gain=1.0):
 
#     # Computing element wise exponential value
#     exp_values = torch.exp(gain*values)
 
#     # Computing sum of these values
#     exp_values_sum = torch.sum(exp_values)
 
#     # Returing the softmax output.
#     return exp_values/exp_values_sum

# logits = torch.tensor([0.7, 0.2, 0.1])

# soft_ = torch.softmax(logits, -1)
# logsumexp = logits - logits.logsumexp(dim=-1, keepdim=True)
# my_soft_ = mysoftmax(logits, 0.001)
# print(soft_, logsumexp.exp(), my_soft_)
# probs = np.array([0.1, 0.2, 0.7])
# logits = np.log(probs)
# temp = np.array([10, 1, 0.1])
# exp_values = np.exp(logits/temp)
# exp_values_sum = np.sum(exp_values)
# temp_logits = exp_values/exp_values_sum
# print(probs, temp_logits)

my_dict = {0:'0', 1:'1', 2:'2'}
for _ in range(10):
    print(my_dict.items())