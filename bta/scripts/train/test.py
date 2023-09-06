import torch

def mysoftmax(values, gain=1.0):
 
    # Computing element wise exponential value
    exp_values = torch.exp(gain*values)
 
    # Computing sum of these values
    exp_values_sum = torch.sum(exp_values)
 
    # Returing the softmax output.
    return exp_values/exp_values_sum

logits = torch.tensor([0.7, 0.2, 0.1])

soft_ = torch.softmax(logits, -1)
logsumexp = logits - logits.logsumexp(dim=-1, keepdim=True)
my_soft_ = mysoftmax(logits, 0.001)
print(soft_, logsumexp.exp(), my_soft_)