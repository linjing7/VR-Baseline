import torch
from einops import rearrange
x = torch.randint(0, 10, (1, 5, 3, 16, 20))
x_patch = rearrange(x, 'n t c (h p1) (w p2)-> (n h w) t c p1 p2', p1=4, p2=4)

print(x[0, 1, 1])

print(x_patch[0, 1, 1])
