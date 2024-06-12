import torch
from vit_pytorch import ViT
from einops import rearrange, reduce, repeat, asnumpy
from einops.layers.torch import Rearrange, Reduce

ims = torch.randn(1, 2, 3, 4)
new_ims = rearrange(tensor=ims, pattern='b c h (w1 w2) -> b (c h w1) w2', w1=2)
print(new_ims.shape)
new_ims2 = reduce(tensor=ims, pattern='b c h (w1 w2) -> (b c) w1 w2', w1=4, reduction='mean')
print(new_ims2.shape)
new_ims3 = reduce(tensor=ims, pattern='b c h (w1 w2) -> (b c) w1 w2', w1=4, reduction='max')
print(new_ims3.shape)
new_ims4 = reduce(tensor=ims, pattern='b c h (w1 w2) -> (b c) w1 w2', w1=4, reduction='min')
print(new_ims4.shape)

new_ims5 = repeat(tensor=ims, pattern='b c h w -> num b c h w', num=2)
print(new_ims5.shape)
# Just converts tensors to numpy (and pulls from gpu if necessary)
y = asnumpy(ims)
print(type(y))
