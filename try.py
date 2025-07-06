from einops import rearrange, repeat
import numpy as np

a = np.ones((3,6,6))
b = rearrange(a, 'b h w -> (b h) w')
print(a.shape)
print(b.shape)