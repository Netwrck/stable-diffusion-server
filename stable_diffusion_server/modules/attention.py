import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, c, n_heads, d_head):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.scale = d_head**-0.5
        self.to_q = nn.Linear(c, n_heads * d_head, bias=False)
        self.to_k = nn.Linear(c, n_heads * d_head, bias=False)
        self.to_v = nn.Linear(c, n_heads * d_head, bias=False)
        self.to_out = nn.Linear(n_heads * d_head, c)

    def forward(self, x, x_skip=None):
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        if x_skip is not None:
            k_skip = self.to_k(x_skip)
            v_skip = self.to_v(x_skip)
            k = torch.cat([k, k_skip], dim=1)
            v = torch.cat([v, v_skip], dim=1)

        q, k, v = map(
            lambda t: t.view(t.shape[0], -1, self.n_heads, self.d_head).transpose(
                1, 2
            ),
            (q, k, v),
        )
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(1, 2).reshape(out.shape[0], -1, self.n_heads * self.d_head)
        return self.to_out(out)
