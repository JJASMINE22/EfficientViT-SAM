import torch
from torch import nn
from torch.nn import functional as F

from flash_attn import flash_attn_func

from typing import Union, Optional, Tuple


def flash_attention_forward(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dropout_p: float = 0.,
        softmax_scale: Optional[float] = None
):
    dtype = q.dtype

    q = q.transpose(1, 2).to(torch.bfloat16)
    k = k.transpose(1, 2).to(torch.bfloat16)
    v = v.transpose(1, 2).to(torch.bfloat16)

    attn_output = flash_attn_func(
        q, k, v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale
    )

    if isinstance(attn_output, Tuple):
        attn_output = attn_output[0]

    attn_output = attn_output.to(dtype)
    attn_output = attn_output.transpose(1, 2)

    return attn_output


def sdpa_attention_forward(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dropout_p: float = 0.,
        softmax_scale: Optional[float] = None
):
    return F.scaled_dot_product_attention(
        q, k, v,
        dropout_p=dropout_p,
        scale=softmax_scale
    )


def eager_attention_forward(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dropout_p: float = 0.,
        softmax_scale: Optional[float] = None
):
    q = q * softmax_scale
    attn_scores = q @ k.transpose(-2, -1)
    attn_probs = F.softmax(attn_scores, dim=-1)
    attn_probs = F.dropout(attn_probs, p=dropout_p)
    return attn_probs @ v


ALL_ATTENTION_FUNCTIONS = {
    "flash_attention": flash_attention_forward,
    "sdpa_attention": sdpa_attention_forward,
    "eager_attention": eager_attention_forward
}

if __name__ == '__main__':
    q = torch.randn(1, 4, 32, 32).cuda()
    k = torch.randn(1, 4, 32, 32).cuda()
    v = torch.randn(1, 4, 32, 32).cuda()
    flash_attention_forward(q, k, v)
