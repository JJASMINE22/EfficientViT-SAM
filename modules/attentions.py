import math
import torch

from .attention_utils import *

from .modules import ConvLayer
from .utils import val2tuple


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            downsample_rate: int = 1,
            attention_type: str = "sdpa_attention"
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

        self.attention_type = attention_type

    def _separate_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor
    ) -> torch.Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape

        scale = 1 / math.sqrt(c_per_head)
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.attention_type]
        out = attention_interface(
            q, k, v,
            softmax_scale=scale
        )

        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class LiteMLA(nn.Module):
    def __init__(
            self,
            in_size: int,
            out_size: int,
            num_head: Optional[int] = None,
            head_dim: int = 8,
            scales: Tuple[int] = (5,),
            norm=(None, "batch_norm"),
            eps: float = 1e-15
    ):
        super().__init__()

        num_head = in_size // head_dim if num_head is None else num_head

        assert in_size == head_dim * num_head

        norm = val2tuple(norm, 2)

        self.size = in_size

        self.qkv = ConvLayer(
            in_size, 3 * in_size,
            ksize=1, norm=norm[0],
        )

        self.qkv_multiscale = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        3 * in_size,
                        3 * in_size,
                        scale,
                        padding=scale // 2,
                        groups=3 * in_size,
                        bias=False
                    ),
                    nn.Conv2d(
                        3 * in_size,
                        3 * in_size,
                        1,
                        groups=3 * num_head,
                        bias=False
                    )

                ) for scale in scales]
        )

        self.nonlinearity = nn.ReLU()

        self.o = ConvLayer(
            in_size * (len(scales) + 1), out_size,
            ksize=1, norm=norm[1]
        )

        self.eps = eps

    def relu_linear_attn(
            self,
            qkv: torch.Tensor
    ):
        B, _, H, W = qkv.shape

        qkv = qkv.reshape(
            B, -1, self.size * 3, H * W
        )
        q, k, v = qkv.chunk(3, dim=2)

        q = self.nonlinearity(q)
        k = self.nonlinearity(k)

        k = k.transpose(-1, -2)
        v = F.pad(v, (0, 0, 0, 1), value=1)

        vk = torch.matmul(v, k)
        out = torch.matmul(vk, q)
        if out.dtype == torch.float16:
            # prevent overflow
            out = out.float()

        out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)

        return out.reshape(B, -1, H, W)

    def relu_quadratic_attn(
            self,
            qkv: torch.Tensor
    ):
        B, _, H, W = qkv.shape

        qkv = qkv.reshape(
            B, -1, self.size * 3, H * W
        )
        q, k, v = qkv.chunk(3, dim=2)

        q = self.nonlinearity(q)
        k = self.nonlinearity(k)

        k = k.transpose(-1, -2)
        attn = torch.matmul(k, q)
        attn = attn / (attn.sum(dim=2, keepdim=True) + self.eps)

        out = torch.matmul(v, attn)

        return out.reshape(B, -1, H, W)

    def forward(
            self,
            x: torch.Tensor
    ):

        H, W = x.shape[-2:]
        qkv = self.qkv(x)

        multi_scale_qkv = [qkv]
        for qkv_multiscale in self.qkv_multiscale:
            multi_scale_qkv.append(qkv_multiscale(qkv))

        qkv = torch.cat(
            multi_scale_qkv,
            dim=1
        )

        if H * W > self.size:
            out = self.relu_linear_attn(qkv)
        else:
            out = self.relu_quadratic_attn(qkv)

        return self.o(out)
