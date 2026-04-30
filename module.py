import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift

class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    """
    SIGReg is a distribution-shaping regularizer that encourages the latent embeddings to be isotropic Gaussian distributed (spherical)
    using random 1D projections and an Epps-Pulley-style normality statistic.

    High-level idea:
    1. samples many random directions (unit vectors)
    2. project each frame embedding onto the unit vectors
    3. checks whether those 1D projected samples look Gaussian based on the characteristic function
    4. averages the mismatch into one scalar penalty

    """

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj # number of random directions to test, more projections means better coverage but more compute
        # linespace creates a 1D tensor containing a specified number of points evenly spaced between a start and end value. Gaussian test are performed on these points.
        # In this case, start: 0, end: 3, number of points: 17
        # Knots: how frequently we samples the points (performs the Epps-Pulley test)? tradeoff between compute and approximation accuracy
        t = torch.linspace(0, 3, knots, dtype=torch.float32) # t: [0.0000, 0.1875, 0.3750, 0.5625, ..., 3.0000]
        dt = 3 / (knots - 1) # distance between neighbour points: 0.1875

        # To find area under the curve for random sampled points, we use *trapezidal rule* to approximate the area,
        # weights are the coefficients for a weighted sum over the t axis, that pattern says: interior knots count twice because they are shared by two trapezoids according 
        # torch.full(shape, value): make a tensor with specific shape (17,) and filled with a single, uniform value (2 * dt = 0.3750), later change the first and last item to 0.1875
        # weights: [0.1875, 0.3750, 0.3750, 0.3750, ..., 0.1875]
        
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt 

        # Target Characteristic function (CF) for gaussian distribution: For each element in t, the target is window[i] = exp(-t[i]^2 / 2)
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        # trapezoidal weights multiplied elementwisely by exp(-t^2/2) -> Epps-Pulley Weighting function: how much each frequency knot contributes to the overall loss
        # [0.1875·1.0000 (0.1875), 0.375·0.9862 (0.3698), 0.375·0.9321 (0.3495), ..., 0.1875·0.0111 (0.0021)]
        self.register_buffer("weights", weights * window)

        # TODO: Q: what's epps pulley test? Why times the weight? Why even bother calculating the area under the curve?

    def forward(self, proj):
        """
        Intuition: “At each of the 4 timesteps, if I look across the 128 embeddings in the batch, 
        and I examine them through 1024 random 1D views, do those 1D samples look Gaussian?”

        projection: (B, T, D) -> (T, B, D), because we want to compute the Gaussianity test per timestep 
        (e.g. at timestep 0, do the embeddings across the batch look Gaussian, at timestep 1, do they look Gaussian? Then average those scores across time)
        It's better than flattening all (B*T) together, because different timesteps may produce different distributions.

        Tensor Shape:
        (T, B, D) = (4, 128, 192)
        
        T (Sequence Length): num_steps = wm.num_preds + wm.history_size = 1 + 3 = 4  lewm.yaml->history size, pusht.yaml->num_steps
        B (Batch size): 128 lewm.yaml -> batch_size
        D (Dimension): 192  lewm.yaml -> embed_dim  
        """
        # sample random projections: 
        # dim: 192, num_proj: 1024 Each column is a random direction in the 192 dim space, there are 1024 such directions
        # A[:, 0] is a random 192-D vector, A[:, 1] is another random 192-D vector
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0)) # computes the L2 norm of each column, now each A[:, p] is a unit vector
        # compute the epps-pulley statistic:
        
        # 1. project the embedding to 1D scalar by taking dot product with the unit vector:
        # proj(4, 128, 192) * A(192, 1024) = (4, 128, 1024)
        # meaning: for each timestep, for each batch item, for each random projection, compute one 1D scalar dot product
        # 2. Expand across the 17 knot values:
        # .unsqueeze(-1): add one item in the last position of the dot product, it has the shape of (4, 128, 1024, 1)
        # * self.t: for each projected scalar, multiply it by every knot value from self.t(17,): [0, 0.1875, 0.375, ..., 3]
        # x_t shape now is (4, 128, 1024, 17)
        x_t = (proj @ A).unsqueeze(-1) * self.t

        # 2. Compute cos and sin over the batch:
        # x_t.cos().mean(-3): average over dim 1 (the batch dim), so x_t.cos()/sin().mean(-3) has shape (4, 1024, 17)
        # meaning: For each timestep, for each random projection, for each knot value(t), calculate knot(t) * scalar(s) pair for each sample (128 values), then take cos/sin for each value
        # finally, average across 128 batch samples (because expectations are estimated by sample averages), this yields the empirical characteristic function (CF) across the batch (17,)

        # 3. Compare to the gaussian target:
        # For each time(T), project(P) pair, it has the shape of (17,), it's used to compare against the self.phi (target CF function), take the loss and squares it
        # So err.shape = (4, 1024, 17) -> cos value (real part) should match phi, sin value (imaginary part) should near 0.  
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()

        # 4. Weighted sum over the 17 knots:
        # err (4, 1024, 17) @ self.weights (17,): weighted sum over the last dimension, result (4, 1024)
        # then rescale the scalar value with 128(batch size) from averged diff to the total diff for one batch to match the formula
        statistic = (err @ self.weights) * proj.size(-2)

        # 5. Epps-Pulley statistic Test: Average everything to one scalar loss:
        # .mean(): averages over the 4 timesteps, and 1024 random projections, output a scalar with shape () (e.g. 0.2946)
        return statistic.mean()
    
class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None):

        if hasattr(self, "input_proj"):
            x = self.input_proj(x)

        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)

        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x

class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        """
        x: (B, T, d)
        c: (B, T, act_dim)
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x
