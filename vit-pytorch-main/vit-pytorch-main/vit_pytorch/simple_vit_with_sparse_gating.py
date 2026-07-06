import torch
from torch import cat
from torch.nn import Module, ModuleList, Sequential, Linear, LayerNorm, GELU, Softmax, Identity, Parameter
from torch.nn.functional import pad

from einops import rearrange, repeat, pack, unpack, reduce, einsum
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

def divisible_by(num, den):
    return (num % den) == 0

def posemb_sincos_2d(h, w, dim, temperature = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing = 'ij')
    assert divisible_by(dim, 4), 'feature dimension must be multiple of 4 for sincos emb'

    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = temperature ** -omega

    y, x = y.flatten(), x.flatten()

    y = torch.outer(y, omega)
    x = torch.outer(x, omega)

    pe = cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = -1)
    return pe.type(dtype)

# classes

class FeedForward(Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = Sequential(
            LayerNorm(dim),
            Linear(dim, hidden_dim),
            GELU(),
            Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)

class Attention(Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dim_mask = 4, num_register_tokens = 4):
        super().__init__()
        inner_dim = dim_head *  heads
        mask_inner_dim = dim_mask * heads

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.mask_scale = dim_mask ** -0.5
        self.num_register_tokens = num_register_tokens

        self.norm = LayerNorm(dim)
        self.attend = Softmax(dim = -1)

        self.split_dims = (inner_dim, inner_dim, inner_dim, mask_inner_dim, mask_inner_dim)

        self.to_qkv_and_mask = Linear(dim, sum(self.split_dims), bias = False)

        self.to_out = Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        q, k, v, q_mask, k_mask = self.to_qkv_and_mask(x).split(self.split_dims, dim = -1)

        q, k, v, q_mask, k_mask = (rearrange(t, 'b n (h d) -> b h n d', h = self.heads) for t in (q, k, v, q_mask, k_mask))

        # slice out registers

        k_mask_patches = k_mask[:, :, self.num_register_tokens:]

        mask_sim = einsum(q_mask, k_mask_patches, 'b h i d, b h j d -> b h i j') * self.mask_scale

        mask_prob = mask_sim.sigmoid()
        mask_log = log(mask_prob)

        # pad registers with 0s on left so they act as attention sinks

        mask_log = pad(mask_log, (self.num_register_tokens, 0), value = 0.)

        sim = einsum(q, k, 'b h i d, b h j d -> b h i j') * self.scale

        # add log of mask probabilities to similarities

        sim = sim + mask_log

        attn = self.attend(sim)

        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out), mask_prob

class Transformer(Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dim_mask, num_register_tokens):
        super().__init__()
        self.norm = LayerNorm(dim)

        self.layers = ModuleList([ModuleList([
            Attention(dim, heads = heads, dim_head = dim_head, dim_mask = dim_mask, num_register_tokens = num_register_tokens),
            FeedForward(dim, mlp_dim)
        ]) for _ in range(depth)])

    def forward(self, x):
        mask_probs = []

        for attn, ff in self.layers:
            attn_out, mask_prob = attn(x)

            x = attn_out + x
            x = ff(x) + x

            mask_probs.append(mask_prob)

        return self.norm(x), mask_probs

class SimpleViTWithSparseGating(Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        num_register_tokens = 4,
        channels = 3,
        dim_head = 64,
        dim_mask = 4,
        sparsity_loss_type = 'thresholded_l1',
        sparsity_threshold = 0.05
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        self.patch_size = patch_height, patch_width = pair(patch_size)

        assert divisible_by(image_height, patch_height) and divisible_by(image_width, patch_width), 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            LayerNorm(patch_dim),
            Linear(patch_dim, dim),
            LayerNorm(dim),
        )

        self.num_register_tokens = num_register_tokens
        self.register_tokens = Parameter(torch.randn(num_register_tokens, dim))

        self.pos_embedding = posemb_sincos_2d(
            h = image_height // patch_height,
            w = image_width // patch_width,
            dim = dim,
        )

        self.sparsity_loss_type = sparsity_loss_type
        self.sparsity_threshold = sparsity_threshold

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dim_mask, num_register_tokens)

        self.pool = 'mean'
        self.to_latent = Identity()

        self.linear_head = Linear(dim, num_classes)

    def forward(self, img, return_loss = False):
        batch, device = img.shape[0], img.device

        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype = x.dtype)

        r = repeat(self.register_tokens, 'n d -> b n d', b = batch)

        # prepend registers

        x, ps = pack([r, x], 'b * d')

        # attend

        x, mask_probs = self.transformer(x)

        # discard registers

        _, x = unpack(x, ps, 'b * d')

        # pool patches

        x = reduce(x, 'b n d -> b d', 'mean')

        # to logits

        x = self.to_latent(x)
        logits = self.linear_head(x)

        if not return_loss:
            return logits

        # calculate sparsity loss per image

        p_mean = reduce(mask_probs, 'd b h i j -> b', 'mean')

        if self.sparsity_loss_type == 'l1':
            loss = p_mean.mean()
        elif self.sparsity_loss_type == 'thresholded_l1':
            loss = (p_mean - self.sparsity_threshold).relu().mean()
        else:
            raise ValueError(f"Unknown sparsity loss type: {self.sparsity_loss_type}")

        return logits, loss

if __name__ == '__main__':
    v = SimpleViTWithSparseGating(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        num_register_tokens = 4,
        dim_mask = 4,
        sparsity_loss_type = 'thresholded_l1',
        sparsity_threshold = 0.05
    )

    img = torch.randn(2, 3, 256, 256)

    logits = v(img)
    assert logits.shape == (2, 1000)

    logits, loss = v(img, return_loss = True)
    assert logits.shape == (2, 1000)
    assert loss.ndim == 0

    loss.backward()

    v_l1 = SimpleViTWithSparseGating(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        num_register_tokens = 4,
        dim_mask = 4,
        sparsity_loss_type = 'l1'
    )

    logits, loss_l1 = v_l1(img, return_loss = True)
    loss_l1.backward()
