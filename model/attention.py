# Adapted from MoCapAnything V2; see LICENSE and THIRD_PARTY.md.
import torch
from torch import nn
from torch.nn import functional as F

from comfy.ops import manual_cast as ops, cast_to_input
from comfy.ldm.flux.math import apply_rope
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention


def attend(module, q, k, v, mask=None):
    return optimized_attention(
        AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v),
        q.shape[1], mask=mask, skip_reshape=True, preferred_attention=module.comfy_attention,
    )


def frame_rope(q, k, frames):
    # Interleaved RoPE, with frame positions repeated for every joint/patch.
    dim = q.shape[-1]
    freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=q.device, dtype=torch.float32) / dim))
    angle = torch.arange(frames, device=q.device, dtype=torch.float32)[:, None] * freq
    cos, sin = angle.cos(), angle.sin()
    pe = torch.stack((cos, -sin, sin, cos), dim=-1).reshape(1, 1, frames, dim // 2, 2, 2)
    return apply_rope(q, k, pe)


class GraphMultiHeadAttention(nn.Module):
    def __init__(self, d_model, nheads=4, max_path_len=5, use_tree_mask=False):
        super().__init__()
        self.nheads = nheads
        self.att_size = d_model // nheads
        self.scale = self.att_size ** -0.5
        self.use_tree_mask = use_tree_mask
        self.comfy_attention = ComfyAttention()
        self.linear_q = ops.Linear(d_model, d_model)
        self.linear_k = ops.Linear(d_model, d_model)
        self.linear_v = ops.Linear(d_model, d_model)
        self.output_layer = ops.Linear(d_model, d_model)
        self.topology_key_emb = ops.Embedding(max_path_len + 1, d_model)
        self.edge_key_emb = ops.Embedding(6, d_model)
        self.topology_query_emb = ops.Embedding(max_path_len + 1, d_model)
        self.edge_query_emb = ops.Embedding(6, d_model)

    def forward(self, q, k, v, distance, edge_attr, mask=None, tree_mask=None):
        batch, joints, _ = q.shape
        distance = distance.reshape(batch, joints, joints)
        edge_attr = edge_attr.reshape(batch, joints, joints)
        if mask is not None:
            mask = mask.reshape(batch, joints)
        if tree_mask is not None:
            tree_mask = tree_mask.reshape(batch, joints, joints)
        q = self.linear_q(q).view(batch, joints, self.nheads, self.att_size).transpose(1, 2)
        k = self.linear_k(k).view(batch, joints, self.nheads, self.att_size).transpose(1, 2)
        v = self.linear_v(v).view(batch, joints, self.nheads, self.att_size).transpose(1, 2)
        # These are query/key-dependent graph biases, not ordinary positional embeddings.
        bias = None
        for tokens, embedding, indices in (
            (q, self.topology_query_emb, distance), (k, self.topology_key_emb, distance),
            (q, self.edge_query_emb, edge_attr), (k, self.edge_key_emb, edge_attr),
        ):
            weight = embedding(torch.arange(embedding.num_embeddings, device=tokens.device), out_dtype=tokens.dtype).view(1, -1, self.nheads, self.att_size).transpose(1, 2)
            scores = (tokens @ weight.transpose(-2, -1)).gather(3, indices[:, None].expand(-1, self.nheads, -1, -1))
            bias = scores if bias is None else bias + scores
        bias = bias * self.scale
        if self.use_tree_mask and tree_mask is not None:
            bias = bias.masked_fill(~tree_mask[:, None], -torch.inf)
        if mask is not None:
            bias = bias.masked_fill(~mask[:, None, None], -torch.inf)
        # Padded query rows have no legal keys; zero them after attention.
        valid_rows = (~torch.isinf(bias)).any(dim=-1)
        bias = torch.where(valid_rows[..., None], bias, 0)
        out = attend(self, q, k, v, bias).view(batch, joints, self.nheads, self.att_size)
        out = out * valid_rows.transpose(1, 2)[..., None]
        out = out.reshape(batch, joints, -1)
        if mask is not None:
            out = out * mask[..., None]
        return self.output_layer(out)


class SimpleAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.comfy_attention = ComfyAttention()
        self.to_q = ops.Linear(dim, dim, bias=False)
        self.to_k = ops.Linear(dim, dim, bias=False)
        self.to_v = ops.Linear(dim, dim, bias=False)
        self.to_out = ops.Linear(dim, dim, bias=False)

    def forward(self, x, encoder_hidden_states=None, joint_mask=None):
        context = x if encoder_hidden_states is None else encoder_hidden_states
        batch, joints, dim = x.shape
        q = self.to_q(x).view(batch, joints, self.heads, -1).transpose(1, 2)
        k = self.to_k(context).view(batch, -1, self.heads, dim // self.heads).transpose(1, 2)
        v = self.to_v(context).view(batch, -1, self.heads, dim // self.heads).transpose(1, 2)
        mask = None
        if joint_mask is not None and encoder_hidden_states is None:
            mask = torch.zeros((batch, 1, 1, joints), device=q.device, dtype=q.dtype).masked_fill(~joint_mask[:, None, None], -torch.inf)
        out = attend(self, q, k, v, mask)
        if joint_mask is not None and encoder_hidden_states is None:
            out = out * joint_mask[..., None]
        return self.to_out(out)


class SlidingWindowAttnProcessor:
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, att_frames=None, att_slidwindow=None, joint_mask=None):
        batch_frames, joints, dim = hidden_states.shape
        heads, head_dim = attn.heads, dim // attn.heads
        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        q = attn.to_q(hidden_states).view(batch_frames, joints, heads, head_dim)
        k = attn.to_k(context).view(batch_frames, -1, heads, head_dim)
        v = attn.to_v(context).view(batch_frames, -1, heads, head_dim)
        if att_frames is not None:
            frames, batch = att_frames, batch_frames // att_frames
            q = q.view(batch, frames, joints, heads, head_dim).permute(0, 2, 3, 1, 4).flatten(0, 1)
            keys = k.shape[1]
            k = k.view(batch, frames, keys, heads, head_dim).permute(0, 2, 3, 1, 4).flatten(0, 1)
            q, k = frame_rope(q, k, frames)
            q = q.view(batch, joints, heads, frames, head_dim).permute(0, 3, 1, 2, 4)
            k = k.view(batch, keys, heads, frames, head_dim).permute(0, 3, 1, 2, 4)
            v = v.view(batch, frames, keys, heads, head_dim)
            if att_slidwindow is not None:
                pad = att_slidwindow // 2
                # Keep the upstream zero-valued boundary patches in cross-attention.
                k = F.pad(k, (0, 0, 0, 0, 0, 0, pad, pad)).unfold(1, att_slidwindow, 1).permute(0, 1, 5, 2, 3, 4)
                v = F.pad(v, (0, 0, 0, 0, 0, 0, pad, pad)).unfold(1, att_slidwindow, 1).permute(0, 1, 5, 2, 3, 4)
        q = q.reshape(batch_frames, joints, heads, head_dim).transpose(1, 2)
        k = k.reshape(batch_frames, -1, heads, head_dim).transpose(1, 2)
        v = v.reshape(batch_frames, -1, heads, head_dim).transpose(1, 2)
        mask = None
        if joint_mask is not None and encoder_hidden_states is None:
            valid = joint_mask
            if att_slidwindow is not None:
                pad = att_slidwindow // 2
                valid = F.pad(valid, (0, 0, pad, pad)).unfold(1, att_slidwindow, 1).transpose(-1, -2)
            valid = valid.reshape(batch_frames, 1, 1, -1)
            mask = torch.zeros(valid.shape, device=q.device, dtype=q.dtype).masked_fill(~valid, -torch.inf)
        return attn.to_out(attend(attn, q, k, v, mask))


class TemporalPerJointMultiHeadAttention(nn.Module):
    def __init__(self, d_model, nheads=8, temporal_window=2, use_temporal_bias=True):
        super().__init__()
        self.nheads = nheads
        self.temporal_window = temporal_window
        self.comfy_attention = ComfyAttention()
        self.linear_q = ops.Linear(d_model, d_model)
        self.linear_k = ops.Linear(d_model, d_model)
        self.linear_v = ops.Linear(d_model, d_model)
        self.output_layer = ops.Linear(d_model, d_model)
        self.temporal_bias = ops.Embedding(2 * temporal_window + 1, nheads) if use_temporal_bias else None

    def forward(self, q, k, v, mask=None):
        batch, frames, joints, dim = q.shape
        q = self.linear_q(q.permute(0, 2, 1, 3).reshape(batch * joints, frames, dim)).view(batch * joints, frames, self.nheads, -1).transpose(1, 2)
        k = self.linear_k(k.permute(0, 2, 1, 3).reshape(batch * joints, frames, dim)).view(batch * joints, frames, self.nheads, -1).transpose(1, 2)
        v = self.linear_v(v.permute(0, 2, 1, 3).reshape(batch * joints, frames, dim)).view(batch * joints, frames, self.nheads, -1).transpose(1, 2)
        q, k = frame_rope(q, k, frames)
        positions = torch.arange(frames, device=q.device)
        delta = positions[:, None] - positions[None, :]
        bias = torch.zeros((1, self.nheads, frames, frames), device=q.device, dtype=q.dtype)
        if self.temporal_bias is not None:
            bias = self.temporal_bias(delta.clamp(-self.temporal_window, self.temporal_window) + self.temporal_window,
                                      out_dtype=q.dtype).permute(2, 0, 1)[None] * (q.shape[-1] ** -0.5)
        bias = bias.masked_fill(delta.abs()[None, None] > self.temporal_window, -torch.inf)
        # The capture batch's joint validity is constant across time.
        out = self.output_layer(attend(self, q, k, v, bias))
        out = out.view(batch, joints, frames, dim).permute(0, 2, 1, 3)
        if mask is not None:
            out = out * (mask[:, None, :, None] if mask.ndim == 2 else mask[..., None])
        return out


class TemporalPerJointTransformerBlock(nn.Module):
    def __init__(self, dim, nheads=8, ff_mult=4, temporal_window=2, use_temporal_bias=True):
        super().__init__()
        self.norm1 = ops.LayerNorm(dim)
        self.attn = TemporalPerJointMultiHeadAttention(dim, nheads, temporal_window, use_temporal_bias)
        self.norm2 = ops.LayerNorm(dim)
        self.ffn = nn.Sequential(ops.Linear(dim, dim * ff_mult), nn.GELU(), nn.Identity(), ops.Linear(dim * ff_mult, dim), nn.Identity())

    def forward(self, x, joint_mask=None):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, mask=joint_mask)
        mask = joint_mask[:, None, :, None] if joint_mask.ndim == 2 else joint_mask[..., None]
        x = x * mask
        return (x + self.ffn(self.norm2(x))) * mask


class MemoryAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.comfy_attention = ComfyAttention()
        self.in_proj_weight = nn.Parameter(torch.empty(3 * dim, dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * dim))
        self.out_proj = ops.Linear(dim, dim)

    def forward(self, q, memory):
        weight = cast_to_input(self.in_proj_weight, q).chunk(3)
        bias = cast_to_input(self.in_proj_bias, q).chunk(3)
        batch, frames, dim = q.shape
        q = F.linear(q, weight[0], bias[0]).view(batch, frames, self.heads, -1).transpose(1, 2)
        k = F.linear(memory, weight[1], bias[1]).view(batch, -1, self.heads, dim // self.heads).transpose(1, 2)
        v = F.linear(memory, weight[2], bias[2]).view(batch, -1, self.heads, dim // self.heads).transpose(1, 2)
        return self.out_proj(attend(self, q, k, v))


class JointMemoryCrossAttention(nn.Module):
    def __init__(self, d_model, nheads=8):
        super().__init__()
        self.norm_q = ops.LayerNorm(d_model)
        self.norm_kv = ops.LayerNorm(d_model)
        self.attn = MemoryAttention(d_model, nheads)
        self.out_proj = ops.Linear(d_model, d_model)

    def forward(self, query_feat, memory_feat, joint_mask):
        batch, frames, joints, dim = query_feat.shape
        q = self.norm_q(query_feat.permute(0, 2, 1, 3).reshape(batch * joints, frames, dim))
        memory = self.norm_kv(memory_feat.permute(0, 2, 1, 3).reshape(batch * joints, -1, dim))
        out = self.out_proj(self.attn(q, memory)).view(batch, joints, frames, dim).permute(0, 2, 1, 3)
        return out * joint_mask[:, None, :, None]
