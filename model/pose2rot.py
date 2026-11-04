# Adapted from MoCapAnything V2; see LICENSE and THIRD_PARTY.md.
import torch
from torch import nn
from comfy.ops import manual_cast as ops
import comfy.model_prefetch

from typing import Optional
from .attention import GraphMultiHeadAttention, TemporalPerJointTransformerBlock, JointMemoryCrossAttention

class FiLMCondition(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.to_scale_shift = nn.Sequential(ops.LayerNorm(dim), ops.Linear(dim, dim * 2))

    def forward(self, x, cond):
        scale, shift = self.to_scale_shift(cond).chunk(2, dim=-1)
        return x * (1.0 + 0.1 * scale) + shift

class RotDecoderBlock(nn.Module):

    def __init__(self, q_dim=256, num_heads=8, temporal_window=2, use_tree_mask=False, use_rest_film=True, use_cross_attn=True):
        super().__init__()
        self.use_rest_film = use_rest_film
        self.use_cross_attn = use_cross_attn
        self.film = FiLMCondition(q_dim) if use_rest_film else None
        self.temporal = TemporalPerJointTransformerBlock(dim=q_dim, nheads=num_heads, ff_mult=4, temporal_window=temporal_window, use_temporal_bias=True)
        self.graph_norm = ops.LayerNorm(q_dim)
        self.graph = GraphMultiHeadAttention(q_dim, num_heads, use_tree_mask=use_tree_mask)
        if self.use_cross_attn:
            self.cross = JointMemoryCrossAttention(d_model=q_dim, nheads=num_heads)
        self.ffn_norm = ops.LayerNorm(q_dim)
        self.ffn = nn.Sequential(ops.Linear(q_dim, q_dim * 4), nn.GELU(), nn.Identity(), ops.Linear(q_dim * 4, q_dim), nn.Identity())

    def forward(self, x, rest_t, memory_feat, joint_mask, ancestor_mask, graph_hop, graph_edge):
        B, T, J, D = x.shape
        joint_mask_bt = joint_mask.unsqueeze(1).expand(-1, T, -1)
        if self.film is not None:
            x = self.film(x, rest_t)
            x = x * joint_mask_bt.unsqueeze(-1)
        x = self.temporal(x, joint_mask=joint_mask_bt)
        x2 = self.graph_norm(x).reshape(B * T, J, D)
        jm = joint_mask_bt.reshape(B * T, J)
        gm = ancestor_mask.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, J, J)
        gh = graph_hop.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, J, J)
        ge = graph_edge.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, J, J)
        x2 = self.graph(x2, x2, x2, gh, ge, mask=jm, tree_mask=gm)
        x = x + x2.reshape(B, T, J, D)
        x = x * joint_mask_bt.unsqueeze(-1)
        if self.use_cross_attn:
            x = x + self.cross(x, memory_feat, joint_mask)
            x = x * joint_mask_bt.unsqueeze(-1)
        h = self.ffn_norm(x)
        h = self.ffn(h)
        x = x + h
        x = x * joint_mask_bt.unsqueeze(-1)
        return x

class RestPoseEncoder(nn.Module):

    def __init__(self, q_dim=256, joint_embed_dim=768, num_heads=8, num_layers=2):
        super().__init__()
        self.offset_proj = ops.Linear(3, q_dim)
        self.joint_t5proj = ops.Linear(joint_embed_dim, q_dim)
        self.layers = nn.ModuleList([nn.ModuleDict({'norm': ops.LayerNorm(q_dim), 'graph': GraphMultiHeadAttention(q_dim, num_heads, use_tree_mask=layer_idx % 2 == 0), 'ffn_norm': ops.LayerNorm(q_dim), 'ffn': nn.Sequential(ops.Linear(q_dim, q_dim * 4), nn.GELU(), nn.Identity(), ops.Linear(q_dim * 4, q_dim), nn.Identity())}) for layer_idx in range(num_layers)])
        self.out_norm = ops.LayerNorm(q_dim)

    def _run_block(self, blk, x, joint_mask, ancestor_mask, graph_hop, graph_edge):
        h = blk['norm'](x)
        h = blk['graph'](h, h, h, graph_hop, graph_edge, mask=joint_mask, tree_mask=ancestor_mask)
        x = x + h
        x = x * joint_mask.unsqueeze(-1)
        h = blk['ffn_norm'](x)
        h = blk['ffn'](h)
        x = x + h
        x = x * joint_mask.unsqueeze(-1)
        return x

    def forward(self, offset, joint_t5embed, joint_mask, ancestor_mask, graph_hop, graph_edge, transformer_options=None):
        transformer_options = {} if transformer_options is None else transformer_options
        x = self.offset_proj(offset) + self.joint_t5proj(joint_t5embed)
        x = x * joint_mask.unsqueeze(-1)
        prefetch = comfy.model_prefetch.make_prefetch_queue(list(self.layers), x.device, transformer_options)
        for blk in self.layers:
            comfy.model_prefetch.prefetch_queue_pop(prefetch, x.device, blk, x.dtype, malloc_scope="block")
            x = self._run_block(blk, x, joint_mask, ancestor_mask, graph_hop, graph_edge)
        comfy.model_prefetch.prefetch_queue_pop(prefetch, x.device, None)
        x = self.out_norm(x)
        x = x * joint_mask.unsqueeze(-1)
        return x

class PoseEncoderLite(nn.Module):

    def __init__(self, q_dim=256, num_layers=2, num_heads=8, temporal_window=2, joint_embed_dim=768, use_rest_film=True, use_graph=True):
        super().__init__()
        self.in_proj = ops.Linear(3, q_dim)
        self.joint_t5proj = ops.Linear(joint_embed_dim, q_dim)
        self.use_rest_film = use_rest_film
        self.use_graph = use_graph
        if use_rest_film:
            self.rest_film_blocks = nn.ModuleList([FiLMCondition(q_dim) for _ in range(num_layers)])
        else:
            self.rest_film_blocks = None
        self.temporal_blocks = nn.ModuleList([TemporalPerJointTransformerBlock(dim=q_dim, nheads=num_heads, ff_mult=4, temporal_window=temporal_window, use_temporal_bias=True) for _ in range(num_layers)])
        if use_graph:
            self.graph_blocks = nn.ModuleList([GraphMultiHeadAttention(q_dim, num_heads, use_tree_mask=layer_idx % 2 == 0) for layer_idx in range(num_layers)])
        else:
            self.graph_blocks = None
        self.out_norm = ops.LayerNorm(q_dim)

    def _run_block(self, temporal_blk, graph_blk, film_blk, x, rest_t, joint_mask_bt, ancestor_mask_bt, graph_hop_bt, graph_edge_bt):
        B, T, J, D = x.shape
        if film_blk is not None:
            x = film_blk(x, rest_t)
            x = x * joint_mask_bt.unsqueeze(-1)
        x = temporal_blk(x, joint_mask=joint_mask_bt)
        if graph_blk is not None:
            x2 = x.reshape(B * T, J, D)
            jm = joint_mask_bt.reshape(B * T, J)
            gm = ancestor_mask_bt.reshape(B * T, J, J)
            gh = graph_hop_bt.reshape(B * T, J, J)
            ge = graph_edge_bt.reshape(B * T, J, J)
            x2 = graph_blk(x2, x2, x2, gh, ge, mask=jm, tree_mask=gm)
            x = x + x2.reshape(B, T, J, D)
            x = x * joint_mask_bt.unsqueeze(-1)
        return x

    def forward(self, pose, rest_embed, joint_t5embed, joint_mask, ancestor_mask, graph_hop, graph_edge, transformer_options=None):
        transformer_options = {} if transformer_options is None else transformer_options
        B, T, J, _ = pose.shape
        pose_feat = self.in_proj(pose)
        joint_sem = self.joint_t5proj(joint_t5embed).unsqueeze(1).expand(-1, T, -1, -1)
        rest_t = rest_embed.unsqueeze(1).expand(-1, T, -1, -1)
        x = pose_feat + joint_sem
        joint_mask_bt = joint_mask.unsqueeze(1).expand(-1, T, -1)
        ancestor_mask_bt = ancestor_mask.unsqueeze(1).expand(-1, T, -1, -1)
        graph_hop_bt = graph_hop.unsqueeze(1).expand(-1, T, -1, -1)
        graph_edge_bt = graph_edge.unsqueeze(1).expand(-1, T, -1, -1)
        x = x * joint_mask_bt.unsqueeze(-1)
        if self.use_rest_film:
            film_blocks = self.rest_film_blocks
        else:
            film_blocks = [None] * len(self.temporal_blocks)
        if self.use_graph:
            graph_blocks = self.graph_blocks
        else:
            graph_blocks = [None] * len(self.temporal_blocks)
        blocks = list(zip(self.temporal_blocks, graph_blocks, film_blocks))
        prefetch = comfy.model_prefetch.make_prefetch_queue([tuple(b for b in group if b is not None) for group in blocks], x.device, transformer_options)
        for temporal_blk, graph_blk, film_blk in blocks:
            group = tuple(b for b in (temporal_blk, graph_blk, film_blk) if b is not None)
            comfy.model_prefetch.prefetch_queue_pop(prefetch, x.device, group, x.dtype, malloc_scope="block")
            x = self._run_block(temporal_blk, graph_blk, film_blk, x, rest_t, joint_mask_bt, ancestor_mask_bt, graph_hop_bt, graph_edge_bt)
        comfy.model_prefetch.prefetch_queue_pop(prefetch, x.device, None)
        x = self.out_norm(x)
        x = x * joint_mask_bt.unsqueeze(-1)
        return x

class MemoryEncoder(nn.Module):

    def __init__(self, q_dim=256, joint_embed_dim=768, num_heads=8, num_layers=2, use_rest_film=True):
        super().__init__()
        self.pose_proj = ops.Linear(3, q_dim)
        self.rot_proj = ops.Linear(6, q_dim)
        self.joint_t5proj = ops.Linear(joint_embed_dim, q_dim)
        self.use_rest_film = use_rest_film
        self.layers = nn.ModuleList([nn.ModuleDict({'norm1': ops.LayerNorm(q_dim), 'graph': GraphMultiHeadAttention(q_dim, num_heads, use_tree_mask=layer_idx % 2 == 0), 'norm2': ops.LayerNorm(q_dim), 'ffn': nn.Sequential(ops.Linear(q_dim, q_dim * 4), nn.GELU(), nn.Identity(), ops.Linear(q_dim * 4, q_dim), nn.Identity()), 'film': FiLMCondition(q_dim) if use_rest_film else nn.Identity()}) for layer_idx in range(num_layers)])
        self.out_norm = ops.LayerNorm(q_dim)

    def _run_block(self, blk, x, rest_n, joint_mask_bn, ancestor_mask_bn, graph_hop_bn, graph_edge_bn):
        if self.use_rest_film:
            x = blk['film'](x, rest_n)
            x = x * joint_mask_bn.unsqueeze(-1)
        h = blk['norm1'](x)
        h = blk['graph'](h, h, h, graph_hop_bn, graph_edge_bn, mask=joint_mask_bn, tree_mask=ancestor_mask_bn)
        x = x + h
        x = x * joint_mask_bn.unsqueeze(-1)
        h = blk['norm2'](x)
        h = blk['ffn'](h)
        x = x + h
        x = x * joint_mask_bn.unsqueeze(-1)
        return x

    def forward(self, memory_pose, memory_rot6d, rest_embed, joint_t5embed, joint_mask, ancestor_mask, graph_hop, graph_edge, transformer_options=None):
        transformer_options = {} if transformer_options is None else transformer_options
        B, N, J, _ = memory_pose.shape
        pose_feat = self.pose_proj(memory_pose)
        rot_feat = self.rot_proj(memory_rot6d)
        joint_sem = self.joint_t5proj(joint_t5embed).unsqueeze(1).expand(-1, N, -1, -1)
        rest_n = rest_embed.unsqueeze(1).expand(-1, N, -1, -1)
        x = pose_feat + rot_feat + joint_sem
        x = x.reshape(B * N, J, -1)
        rest_n = rest_n.reshape(B * N, J, -1)
        joint_mask_bn = joint_mask.unsqueeze(1).expand(-1, N, -1).reshape(B * N, J)
        ancestor_mask_bn = ancestor_mask.unsqueeze(1).expand(-1, N, -1, -1).reshape(B * N, J, J)
        graph_hop_bn = graph_hop.unsqueeze(1).expand(-1, N, -1, -1).reshape(B * N, J, J)
        graph_edge_bn = graph_edge.unsqueeze(1).expand(-1, N, -1, -1).reshape(B * N, J, J)
        x = x * joint_mask_bn.unsqueeze(-1)
        prefetch = comfy.model_prefetch.make_prefetch_queue(list(self.layers), x.device, transformer_options)
        for blk in self.layers:
            comfy.model_prefetch.prefetch_queue_pop(prefetch, x.device, blk, x.dtype, malloc_scope="block")
            x = self._run_block(blk, x, rest_n, joint_mask_bn, ancestor_mask_bn, graph_hop_bn, graph_edge_bn)
        comfy.model_prefetch.prefetch_queue_pop(prefetch, x.device, None)
        x = self.out_norm(x)
        x = x * joint_mask_bn.unsqueeze(-1)
        x = x.reshape(B, N, J, -1)
        return x

class RotDecoder(nn.Module):

    def __init__(self, q_dim=256, num_layers=4, num_heads=8, temporal_window=2, joint_embed_dim=768, cond_mode='add', use_rest_film=True, use_cross_layers=None):
        super().__init__()
        assert cond_mode in ['add', 'concat']
        self.cond_mode = cond_mode
        self.use_rest_film = use_rest_film
        self.joint_t5proj = ops.Linear(joint_embed_dim, q_dim)
        if cond_mode == 'concat':
            self.init_proj = ops.Linear(q_dim * 2, q_dim)
        else:
            self.init_proj = None
        if use_cross_layers is None:
            use_cross_layers = num_layers
        use_cross_layers = max(0, min(use_cross_layers, num_layers))
        self.use_cross_layers = use_cross_layers
        self.blocks = nn.ModuleList([RotDecoderBlock(q_dim=q_dim, num_heads=num_heads, temporal_window=temporal_window, use_tree_mask=layer_idx % 2 == 0, use_rest_film=use_rest_film, use_cross_attn=layer_idx < use_cross_layers) for layer_idx in range(num_layers)])
        self.head = nn.Sequential(ops.LayerNorm(q_dim), ops.Linear(q_dim, q_dim), nn.GELU(), ops.Linear(q_dim, 6))

    def _run_block(self, blk, x, rest_t, memory_feat, joint_mask, ancestor_mask, graph_hop, graph_edge):
        return blk(x=x, rest_t=rest_t, memory_feat=memory_feat, joint_mask=joint_mask, ancestor_mask=ancestor_mask, graph_hop=graph_hop, graph_edge=graph_edge)

    def forward(self, query_feat, memory_feat, rest_embed, joint_t5embed, joint_mask, ancestor_mask, graph_hop, graph_edge, transformer_options=None):
        transformer_options = {} if transformer_options is None else transformer_options
        B, T, J, D = query_feat.shape
        rest_t = rest_embed.unsqueeze(1).expand(-1, T, -1, -1)
        joint_sem = self.joint_t5proj(joint_t5embed).unsqueeze(1).expand(-1, T, -1, -1)
        if self.cond_mode == 'add':
            x = query_feat + joint_sem
        elif self.cond_mode == 'concat':
            x = self.init_proj(torch.cat([query_feat, joint_sem], dim=-1))
        else:
            raise ValueError(f'Unknown decoder cond mode: {self.cond_mode}')
        x = x * joint_mask.unsqueeze(1).unsqueeze(-1)
        prefetch = comfy.model_prefetch.make_prefetch_queue(list(self.blocks), x.device, transformer_options)
        for blk in self.blocks:
            comfy.model_prefetch.prefetch_queue_pop(prefetch, x.device, blk, x.dtype, malloc_scope="block")
            x = self._run_block(blk, x, rest_t, memory_feat, joint_mask, ancestor_mask, graph_hop, graph_edge)
        comfy.model_prefetch.prefetch_queue_pop(prefetch, x.device, None)
        out = self.head(x.reshape(B * T, J, D)).reshape(B, T, J, 6)
        out = out * joint_mask.unsqueeze(1).unsqueeze(-1)
        return out

class Pose2RotMemoryRestModel(nn.Module):

    def __init__(self, q_dim=256, rest_layers=2, pose_layers=2, memory_layers=2, decoder_layers=4, num_heads=8, joint_embed_dim=768, temporal_window=2, decoder_cond_mode='add', pose_rest_film=True, memory_rest_film=True, decoder_rest_film=True, pose_use_graph=True, decoder_use_cross_layers=None):
        super().__init__()
        self.rest_encoder = RestPoseEncoder(q_dim=q_dim, joint_embed_dim=joint_embed_dim, num_heads=num_heads, num_layers=rest_layers)
        self.pose_encoder = PoseEncoderLite(q_dim=q_dim, num_layers=pose_layers, num_heads=num_heads, temporal_window=temporal_window, joint_embed_dim=joint_embed_dim, use_rest_film=pose_rest_film, use_graph=pose_use_graph)
        self.memory_encoder = MemoryEncoder(q_dim=q_dim, joint_embed_dim=joint_embed_dim, num_heads=num_heads, num_layers=memory_layers, use_rest_film=memory_rest_film)
        self.decoder = RotDecoder(q_dim=q_dim, num_layers=decoder_layers, num_heads=num_heads, temporal_window=temporal_window, joint_embed_dim=joint_embed_dim, cond_mode=decoder_cond_mode, use_rest_film=decoder_rest_film, use_cross_layers=decoder_use_cross_layers)

    def forward(self, batch, pose_override: Optional[torch.Tensor]=None, transformer_options=None):
        transformer_options = {} if transformer_options is None else transformer_options
        joint_mask = batch['joint_mask'].bool()
        ancestor_mask = batch['ancestor_mask'].bool()
        graph_hop = batch['graph_hop']
        graph_edge = batch['graph_edge']
        static_rot_joint_mask = batch['static_rot_joint_mask'].bool()
        pose = pose_override if pose_override is not None else batch['position']
        memory_pose = batch['memory_pose']
        memory_rot6d = batch['memory_rot6d']
        ref_rot6d = batch['ref_rot6d_a']
        offset = batch['offset_a']
        joint_t5embed = batch['joint_t5embed']
        rest_embed = self.rest_encoder(offset=offset, joint_t5embed=joint_t5embed, joint_mask=joint_mask, ancestor_mask=ancestor_mask, graph_hop=graph_hop, graph_edge=graph_edge, transformer_options=transformer_options)
        q_feat = self.pose_encoder(pose=pose, rest_embed=rest_embed, joint_t5embed=joint_t5embed, joint_mask=joint_mask, ancestor_mask=ancestor_mask, graph_hop=graph_hop, graph_edge=graph_edge, transformer_options=transformer_options)
        mem_feat = self.memory_encoder(memory_pose=memory_pose, memory_rot6d=memory_rot6d, rest_embed=rest_embed, joint_t5embed=joint_t5embed, joint_mask=joint_mask, ancestor_mask=ancestor_mask, graph_hop=graph_hop, graph_edge=graph_edge, transformer_options=transformer_options)
        pred_rot6d = self.decoder(query_feat=q_feat, memory_feat=mem_feat, rest_embed=rest_embed, joint_t5embed=joint_t5embed, joint_mask=joint_mask, ancestor_mask=ancestor_mask, graph_hop=graph_hop, graph_edge=graph_edge, transformer_options=transformer_options)
        static_rot_joint_mask_4d = static_rot_joint_mask.unsqueeze(1).unsqueeze(-1)
        ref_rot6d_4d = ref_rot6d.unsqueeze(1).expand_as(pred_rot6d)
        pred_rot6d = pred_rot6d * ~static_rot_joint_mask_4d + ref_rot6d_4d * static_rot_joint_mask_4d
        return {'pred_rot6d': pred_rot6d, 'rest_embed': rest_embed, 'q_feat': q_feat, 'mem_feat': mem_feat}
