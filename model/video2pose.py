# Adapted from MoCapAnything V2; see LICENSE and THIRD_PARTY.md.
import torch
from torch import nn
from comfy.ops import manual_cast as ops
import comfy.model_management
import comfy.model_prefetch

from .attention import GraphMultiHeadAttention, SimpleAttention, SlidingWindowAttnProcessor, ComfyAttention

class FrequencyPositionalEmbedding(nn.Module):

    def __init__(self, num_freqs: int=6, logspace: bool=True, input_dim: int=3, include_input: bool=True, include_pi: bool=True) -> None:
        super().__init__()
        if logspace:
            frequencies = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        else:
            frequencies = torch.linspace(1.0, 2.0 ** (num_freqs - 1), num_freqs, dtype=torch.float32)
        if include_pi:
            frequencies *= torch.pi
        self.register_buffer('frequencies', frequencies, persistent=False)
        self.include_input = include_input
        self.num_freqs = num_freqs
        self.out_dim = self.get_dims(input_dim)

    def get_dims(self, input_dim):
        temp = 1 if self.include_input or self.num_freqs == 0 else 0
        out_dim = input_dim * (self.num_freqs * 2 + temp)
        return out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_freqs > 0:
            embed = (x[..., None].contiguous() * comfy.model_management.cast_to(self.frequencies, dtype=x.dtype, device=x.device)).view(*x.shape[:-1], -1)
            if self.include_input:
                return torch.cat((x, embed.sin(), embed.cos()), dim=-1)
            else:
                return torch.cat((embed.sin(), embed.cos()), dim=-1)
        else:
            return x

class SlidingWindowAttention(nn.Module):

    def __init__(self, dim, heads, cross_attention_dim=None, processor=None):
        super().__init__()
        self.heads = heads
        self.comfy_attention = ComfyAttention()
        self.to_q = ops.Linear(dim, dim, bias=False)
        self.to_k = ops.Linear(cross_attention_dim or dim, dim, bias=False)
        self.to_v = ops.Linear(cross_attention_dim or dim, dim, bias=False)
        self.to_out = ops.Linear(dim, dim, bias=False)
        self.processor = processor or SlidingWindowAttnProcessor()

    def forward(self, x, encoder_hidden_states=None, **kwargs):
        return self.processor(self, x, encoder_hidden_states=encoder_hidden_states, **kwargs)

class SlidingWindowDiTBlock(nn.Module):

    def __init__(self, dim, num_attention_heads, cross_attention_dim=None, norm_eps=1e-05, use_cross_attention=False, layer_idx=0, processor=None):
        super().__init__()
        self.norm1 = ops.LayerNorm(dim, eps=norm_eps)
        self.norm2 = ops.LayerNorm(dim, eps=norm_eps)
        self.self_attn = SlidingWindowAttention(dim, num_attention_heads, processor=processor)
        self.cross_attn = SlidingWindowAttention(dim, num_attention_heads, cross_attention_dim, processor=processor) if use_cross_attention else None
        self.layer_idx = layer_idx

    def forward(self, x, encoder_hidden_states=None, attention_kwargs=None, joint_mask=None):
        block_attention_kwargs = attention_kwargs.copy() if attention_kwargs is not None else {}
        cross_attention_scale = block_attention_kwargs.pop('cross_attention_scale', 1.0)
        seq_len = block_attention_kwargs.pop('seq_len', None)
        selfatt_temporal_layer_flag = block_attention_kwargs.pop('selfatt_temporal_layer_flag', None)
        selfatt_frame = seq_len if block_attention_kwargs.pop('selfatt_temporal', None) else None
        crossatt2_frame = seq_len if block_attention_kwargs.pop('crossatt2_temporal', None) else None
        selfatt_slidwindow = block_attention_kwargs.pop('selfatt_slidwindow', None)
        crossatt2_slidwindow = block_attention_kwargs.pop('crossatt2_slidwindow', None)
        if selfatt_temporal_layer_flag is not None:
            if self.layer_idx in selfatt_temporal_layer_flag:
                pass
            else:
                selfatt_frame = None
                selfatt_slidwindow = None
        if joint_mask is not None:
            B, seq_len, J = joint_mask.shape
            x = x * joint_mask.reshape(B * seq_len, J).unsqueeze(-1)
        x1 = self.norm1(x)
        x_sa = self.self_attn(x1, att_frames=selfatt_frame, att_slidwindow=selfatt_slidwindow, joint_mask=joint_mask)
        x = x + x_sa
        if joint_mask is not None:
            B, seq_len, J = joint_mask.shape
            x = x * joint_mask.reshape(B * seq_len, J).unsqueeze(-1)
        if self.cross_attn is not None and encoder_hidden_states is not None:
            x2 = self.norm2(x)
            x_ca = self.cross_attn(x2, encoder_hidden_states=encoder_hidden_states, att_frames=crossatt2_frame, att_slidwindow=crossatt2_slidwindow)
            x = x + x_ca * cross_attention_scale
        if joint_mask is not None:
            B, seq_len, J = joint_mask.shape
            x = x * joint_mask.reshape(B * seq_len, J).unsqueeze(-1)
        return x

class Video2PoseModelSliding(nn.Module):

    def __init__(self, num_layers=4, q_dim=256, img_dim=1024, num_heads=8, use_graph_temporal_outer=False, use_graph_temporal_inner=False):
        super().__init__()
        self.use_graph_temporal_outer = use_graph_temporal_outer
        self.use_graph_temporal_inner = use_graph_temporal_inner
        self.img_proj = ops.Linear(img_dim, q_dim)
        if self.use_graph_temporal_outer:
            self.graph_attn = GraphMultiHeadAttention(q_dim, num_heads, use_tree_mask=True)
        if self.use_graph_temporal_inner:
            self.blocks_graph = nn.ModuleList([GraphMultiHeadAttention(q_dim, num_heads, use_tree_mask=i % 2 == 0) for i in range(num_layers)])
        self.blocks_img = nn.ModuleList([SlidingWindowDiTBlock(q_dim, num_heads, cross_attention_dim=q_dim, use_cross_attention=True, layer_idx=i, processor=SlidingWindowAttnProcessor()) for i in range(num_layers)])
        self.head = nn.Sequential(ops.LayerNorm(q_dim), ops.Linear(q_dim, 128), nn.ReLU(), ops.Linear(128, 3))

    def forward(self, ref_query, cond_img, joint_mask=None, attention_kwargs=None, graph_hop=None, graph_edge=None, tree_mask=None, transformer_options=None):
        transformer_options = {} if transformer_options is None else transformer_options
        B, T, _, _ = cond_img.shape
        _, J, D = ref_query.shape
        queries = ref_query.unsqueeze(1).expand(B, T, J, D).reshape(B * T, J, D)
        cond_img = self.img_proj(cond_img)
        if self.use_graph_temporal_outer and graph_hop is not None:
            queries = queries + self.graph_attn(queries, queries, queries, graph_hop, graph_edge, joint_mask, tree_mask=tree_mask)
        blocks = list(zip(self.blocks_img, self.blocks_graph)) if self.use_graph_temporal_inner else list(self.blocks_img)
        prefetch = comfy.model_prefetch.make_prefetch_queue(blocks, queries.device, transformer_options)
        for i, block_img in enumerate(self.blocks_img):
            comfy.model_prefetch.prefetch_queue_pop(prefetch, queries.device, blocks[i], queries.dtype, malloc_scope="block")
            queries = block_img(queries, encoder_hidden_states=cond_img, attention_kwargs=attention_kwargs, joint_mask=joint_mask)
            if self.use_graph_temporal_inner and graph_hop is not None:
                queries = queries + self.blocks_graph[i](queries, queries, queries, graph_hop, graph_edge, joint_mask, tree_mask=tree_mask)
                if joint_mask is not None:
                    queries = queries * joint_mask.reshape(B * T, J).unsqueeze(-1)
        comfy.model_prefetch.prefetch_queue_pop(prefetch, queries.device, None)
        out = self.head(queries)
        out = out.reshape(B, T, out.size(1), out.size(2))
        return out

class RefFusionBlock(nn.Module):

    def __init__(self, q_dim=256, num_heads=8, use_tree_mask=True):
        super().__init__()
        self.graph_attn = GraphMultiHeadAttention(q_dim, num_heads, use_tree_mask=use_tree_mask)
        self.self_attn = SimpleAttention(q_dim, num_heads)
        self.cross_img = SimpleAttention(q_dim, num_heads)
        self.norm = ops.LayerNorm(q_dim)
        self.res_scale = nn.Parameter(torch.empty(1))

    def forward(self, x, img_cond, joint_mask=None, attention_kwargs=None, graph_hop=None, graph_edge=None, tree_mask=None):
        if graph_hop is not None and graph_edge is not None:
            x = x + self.graph_attn(x, x, x, graph_hop, graph_edge, joint_mask, tree_mask=tree_mask)
        x = x + self.self_attn(x, joint_mask=joint_mask)
        x = x + self.cross_img(x, encoder_hidden_states=img_cond, joint_mask=joint_mask)
        if joint_mask is not None:
            x = x * joint_mask.unsqueeze(-1)
        return self.norm(x * comfy.model_management.cast_to(self.res_scale, dtype=x.dtype, device=x.device))

class RefQueryEncoder(nn.Module):

    def __init__(self, q_dim=256, img_dim=1024, num_heads=8, num_layers=3, joint_embed_dim=768, use_joint_embed=False, use_graph_ref_outer=False, use_graph_ref_inner=False):
        super().__init__()
        self.pos_embedder = FrequencyPositionalEmbedding(num_freqs=8, input_dim=3)
        self.pose_proj = ops.Linear(self.pos_embedder.out_dim, q_dim)
        self.img_proj = ops.Linear(img_dim, q_dim)
        self.use_joint_embed = use_joint_embed
        self.use_graph_ref_outer = use_graph_ref_outer
        self.use_graph_ref_inner = use_graph_ref_inner
        if self.use_joint_embed:
            self.joint_t5proj = ops.Linear(joint_embed_dim, q_dim)
        if self.use_graph_ref_outer:
            self.graph_attn = GraphMultiHeadAttention(q_dim, num_heads, use_tree_mask=True)
        self.fusion_blocks = nn.ModuleList([RefFusionBlock(q_dim=q_dim, num_heads=num_heads, use_tree_mask=i % 2 == 0) for i in range(num_layers)])
        self.final_norm = ops.LayerNorm(q_dim)

    def forward(self, ref_position, ref_image_embed, joint_mask=None, attention_kwargs=None, graph_hop=None, graph_edge=None, joint_t5embed=None, tree_mask=None, transformer_options=None):
        transformer_options = {} if transformer_options is None else transformer_options
        ref_position_enc = self.pos_embedder(ref_position)
        x = self.pose_proj(ref_position_enc)
        if self.use_joint_embed and joint_t5embed is not None:
            joint_feat = self.joint_t5proj(joint_t5embed)
            x = x + joint_feat
        if joint_mask is not None:
            x = x * joint_mask.unsqueeze(-1)
        if self.use_graph_ref_outer and graph_hop is not None:
            x = x + self.graph_attn(x, x, x, graph_hop, graph_edge, joint_mask, tree_mask=tree_mask)
        img_cond = self.img_proj(ref_image_embed)
        prefetch = comfy.model_prefetch.make_prefetch_queue(list(self.fusion_blocks), x.device, transformer_options)
        for blk in self.fusion_blocks:
            comfy.model_prefetch.prefetch_queue_pop(prefetch, x.device, blk, x.dtype, malloc_scope="block")
            x = blk(x, img_cond, joint_mask=joint_mask, attention_kwargs=attention_kwargs, graph_hop=graph_hop if self.use_graph_ref_inner else None, graph_edge=graph_edge if self.use_graph_ref_inner else None, tree_mask=tree_mask if self.use_graph_ref_inner else None)
        comfy.model_prefetch.prefetch_queue_pop(prefetch, x.device, None)
        if joint_mask is not None:
            x = x * joint_mask.unsqueeze(-1)
        return self.final_norm(x)

class RefGuidedVideo2PoseModel(nn.Module):

    def __init__(self, num_layers=4, q_dim=256, img_dim=1024, num_heads=8, ref_layers=3, use_joint_embed=False, use_graph_ref_outer=False, use_graph_ref_inner=False, use_graph_temporal_outer=False, use_graph_temporal_inner=False):
        super().__init__()
        self.ref_encoder = RefQueryEncoder(q_dim=q_dim, img_dim=img_dim, num_heads=num_heads, num_layers=ref_layers, use_joint_embed=use_joint_embed, use_graph_ref_outer=use_graph_ref_outer, use_graph_ref_inner=use_graph_ref_inner)
        self.temporal_model = Video2PoseModelSliding(num_layers=num_layers, q_dim=q_dim, img_dim=img_dim, num_heads=num_heads, use_graph_temporal_outer=use_graph_temporal_outer, use_graph_temporal_inner=use_graph_temporal_inner)

    def forward(self, batch, attention_kwargs=None, transformer_options=None):
        transformer_options = {} if transformer_options is None else transformer_options
        image_embed = batch['image_embed']
        ref_pos = batch['ref_position']
        ref_img = batch['ref_image_embed']
        joint_mask = batch['joint_mask']
        graph_hop = batch['graph_hop']
        graph_edge = batch['graph_edge']
        joint_t5embed = batch['joint_t5embed']
        ancestor_mask = batch['ancestor_mask'].bool()
        ref_query = self.ref_encoder(ref_position=ref_pos, ref_image_embed=ref_img, joint_mask=joint_mask, attention_kwargs=attention_kwargs, graph_hop=graph_hop, graph_edge=graph_edge, joint_t5embed=joint_t5embed, tree_mask=ancestor_mask, transformer_options=transformer_options)
        F = image_embed.shape[1]
        joint_mask = joint_mask.unsqueeze(1).expand(-1, F, -1)
        graph_hop = graph_hop.unsqueeze(1).expand(-1, F, -1, -1)
        graph_edge = graph_edge.unsqueeze(1).expand(-1, F, -1, -1)
        ancestor_mask_t = ancestor_mask.unsqueeze(1).expand(-1, F, -1, -1)
        pose_pred = self.temporal_model(ref_query=ref_query, cond_img=image_embed, joint_mask=joint_mask, attention_kwargs=attention_kwargs, graph_hop=graph_hop, graph_edge=graph_edge, tree_mask=ancestor_mask_t, transformer_options=transformer_options)
        static_pos_joint_mask = batch['static_pos_joint_mask'].bool()
        ref_pos = batch['ref_position']
        static_pos_joint_mask_4d = static_pos_joint_mask.unsqueeze(1).unsqueeze(-1)
        ref_pos_4d = ref_pos.unsqueeze(1).expand(-1, pose_pred.shape[1], -1, -1)
        pose_pred = pose_pred * ~static_pos_joint_mask_4d + ref_pos_4d * static_pos_joint_mask_4d
        return pose_pred
