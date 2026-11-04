from torch import nn

from .video2pose import RefGuidedVideo2PoseModel
from .pose2rot import Pose2RotMemoryRestModel


class MoCapAnything(nn.Module):
    def __init__(self):
        super().__init__()
        self.video2pos = RefGuidedVideo2PoseModel(
            num_layers=8, ref_layers=4, use_joint_embed=True,
            use_graph_ref_inner=True, use_graph_temporal_inner=True,
        )
        self.pos2rot = Pose2RotMemoryRestModel(
            rest_layers=4, pose_layers=4, memory_layers=4, decoder_layers=8,
            decoder_use_cross_layers=6,
        )

    def forward(self, batch, transformer_options=None):
        options = {} if transformer_options is None else transformer_options
        position = self.video2pos(batch, attention_kwargs={
            "seq_len": batch["image_embed"].shape[1],
            "selfatt_temporal_layer_flag": [], "selfatt_temporal": True,
            "crossatt2_temporal": True, "selfatt_slidwindow": 5, "crossatt2_slidwindow": 5,
        }, transformer_options=options)
        rotation = self.pos2rot(batch, pose_override=position, transformer_options=options)
        return position, rotation["pred_rot6d"]
