# Reference NPZ format

References live in `ComfyUI/input/mocapanything/` and load with `allow_pickle=False`. `J` is the actual joint count; padding is unnecessary. Parents must appear before children, with joint 0 as the only root. All arrays must follow the same joint order.

| Field | Shape/type | Meaning |
| --- | --- | --- |
| `names` | `[J]` Unicode | BVH joint names, without whitespace/braces |
| `parents` | `[J]` int64 | Root parent `-1`; other entries reference earlier joints |
| `offsets` | `[J,3]` float32 | BVH local rest offsets in original units |
| `global_scale` | scalar float | Positive upstream species scale |
| `ref_position` | `[J,3]` float32 | Root-relative reference pose, divided by scale |
| `ref_rot6d_a` | `[J,6]` float32 | First two rows of local rotation matrices, flattened |
| `ref_image_embed` | `[P,1024]` float32 | Matching DINOv2-Large last hidden state, including CLS; normally P=257 |
| `joint_t5embed` | `[J,768]` float32 | Upstream joint semantic features |
| `graph_hop` | `[J,J]` int64 | Upstream hop indices, range 0–5 |
| `graph_edge` | `[J,J]` int64 | Upstream relation indices, range 0–5 |
| `static_pos_joint_mask` | `[J]` bool | Preserve these reference joint positions |
| `static_rot_joint_mask` | `[J]` bool | Preserve these reference joint rotations |

Use the converter on upstream-prepared rigs. Do not substitute zero semantic embeddings or an arbitrary scale for missing reference data. The capture step builds the ancestor mask from parents and uses the reference pose/rotation as its single memory example, matching upstream's default when no memory bank is supplied.
