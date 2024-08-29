_base_ = [
    '../_base_/models/ann_r50-d8.py', '../_base_/datasets/eucalipto.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
model = dict(
    decode_head=dict(num_classes=2), auxiliary_head=dict(num_classes=2))
