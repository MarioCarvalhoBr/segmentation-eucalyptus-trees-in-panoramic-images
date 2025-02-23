_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', '../_base_/datasets/arvore_vant.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
model = dict(test_cfg=dict(crop_size=(256, 256), stride=(170, 170)))
evaluation = dict(metric='mIoU')
data_root = '/home/wnunes/Documents/Mario/datasets/mauricio_arvores/exp_2/'


model = dict(decode_head=dict(num_classes=2), auxiliary_head=dict(num_classes=2))
