_base_ = [
    '../_base_/models/pointrend_r50.py', '../_base_/datasets/telhados_unidos.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
lr_config = dict(warmup='linear', warmup_iters=200)
# model["decode_head"][0]['num_classes'] = 2
# model["decode_head"][1]['num_classes'] = 2
# print('MODELLLLLLLLLLLLLLLL: ', model)