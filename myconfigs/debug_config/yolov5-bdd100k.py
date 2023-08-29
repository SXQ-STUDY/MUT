default_scope = 'mmyolo'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(
        type='YOLOv5ParamSchedulerHook',
        scheduler_type='linear',
        lr_factor=0.15135,
        max_epochs=50,
        warmup_epochs=3.3835,
        warmup_momentum=0.59462,
        warmup_bias_lr=0.18657),
    checkpoint=dict(
        type='CheckpointHook', interval=1, save_best='auto',
        max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
visualizer = dict(
    type='mmdet.DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ],
    name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_n-v61_syncbn_fast_8xb16-300e_coco/yolov5_n-v61_syncbn_fast_8xb16-300e_coco_20220919_090739-b804c1ad.pth'
resume = False

model = dict(
    type='mmdet.Mut_SingleStageDetector',
    data_preprocessor=dict(
        type='mmdet.MUT_DetDataPreprocessor',
        mean=[0.0,0.0,0.0,],
        std=[255.0,255.0,255.0,],
        pad_size_divisor=32,
        pad_seg=True,
        bgr_to_rgb=True),
    backbone=dict(
        type='YOLOv5CSPDarknet',
        deepen_factor=0.33,
        widen_factor=0.25,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True)),
    neck_det=dict(
        type='YOLOv5PAFPN',
        deepen_factor=0.33,
        widen_factor=0.25,
        in_channels=[256,512,1024,],
        out_channels=[256,512,1024,],
        num_csp_blocks=3,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True)),
    neck_seg=dict(
        type='YOLOv5PAFPN',
        deepen_factor=0.33,
        widen_factor=0.25,
        in_channels=[256,512,1024,],
        out_channels=[256,512,1024,],
        num_csp_blocks=3,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='YOLOv5Head',
        head_module=dict(
            type='YOLOv5HeadModule',
            num_classes=20,
            in_channels=[256,512,1024,],
            widen_factor=0.25,
            featmap_strides=[8,16,32,],
            num_base_priors=3),
        prior_generator=dict(
            type='mmdet.YOLOAnchorGenerator',
            base_sizes=[
                [(26, 44), (67, 57), (61, 130)], 
                [(121, 118), (120, 239), (206, 182)],
                [(376, 161), (234, 324), (428, 322)]
            ],
            strides=[8,16,32,]),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=0.054095,
            class_weight=0.5),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='ciou',
            bbox_format='xywh',
            eps=1e-07,
            reduction='mean',
            loss_weight=0.02,
            return_iou=True),
        loss_obj=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=0.33105920000000005,
            class_weight=0.67198),
        prior_match_thr=3.3744,
        obj_level_weights=[4.0,1.0,0.4,]),
    seg_head=dict(
        type='mmseg.FCNHead',
        in_channels=256,
        in_index=2,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    det_test_cfg=dict(
        multi_label=True,
        nms_pre=30000,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=300))

data_root = r'D:\sxq\datasets\bdd100k_mut'

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations', with_bbox=True, with_seg=True),
    dict(type='mmseg.GenerateEdge', edge_width=4),
    dict(
        type='YOLOv5HSVRandomAug',
        hue_delta=0.01041,
        saturation_delta=0.54703,
        value_delta=0.27739),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.MUT_PackDetInputs',
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'flip',
            'flip_direction',
        )),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        dict(
            type='mmdet.BDD100K_MUT_Dataset',
            data_root=data_root,
            ann_file='train_val_txt/train.txt',
            img_subdir='images',
            ann_subdir='annotations',
            LD_subdir='LD_label_imgs',
            filter_cfg=dict(filter_empty_gt=True, min_size=0),
            pipeline=train_pipeline)))
    # collate_fn=dict(type='yolov5_collate_mut'))

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='YOLOv5KeepRatioResize', scale=(
        512,
        512,
    )),
    dict(
        type='LetterResize',
        scale=(
            512,
            512,
        ),
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, with_seg=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'pad_param',
        )),
]
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='mmdet.BDD100K_MUT_Dataset',
        data_root=data_root,
        test_mode=True,
        ann_file='train_val_txt/val.txt',
        img_subdir='images',
        ann_subdir='annotations',
        LD_subdir='LD_label_imgs',
        pipeline=test_pipeline,
        batch_shapes_cfg=dict(
            type='BatchShapePolicy',
            batch_size=1,
            img_size=512,
            size_divisor=32,
            extra_pad_ratio=0.5)))
test_dataloader = val_dataloader
param_scheduler = None

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.00334,
        momentum=0.74832,
        weight_decay=0.00025,
        nesterov=True,
        batch_size_per_gpu=64),
    constructor='YOLOv5OptimizerConstructor')
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49),
]
val_evaluator = dict(type='mmdet.VOCMetric', metric='mAP', eval_mode='area')
test_evaluator = dict(type='mmdet.VOCMetric', metric='mAP', eval_mode='area')
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


# dict(
#         type='RandomChoice',
#         transforms=[
#             [
#                 dict(
#                     type='Mosaic',
#                     img_scale=(
#                         512,
#                         512,
#                     ),
#                     pad_val=114.0,
#                     pre_transform=[
#                         dict(
#                             type='LoadImageFromFile',
#                             file_client_args=dict(backend='disk')),
#                         dict(type='LoadAnnotations', with_bbox=True),
#                     ]),
#                 dict(
#                     type='YOLOv5RandomAffine',
#                     max_rotate_degree=0.0,
#                     max_translate_ratio=0.04591,
#                     max_shear_degree=0.0,
#                     scaling_ratio_range=(
#                         0.24456,
#                         1.7554400000000001,
#                     ),
#                     border=(
#                         -256,
#                         -256,
#                     ),
#                     border_val=(
#                         114,
#                         114,
#                         114,
#                     )),
#                 dict(
#                     type='YOLOv5MixUp',
#                     prob=0.04266,
#                     pre_transform=[
#                         dict(
#                             type='LoadImageFromFile',
#                             file_client_args=dict(backend='disk')),
#                         dict(type='LoadAnnotations', with_bbox=True),
#                         dict(
#                             type='Mosaic',
#                             img_scale=(
#                                 512,
#                                 512,
#                             ),
#                             pad_val=114.0,
#                             pre_transform=[
#                                 dict(
#                                     type='LoadImageFromFile',
#                                     file_client_args=dict(backend='disk')),
#                                 dict(type='LoadAnnotations', with_bbox=True),
#                             ]),
#                         dict(
#                             type='YOLOv5RandomAffine',
#                             max_rotate_degree=0.0,
#                             max_translate_ratio=0.04591,
#                             max_shear_degree=0.0,
#                             scaling_ratio_range=(
#                                 0.24456,
#                                 1.7554400000000001,
#                             ),
#                             border=(
#                                 -256,
#                                 -256,
#                             ),
#                             border_val=(
#                                 114,
#                                 114,
#                                 114,
#                             )),
#                     ]),
#             ],
#             [
#                 dict(
#                     type='YOLOv5RandomAffine',
#                     max_rotate_degree=0.0,
#                     max_translate_ratio=0.04591,
#                     max_shear_degree=0.0,
#                     scaling_ratio_range=(
#                         0.24456,
#                         1.7554400000000001,
#                     ),
#                     border=(
#                         0,
#                         0,
#                     ),
#                     border_val=(
#                         114,
#                         114,
#                         114,
#                     )),
#                 dict(
#                     type='LetterResize',
#                     scale=(
#                         512,
#                         512,
#                     ),
#                     allow_scale_up=True,
#                     pad_val=dict(img=114)),
#             ],
#         ],
#         prob=[0.85834,0.14166,]),


# dict(
#         type='mmdet.Albu',
#         transforms=[
#             dict(type='Blur', p=0.01),
#             dict(type='MedianBlur', p=0.01),
#             dict(type='ToGray', p=0.01),
#             dict(type='CLAHE', p=0.01),
#         ],
#         bbox_params=dict(
#             type='BboxParams',
#             format='pascal_voc',
#             label_fields=[
#                 'gt_bboxes_labels',
#                 'gt_ignore_flags',
#             ]),
#         keymap=dict(img='image', gt_bboxes='bboxes')),