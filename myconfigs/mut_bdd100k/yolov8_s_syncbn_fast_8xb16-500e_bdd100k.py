_base_ = ['../_base_/default_runtime.py', '../_base_/det_p5_tta.py']

# ========================Frequently modified parameters======================
# -----data related-----
data_root = r'D:\sxq\datasets\bdd100k_mut'  # Root path of data

num_classes = 1  # Number of classes for classification
# Batch size of a single GPU during training
train_batch_size_per_gpu = 16
# Worker to pre-fetch data for each single GPU during training
train_num_workers = 2
# persistent_workers must be False if num_workers is 0
persistent_workers = True

# -----train val related-----
# Base learning rate for optim_wrapper. Corresponding to 8xb16=64 bs
base_lr = 0.01
max_epochs = 100  # Maximum training epochs
# Disable mosaic augmentation for final 10 epochs (stage 2)
close_mosaic_epochs = 10

model_test_cfg = dict(
    # The config of multi-label for multi-class prediction.
    multi_label=True,
    # The number of boxes before NMS
    nms_pre=30000,
    score_thr=0.001,  # Threshold to filter out boxes.
    nms=dict(type='nms', iou_threshold=0.7),  # NMS type and threshold
    max_per_img=300)  # Max number of detections of each image

# ========================Possible modified parameters========================
# -----data related-----
img_scale = (512, 512)  # width, height
crop_size = (1024, 1024)


# Batch size of a single GPU during validation
val_batch_size_per_gpu = 1
# Worker to pre-fetch data for each single GPU during validation
val_num_workers = 2

# Config of batch shapes. Only on val.
# We tested YOLOv8-m will get 0.02 higher than not using it.
batch_shapes_cfg = None
# You can turn on `batch_shapes_cfg` by uncommenting the following lines.
# batch_shapes_cfg = dict(
#     type='BatchShapePolicy',
#     batch_size=val_batch_size_per_gpu,
#     img_size=img_scale[0],
#     # The image scale of padding should be divided by pad_size_divisor
#     size_divisor=32,
#     # Additional paddings for pixel scale
#     extra_pad_ratio=0.5)

# -----model related-----
# The scaling factor that controls the depth of the network structure
deepen_factor = 0.33
# The scaling factor that controls the width of the network structure
widen_factor = 0.5
# Strides of multi-scale prior box
strides = [8, 16, 32]
# The output channel of the last stage
last_stage_out_channels = 1024
num_det_layers = 3  # The number of model output scales
norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)  # Normalization config

# -----train val related-----
affine_scale = 0.5  # YOLOv5RandomAffine scaling ratio
# YOLOv5RandomAffine aspect ratio of width and height thres to filter bboxes
max_aspect_ratio = 100
tal_topk = 10  # Number of bbox selected in each level
tal_alpha = 0.5  # A Hyper-parameter related to alignment_metrics
tal_beta = 6.0  # A Hyper-parameter related to alignment_metrics
# TODO: Automatically scale loss_weight based on number of detection layers
loss_cls_weight = 0.5
loss_bbox_weight = 7.5
# Since the dfloss is implemented differently in the official
# and mmdet, we're going to divide loss_weight by 4.
loss_dfl_weight = 1.5 / 4
lr_factor = 0.01  # Learning rate scaling factor
weight_decay = 0.0005
# Save model checkpoint and validation intervals in stage 1
save_epoch_intervals = 1
# validation intervals in stage 2
val_interval_stage2 = 1
# The maximum checkpoints to keep.
max_keep_ckpts = 2
# Single-scale training is recommended to
# be turned on, which can speed up training.
env_cfg = dict(cudnn_benchmark=True)

# ===============================Unmodified in most cases====================
model = dict(
    type='mmdet.Mut_SingleStageDetector',
    data_preprocessor=dict(
        type='mmdet.MUT_DetDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        pad_size_divisor=32,
        pad_seg=True,
        bgr_to_rgb=True),
    backbone=dict(
        type='YOLOv8CSPDarknet',
        arch='P5',
        last_stage_out_channels=last_stage_out_channels,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True)),
    neck_det=dict(
        type='YOLOv8PAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, last_stage_out_channels],
        out_channels=[256, 512, last_stage_out_channels],
        num_csp_blocks=3,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True)),
    neck_seg=dict(
        type='mmseg.FPN',
        in_channels=[256, 512, last_stage_out_channels],
        out_channels=256,
        num_outs=1),
    bbox_head=dict(
        type='YOLOv8Head',
        head_module=dict(
            type='YOLOv8HeadModule',
            num_classes=num_classes,
            in_channels=[256, 512, last_stage_out_channels],
            widen_factor=widen_factor,
            reg_max=16,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='SiLU', inplace=True),
            featmap_strides=strides),
        prior_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0.5, strides=strides),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        # scaled based on number of detection layers
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='none',
            loss_weight=loss_cls_weight),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='ciou',
            bbox_format='xyxy',
            reduction='sum',
            loss_weight=loss_bbox_weight,
            return_iou=False),
        loss_dfl=dict(
            type='mmdet.DistributionFocalLoss',
            reduction='mean',
            loss_weight=loss_dfl_weight)),
    seg_head=dict(
        type='mmseg.FCNHead',
        in_channels=256,
        in_index=0,
        channels=128,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    det_train_cfg=dict(
        assigner=dict(
            type='BatchTaskAlignedAssigner',
            num_classes=num_classes,
            use_ciou=True,
            topk=tal_topk,
            alpha=tal_alpha,
            beta=tal_beta,
            eps=1e-9)),
    det_test_cfg=model_test_cfg)

pre_transform = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(type='LoadAnnotations', with_bbox=True, with_seg=True)
]

train_pipeline = [
    *pre_transform,
    # dict(type='YOLOv5KeepRatioResize', scale=img_scale),# scale为(long edge,short edge),顺序无所谓
    dict(
        type='mmdet.RandomResize',
        scale=(1280, 720),
        ratio_range=(0.6, 1.2),
        resize_type='mmdet.Resize',
        keep_ratio=True),
    dict(type='mmdet.RandomCrop', crop_size=crop_size),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmseg.GenerateEdge', edge_width=4),
    dict(
        type='mmdet.MUT_PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction', 'scale_factor'))
]

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    # collate_fn=dict(type='yolov5_collate'),
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

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(type='mmdet.Resize', scale=(1024, 576), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_seg=True),
    dict(
        type='mmdet.MUT_PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=persistent_workers,
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
        batch_shapes_cfg=batch_shapes_cfg))

test_dataloader = val_dataloader

param_scheduler = None
optim_wrapper = dict(
    type='OptimWrapper',
    clip_grad=dict(max_norm=10.0),
    optimizer=dict(
        type='SGD',
        lr=base_lr,
        momentum=0.937,
        weight_decay=weight_decay,
        nesterov=True,
        batch_size_per_gpu=train_batch_size_per_gpu),
    constructor='YOLOv5OptimizerConstructor')

default_hooks = dict(
    param_scheduler=dict(
        type='YOLOv5ParamSchedulerHook',
        scheduler_type='linear',
        lr_factor=lr_factor,
        max_epochs=max_epochs),
    checkpoint=dict(
        type='CheckpointHook',
        interval=save_epoch_intervals,
        save_best='auto',
        max_keep_ckpts=max_keep_ckpts))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49)
]
# ori_coco_metric
# val_evaluator = dict(
#     type='mmdet.CocoMetric',
#     proposal_nums=(100, 1, 10),
#     ann_file=data_root + '/VOC2007test/ImageSets/Main/test.txt',
#     metric='bbox')
# pascal voc metric
# val_evaluator = [
#     dict(type='mmdet.VOCMetric', metric='mAP', eval_mode='area'),
#     dict(type='mmseg.MUT_IoUMetric', iou_metrics=['mIoU'])]
val_evaluator = [
    dict(
        type='mmdet.MUT_CocoMetric',
        ann_file = r'D:\sxq\datasets\bdd100k_mut\bdd100k_coco_ann\bdd100k_mut_coco_val.json', 
        img_txt_path = r'D:\sxq\datasets\bdd100k_mut\train_val_txt\val.txt',
        proposal_nums=(100, 1, 10), 
        metric='bbox'
    ),
    dict(type='mmseg.MUT_IoUMetric', iou_metrics=['mIoU'])]
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=save_epoch_intervals,
    dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                        val_interval_stage2)])

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
