# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .base import BaseDetector
from mmseg.utils import add_prefix

from typing import List, Tuple

from mmengine.model import BaseModel
from mmengine.structures import PixelData
from torch import Tensor

from mmseg.structures import SegDataSample
from mmseg.utils import (ForwardResults, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList)

import warnings

import torch.nn as nn
import torch.nn.functional as F


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)



@MODELS.register_module()
class Mut_SingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck_det: Union[dict, None] = None,
                 neck_seg: Union[dict, None] = None,
                 bbox_head: OptConfigType = None,
                 seg_head: OptConfigType = None,
                 det_train_cfg: OptConfigType = None,
                 det_test_cfg: OptConfigType = None,
                 seg_train_cfg: OptConfigType = None,
                 seg_test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        if neck_det is not None:
            self.neck_det = MODELS.build(neck_det)
        if neck_seg is not None:
            self.neck_seg = MODELS.build(neck_seg)
        bbox_head.update(train_cfg=det_train_cfg)
        bbox_head.update(test_cfg=det_test_cfg)
        self.bbox_head = MODELS.build(bbox_head)
        self.seg_head = MODELS.build(seg_head)
        self.det_train_cfg = det_train_cfg
        self.det_test_cfg = det_test_cfg
        self.seg_train_cfg = seg_train_cfg
        self.seg_test_cfg = seg_test_cfg
        self.align_corners = self.seg_head.align_corners

    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:
        """Exchange bbox_head key to rpn_head key when loading two-stage
        weights into single-stage model."""
        bbox_head_prefix = prefix + '.bbox_head' if prefix else 'bbox_head'
        bbox_head_keys = [
            k for k in state_dict.keys() if k.startswith(bbox_head_prefix)
        ]
        rpn_head_prefix = prefix + '.rpn_head' if prefix else 'rpn_head'
        rpn_head_keys = [
            k for k in state_dict.keys() if k.startswith(rpn_head_prefix)
        ]
        if len(bbox_head_keys) == 0 and len(rpn_head_keys) != 0:
            for rpn_head_key in rpn_head_keys:
                bbox_head_key = bbox_head_prefix + \
                                rpn_head_key[len(rpn_head_prefix):]
                state_dict[bbox_head_key] = state_dict.pop(rpn_head_key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        x_det_neck, x_seg_neck = self.extract_feat(batch_inputs)
        losses = dict()
        if x_det_neck:
            batch_det_samples = {
                'bboxes_labels':batch_data_samples['bboxes_labels'],
                'img_metas':batch_data_samples['img_metas'],
            }
            loss_det = self.bbox_head.loss(x_det_neck, batch_det_samples)
            loss_det = add_prefix(loss_det, 'det')
            losses.update(loss_det)
        if x_seg_neck:
            batch_seg_samples = batch_data_samples['batch_seg_data_samples']
            loss_seg = self.seg_head.loss(x_seg_neck, batch_seg_samples, self.seg_train_cfg)
            loss_seg = add_prefix(loss_seg, 'seg')
            losses.update(loss_seg)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        x_det_neck, x_seg_neck = self.extract_feat(batch_inputs)
        batch_det_data_samples = batch_data_samples['batch_det_data_samples']
        batch_seg_data_samples = batch_data_samples['batch_seg_data_samples']
        # object detection predict
        results_bbox = self.bbox_head.predict(
            x_det_neck, batch_det_data_samples, rescale=rescale
        )
        # get seg_img_metas
        if batch_seg_data_samples is not None:
            batch_seg_img_metas = [
                data_sample.metainfo for data_sample in batch_seg_data_samples
            ]
        else:
            batch_seg_img_metas = [
                dict(
                    ori_shape=batch_inputs.shape[2:],
                    img_shape=batch_inputs.shape[2:],
                    pad_shape=batch_inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * batch_inputs.shape[0]
        # semantic segmentation predict
        seg_logits = self.seg_head.predict(
            x_seg_neck, batch_seg_img_metas, self.seg_test_cfg
        )
        results_seg = self.seg_postprocess_result(seg_logits, batch_seg_data_samples)
        results_object = self.add_pred_to_datasample(batch_det_data_samples, results_bbox)
        
        for r_o, r_s in zip(results_object, results_seg):
            r_o.pred_sem_seg = r_s.pred_sem_seg

        return results_object

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """
        x_det_neck, x_seg_neck = self.extract_feat(batch_inputs)
        results_det = self.bbox_head.forward(x_det_neck)
        results_seg = self.seg_head.forward(x_seg_neck)
        
        return results_det, results_seg

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        x = self.backbone(batch_inputs)
        x_det_neck = None
        x_seg_neck = None
        if hasattr(self, 'neck_det') is not None:
            x_det_neck = self.neck_det(x)
        else:
            x_det_neck = x
        if hasattr(self, 'neck_seg') is not None:
            x_seg_neck = self.neck_seg(x)
        else:
            x_seg_neck = x
        return x_det_neck, x_seg_neck
    
    def seg_postprocess_result(self,
                           seg_logits: Tensor,
                           data_samples: OptSampleList = None) -> SampleList:
        """ Convert results list to `SegDataSample`.
        Args:
            seg_logits (Tensor): The segmentation results, seg_logits from
                model of each input image.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`. Default to None.
        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        batch_size, C, H, W = seg_logits.shape

        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(batch_size)]
            only_prediction = True
        else:
            only_prediction = False

        for i in range(batch_size):
            if not only_prediction:
                img_meta = data_samples[i].metainfo
                # remove padding area
                if 'img_padding_size' not in img_meta:
                    padding_size = img_meta.get('padding_size', [0] * 4)
                else:
                    padding_size = img_meta['img_padding_size']
                padding_left, padding_right, padding_top, padding_bottom =\
                    padding_size
                # i_seg_logits shape is 1, C, H, W after remove padding
                i_seg_logits = seg_logits[i:i + 1, :,
                                          padding_top:H - padding_bottom,
                                          padding_left:W - padding_right]
 
                flip = img_meta.get('flip', None)
                if flip:
                    flip_direction = img_meta.get('flip_direction', None)
                    assert flip_direction in ['horizontal', 'vertical']
                    if flip_direction == 'horizontal':
                        i_seg_logits = i_seg_logits.flip(dims=(3, ))
                    else:
                        i_seg_logits = i_seg_logits.flip(dims=(2, ))

                # resize as original shape
                i_seg_logits = resize(
                    i_seg_logits,
                    size=img_meta['ori_shape'],
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False).squeeze(0)
            else:
                i_seg_logits = seg_logits[i]

            if C > 1:
                i_seg_pred = i_seg_logits.argmax(dim=0, keepdim=True)
            else:
                i_seg_logits = i_seg_logits.sigmoid()
                i_seg_pred = (i_seg_logits >
                              self.decode_head.threshold).to(i_seg_logits)
            data_samples[i].set_data({
                'seg_logits':
                PixelData(**{'data': i_seg_logits}),
                'pred_sem_seg':
                PixelData(**{'data': i_seg_pred})
            })

        return data_samples

