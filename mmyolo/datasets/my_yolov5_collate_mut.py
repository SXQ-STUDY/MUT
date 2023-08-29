# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Sequence

import numpy as np
import torch
from mmengine.dataset import COLLATE_FUNCTIONS

from ..registry import TASK_UTILS
from typing import List, Union
import torch.nn.functional as F


def stack_batch(tensor_list: List[torch.Tensor],
                pad_size_divisor: int = 1,
                pad_value: Union[int, float] = 0) -> torch.Tensor:
    """Stack multiple tensors to form a batch and pad the tensor to the max
    shape use the right bottom padding mode in these images. If
    ``pad_size_divisor > 0``, add padding to ensure the shape of each dim is
    divisible by ``pad_size_divisor``.

    Args:
        tensor_list (List[Tensor]): A list of tensors with the same dim.
        pad_size_divisor (int): If ``pad_size_divisor > 0``, add padding
            to ensure the shape of each dim is divisible by
            ``pad_size_divisor``. This depends on the model, and many
            models need to be divisible by 32. Defaults to 1
        pad_value (int, float): The padding value. Defaults to 0.

    Returns:
       Tensor: The n dim tensor.
    """
    assert isinstance(
        tensor_list,
        list), (f'Expected input type to be list, but got {type(tensor_list)}')
    assert tensor_list, '`tensor_list` could not be an empty list'
    assert len({
        tensor.ndim
        for tensor in tensor_list
    }) == 1, (f'Expected the dimensions of all tensors must be the same, '
              f'but got {[tensor.ndim for tensor in tensor_list]}')

    dim = tensor_list[0].dim()
    num_img = len(tensor_list)
    all_sizes: torch.Tensor = torch.Tensor(
        [tensor.shape for tensor in tensor_list])
    max_sizes = torch.ceil(
        torch.max(all_sizes, dim=0)[0] / pad_size_divisor) * pad_size_divisor
    padded_sizes = max_sizes - all_sizes
    # The first dim normally means channel,  which should not be padded.
    padded_sizes[:, 0] = 0
    if padded_sizes.sum() == 0:
        return torch.stack(tensor_list)
    # `pad` is the second arguments of `F.pad`. If pad is (1, 2, 3, 4),
    # it means that padding the last dim with 1(left) 2(right), padding the
    # penultimate dim to 3(top) 4(bottom). The order of `pad` is opposite of
    # the `padded_sizes`. Therefore, the `padded_sizes` needs to be reversed,
    # and only odd index of pad should be assigned to keep padding "right" and
    # "bottom".
    pad = torch.zeros(num_img, 2 * dim, dtype=torch.int)
    pad[:, 1::2] = padded_sizes[:, range(dim - 1, -1, -1)]
    batch_tensor = []
    for idx, tensor in enumerate(tensor_list):
        batch_tensor.append(
            F.pad(tensor, tuple(pad[idx].tolist()), value=pad_value))
    return batch_tensor



@COLLATE_FUNCTIONS.register_module()
def yolov5_collate_mut(
    data_batch: Sequence,
    use_ms_training: bool = False,
    pad_size_divisor=1) -> dict:
    """Rewrite collate_fn to get faster training speed.

    Args:
       data_batch (Sequence): Batch of data.
       use_ms_training (bool): Whether to use multi-scale training.
    """
    batch_imgs = []
    batch_bboxes_labels = []
    batch_gt_sem_segs = []
    batch_masks = []
    batch_img_metas = []
    for i in range(len(data_batch)):
        batch_img_metas.append(data_batch[i]['data_samples'])
    
    for i in range(len(data_batch)):
        datasamples = data_batch[i]['data_samples']
        inputs = data_batch[i]['inputs']
        batch_imgs.append(inputs)

        gt_bboxes = datasamples.gt_instances.bboxes.tensor
        gt_labels = datasamples.gt_instances.labels
        gt_sem_segs = datasamples.gt_sem_seg.sem_seg
        batch_gt_sem_segs.append(gt_sem_segs)
        
        if 'masks' in datasamples.gt_instances:
            masks = datasamples.gt_instances.masks.to_tensor(
                dtype=torch.bool, device=gt_bboxes.device)
            batch_masks.append(masks)
        batch_idx = gt_labels.new_full((len(gt_labels), 1), i)
        bboxes_labels = torch.cat((batch_idx, gt_labels[:, None], gt_bboxes),
                                  dim=1)
        batch_bboxes_labels.append(bboxes_labels)

    collated_results = {
        'data_samples': {
            'bboxes_labels': torch.cat(batch_bboxes_labels, 0),
            'gt_sem_segs': torch.cat(batch_gt_sem_segs, 0)
        }
    }
    collated_results['batch_img_metas'] = batch_img_metas
    if len(batch_masks) > 0:
        collated_results['data_samples']['masks'] = torch.cat(batch_masks, 0)

    # stack_batch(batch_imgs, pad_size_divisor, pad_value=0)
    if use_ms_training:
        collated_results['inputs'] = batch_imgs
    else:
        collated_results['inputs'] = torch.stack(batch_imgs, 0)
    return collated_results