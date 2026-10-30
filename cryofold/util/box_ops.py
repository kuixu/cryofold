# Copyright (c) CryoNet Team, and its affiliates. All Rights Reserved

"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
# from torchvision.ops.boxes import box_area


def box_cxcyczwhl_to_xyzxyz(x):
    x_c, y_c, z_c, w, h, l = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (z_c - 0.5 * l),
         (x_c + 0.5 * w), (y_c + 0.5 * h), (z_c + 0.5 * l)]
    return torch.stack(b, dim=-1)


def box_xyzxyz_to_cxcyczwhl(x):
    x0, y0, z0, x1, y1, z1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2,
         (x1 - x0), (y1 - y0), (z1 - z0)]
    return torch.stack(b, dim=-1)


def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, z1, x2, y2, z2) coordinates.
    Arguments:
        boxes (Tensor[N, 6]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, z1, x2, y2, z2) format
    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 3] - boxes[:, 0]) * (boxes[:, 4] - boxes[:, 1]) * (boxes[:, 5] - boxes[:, 2]) 



# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :3], boxes2[:, :3])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 3:], boxes2[:, 3:])  # [N,M,2]

    whl = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = whl[:, :, 0] * whl[:, :, 1] * whl[:, :, 2] # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 3:] >= boxes1[:, :3]).all()
    assert (boxes2[:, 3:] >= boxes2[:, :3]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :3], boxes2[:, :3])
    rb = torch.max(boxes1[:, None, 3:], boxes2[:, 3:])

    whl = (rb - lt).clamp(min=0)  # [N,M,2]
    area = whl[:, :, 0] * whl[:, :, 1] * whl[:, :, 2]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, W, H, L] where N is the number of masks, (W, H, L) are the spatial dimensions.

    Returns a [N, 6] tensors, with the boxes in xyzxyz format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 6), device=masks.device)

    w, h, l = masks.shape[-3:]

    x = torch.arange(0, w, dtype=torch.float)
    y = torch.arange(0, h, dtype=torch.float)
    z = torch.arange(0, l, dtype=torch.float)
    x, y, z = torch.meshgrid(x, y, z)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    z_mask = (masks * z.unsqueeze(0))
    z_max = z_mask.flatten(1).max(-1)[0]
    z_min = z_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, z_min, x_max, y_max, z_max], 1)
