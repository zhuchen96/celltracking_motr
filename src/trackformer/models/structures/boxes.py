# ------------------------------------------------------------------------
# Modified from Detectron2 (https://github.com/facebookresearch/detectron2)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch
from typing import List, Tuple


class Boxes:
    """Nx4 bounding box container (x1, y1, x2, y2)."""

    def __init__(self, tensor: torch.Tensor):
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            tensor = tensor.reshape((-1, 4)).to(dtype=torch.float32, device=device)
        assert tensor.dim() == 2 and tensor.size(-1) == 4, tensor.size()
        self.tensor = tensor

    def clone(self) -> "Boxes":
        return Boxes(self.tensor.clone())

    def to(self, device: torch.device):
        return Boxes(self.tensor.to(device=device))

    def area(self) -> torch.Tensor:
        box = self.tensor
        area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
        return area

    def __getitem__(self, item) -> "Boxes":
        if isinstance(item, int):
            return Boxes(self.tensor[item].view(1, -1))
        b = self.tensor[item]
        assert b.dim() == 2, "Indexing on Boxes with {} failed to return a matrix!".format(item)
        return Boxes(b)

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        return "Boxes(" + str(self.tensor) + ")"

    @classmethod
    def cat(cls, boxes_list: List["Boxes"]) -> "Boxes":
        assert isinstance(boxes_list, (list, tuple))
        if len(boxes_list) == 0:
            return cls(torch.empty(0))
        assert all([isinstance(box, Boxes) for box in boxes_list])
        cat_boxes = cls(torch.cat([b.tensor for b in boxes_list], dim=0))
        return cat_boxes

    @property
    def device(self) -> torch.device:
        return self.tensor.device


def pairwise_intersection(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    boxes1, boxes2 = boxes1.tensor, boxes2.tensor
    width_height = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(
        boxes1[:, None, :2], boxes2[:, :2]
    )
    width_height.clamp_(min=0)
    intersection = width_height.prod(dim=2)
    return intersection


def pairwise_iou(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    area1 = boxes1.area()
    area2 = boxes2.area()
    inter = pairwise_intersection(boxes1, boxes2)
    iou = torch.where(
        inter > 0,
        inter / (area1[:, None] + area2 - inter),
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )
    return iou


def matched_boxlist_iou(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    assert len(boxes1) == len(boxes2)
    area1 = boxes1.area()
    area2 = boxes2.area()
    box1, box2 = boxes1.tensor, boxes2.tensor
    lt = torch.max(box1[:, :2], box2[:, :2])
    rb = torch.min(box1[:, 2:], box2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    iou = inter / (area1 + area2 - inter)
    return iou
