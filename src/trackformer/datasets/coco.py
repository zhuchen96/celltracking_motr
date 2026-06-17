# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""

from collections import Counter

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask
import numpy as np
from . import transforms as T
from PIL import Image

class CocoDetection(torchvision.datasets.CocoDetection):

    fields = ["labels", "area", "iscrowd", "boxes", "track_ids", "masks"]

    def __init__(self,  img_folder, ann_file, transforms, norm_transforms,
                 return_masks=False, overflow_boxes=False, remove_no_obj_imgs=False,
                 min_num_objects=0, crop=False, shift=False):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self._norm_transforms = norm_transforms
        self.prepare = ConvertCocoPolysToMask(return_masks, overflow_boxes)
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.crop = crop
        self.shift = shift

        self.edge_distance = 5 

        annos_image_ids = [ann['image_id'] for ann in self.coco.loadAnns(self.coco.getAnnIds())]
            
        if remove_no_obj_imgs:
            self.ids = sorted(list(set(annos_image_ids)))

        counter = Counter(annos_image_ids)
        
        if min_num_objects:
            self.ids = [i for i in self.ids if counter[i] >= min_num_objects]

    def is_touching_edge(self, target):

        bboxes_xyxy = target['boxes']

        if bboxes_xyxy.shape[1] > 0:
            if self.dataset_type != 'moma':
                is_touching_edge = torch.stack(((bboxes_xyxy[:,0] <= self.edge_distance),(bboxes_xyxy[:,1] <= self.edge_distance), (bboxes_xyxy[:,2] >= self.target_size[1] - self.edge_distance), (bboxes_xyxy[:,3] >= self.target_size[0] - self.edge_distance)),axis=0).any(0)
            else:
                is_touching_edge = (bboxes_xyxy[:,3] >= self.target_size[0] - self.edge_distance)
        else:
            is_touching_edge = torch.zeros_like(target['track_ids']).bool()

        target['is_touching_edge'] = is_touching_edge
        target['is_touching_edge_orig'] = is_touching_edge

        return target

    def _getitem_from_id(self, image_id, framenb, random_state=None):

        img_raw, annotations = super(CocoDetection, self).__getitem__(image_id)
        target = {
            'image_id': torch.tensor(self.ids[image_id]),
            'annotations': annotations,
            'framenb': torch.tensor(framenb),
            'dataset': self.dataset_type,
            'target_size': self.target_size,
        }

        if 'ctc_counter' in annotations[0]:
            target['ctc_counter'] = torch.tensor(annotations[0]['ctc_counter'])

        img_w_raw,img_h_raw = img_raw.size
        h,w = self.target_size[0].item(), self.target_size[1].item()

        # Calculate padding (int() casts avoid PIL overflow when target_size elements are tensors)
        padding_width = max(int(self.target_size[1]) - img_w_raw, 0)
        padding_height = max(int(self.target_size[0]) - img_h_raw, 0)

        # Apply padding (left, top, right, bottom)
        img = Image.new('RGB', (img_w_raw + padding_width, img_h_raw + padding_height), (0, 0, 0))
        img.paste(img_raw, (0, 0))

        img_w, img_h = img.size
        
        if self.crop and self.RandomCrop.region is None:

            if self.shift:
                bbox_areas = torch.tensor([ann['bbox'][2] * ann['bbox'][3] for ann in annotations])
                self.shift_value = torch.sqrt(bbox_areas).mean() / 2
                self.shift_value = self.shift_value * np.power(100 / bbox_areas.shape[0],1/3)

                self.img_w, self.img_h = img_w, img_h
                self.num_cells = bbox_areas.shape[0]

            if img_w > self.target_size[1] or img_h > self.target_size[0]:
                bbox_centers = torch.tensor([[ann['bbox'][0] + ann['bbox'][2]//2, ann['bbox'][1] + ann['bbox'][3]//2] for ann in annotations])
                
                if torch.rand(1) < 0.9:

                    random_index = torch.randint(0, len(bbox_centers), (1,))[0]
                    bbox_center = bbox_centers[random_index]
                    i,j = bbox_center[0].item() - h//2, bbox_center[1].item() - w//2

                    if torch.rand(1) < 0.2:
                        i += torch.randint(-self.target_size[1]//2,self.target_size[1]//2,(1,)).item()
                        j += torch.randint(-self.target_size[0]//2,self.target_size[0]//2,(1,)).item()
                        self.shift_value /= 2
                else:
                    i = torch.randint(0, max(img_h - h,1), (1,)).item()
                    j = torch.randint(0, max(img_w - w,1), (1,)).item()

                if j+h > img_h:
                    j = img_h - h
                elif j < 0:
                    j = 0

                if i+w > img_w:
                    i = img_w - w
                elif i < 0:
                    i = 0

            elif img_w == self.target_size[1] and img_h == self.target_size[0]:
                i = 0
                j = 0
                w = img_w
                h = img_h

            else:
                raise NotImplementedError

            self.RandomCrop.region = [j,i,h,w]

        img = img.crop((0, 0, img_w_raw, img_h_raw))

        img, target = self.prepare(img, target, self.RandomCrop)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        assert target['boxes'].shape[0] == target['flexible_divisions'].shape[0]

        target = self.is_touching_edge(target)

        assert target['boxes'].shape[0] == target['is_touching_edge'].shape[0]

        img, target = self._norm_transforms(img, target)

        return img, target


def convert_coco_poly_to_mask(segmentations, height, width, crop_region):
    if crop_region is not None:
        height, width = crop_region[-2:]

    masks = torch.zeros((len(segmentations), height, width), dtype=torch.uint8)
    for pidx, polygons in enumerate(segmentations):
        if isinstance(polygons, dict):
            if crop_region is not None:
                raise NotImplementedError
            rles = {'size': polygons['size'],
                    'counts': polygons['counts'].encode(encoding='UTF-8')}
            masks[pidx] = torch.from_numpy(coco_mask.decode(rles)[:,:,0])
        else:
            assert len(polygons) == 1
            polygon = polygons[0]

            if crop_region is not None:
                adjusted_polygon = []

                for poly in polygon:

                    adjusted_poly = []
                    for i in range(0, len(poly), 2):
                        adjusted_x = poly[i] - crop_region[1]
                        adjusted_y = poly[i+1] - crop_region[0]
                        adjusted_poly.extend([adjusted_x, adjusted_y])

                    adjusted_polygon.append(adjusted_poly)

                polygon = adjusted_polygon

            rles = coco_mask.frPyObjects(polygon, height, width)

            masks[pidx] = torch.from_numpy(coco_mask.decode(rles)[:,:,0])
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False, overflow_boxes=False):
        self.return_masks = return_masks
        self.overflow_boxes = overflow_boxes

    def __call__(self, image, target, RandomCrop):
        w, h = image.size

        anno = target.pop("annotations")

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        empty = anno[0]['empty']

        boxes = torch.as_tensor([obj["bbox"] for obj in anno], dtype=torch.float32)

        if not empty:
            # x,y,w,h --> x,y,x,y
            boxes[:, 2:4] += boxes[:, :2]
            
            if not self.overflow_boxes:
                boxes[:, 0::2].clamp_(min=0, max=w)
                boxes[:, 1::2].clamp_(min=0, max=h)

            if not empty:
                assert ((boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])).all() == True, 'Boxes are not correct in coco.py script'

        target["boxes"] = boxes

        if 'segmentation' in anno[0]:
            if not empty:
                segmentations = [[obj["segmentation"]] for obj in anno]
                masks = convert_coco_poly_to_mask(segmentations, h, w, RandomCrop.region)
            else:
                masks = torch.zeros_like(torchvision.transforms.ToTensor()(image)[0])
                masks = torch.zeros((image.size[1],image.size[0]),dtype=torch.float32,device=boxes.device)
            target["masks"] = masks

        # for conversion to coco api
        track_ids = torch.tensor([obj["track_id"] for obj in anno])
        classes = torch.tensor([obj["category_id"] for obj in anno]) 
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])

        classes = torch.stack((classes,2 * torch.ones_like(classes)),axis=1)

        target['track_ids'] = track_ids
        target['flexible_divisions'] = torch.zeros_like(track_ids).bool()
        assert target['boxes'].shape[0] == target['flexible_divisions'].shape[0]
        target["labels"] = classes - 1
        target["area"] = area
        target["iscrowd"] = iscrowd

        target["size"] = torch.as_tensor([int(h), int(w)])
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["empty"] = torch.tensor(empty)

        if target['dataset'] == '2D':
            image, target = RandomCrop(image, target)
            
        assert target['boxes'].shape[0] == target['flexible_divisions'].shape[0]

        if target['boxes'].shape[0] == 0 or target['boxes'].shape[1] == 0:
            target['empty'] = torch.tensor(True)
            
        target.pop('dataset')

        target['track_ids_orig'] = target['track_ids'].clone()
        target['flexible_divisions_orig'] = torch.zeros_like(target['track_ids']).bool()
        target["labels_orig"] = target["labels"].clone()

        target['boxes'] = torch.cat((target['boxes'],torch.zeros_like(target['boxes'])),axis=1)

        if self.return_masks:
            target['masks'] = torch.stack((target['masks'],torch.zeros_like(target['masks'])),axis=1)
            target["masks_orig"] =  target["masks"].clone()
        elif 'masks' in target:
            target.pop('masks')

        target["boxes_orig"] = target["boxes"].clone()

        return image, target


def make_coco_transforms_cells(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if image_set == 'train':
        transforms = [
            T.RandomGaussianBlur(),
            T.RandomGaussianNoise(),
            T.RandomIlluminationVoodoo(),      
        ]
    elif image_set == 'val':
        transforms = []
    else:
        ValueError(f'unknown {image_set}')

    return T.Compose(transforms), normalize
