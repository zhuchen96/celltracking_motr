import cv2 
import json
import numpy as np
from itertools import groupby
from skimage.measure import label
import random
import re

random.seed(24)

def get_info(dataset):
    if dataset == 'moma':
        info = {
            'contributor': 'Dunlop Lab (Owen OConnor)',
            'date_created':'2022',
            'description':'E. Coli growing in mother machine',
            'version': '1.0',
            'year': '2024'
            }
    elif dataset == '2D':
        info = {
            'contributor': 'Simon van Vliet',
            'paper':'Spatially Correlated Gene Expression in Bacterial Groups: The Role of Lineage History, Spatial Gradients, and Cell-Cell Interactions (2018 van Vliet et al.)',
            'description':'E. Coli and Salmonella growing on agarose pads',
            'version': '1.0',
            'year': '2024'
            }
    elif dataset == 'DynamicNuclearNet-tracking-v1_0':
        info = {
            'contributor': 'Van Valen Lab',
            'paper':'Caliban: Accurate cell tracking and lineage construction in live-cell imaging experiments with deep learning (2023 M. Schwartz et al.)',
            'version': '1.0',
            'year': '2024'
            }
    elif dataset == 'sim':
        info = {
            'contributor': 'NA',
            'paper':'NA',
            'version': 'NA',
            'year': 'NA'
            }
        
    return info

def create_folders(datapath,folders):

    # Create annotations folder
    (datapath / 'annotations').mkdir(exist_ok=True)
    (datapath / 'man_track').mkdir(exist_ok=True)

    # Create train and val folder to store images and ground truths
    for folder in folders:

        (datapath / 'man_track' / folder).mkdir(exist_ok=True)
        man_track_paths = (datapath / 'man_track' / folder).glob('*.txt')
        for man_track_path in man_track_paths:
            man_track_path.unlink()

        (datapath / 'annotations' / folder).mkdir(exist_ok=True)
        (datapath / folder).mkdir(exist_ok=True)
        for img_type in ['img','gt']:
            (datapath / folder / img_type).mkdir(exist_ok=True)

    # Remove all data (images + json file) from folders
    for folder in folders:
        for img_type in ['img','gt']:
            delete_fps = (datapath / folder / img_type).glob('*.tif')
            for delete_fp in delete_fps:
                delete_fp.unlink()

        json_paths = (datapath / 'annotations' / folder).glob('*.json')
        for json_path in json_paths:
            json_path.unlink()

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle

def polygonFromMask(seg):
    # adapted from https://github.com/hazirbas/coco-json-converter/blob/master/generate_coco_json.py
    contours, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    valid_poly = 0
    for contour in contours:
    # Valid polygons have >= 6 coordinates (3 points)
        if contour.size >= 6:
            segmentation.append(contour.astype(float).flatten().tolist())
            valid_poly += 1
    if valid_poly == 0:
        raise ValueError
    return segmentation


class reader():
    def __init__(self,dataset,target_size,resize,min_area):

        self.target_size = target_size
        self.min_area = min_area
        self.resize = resize
        self.dtype = {'uint8': 255,
                      'uint16': 65535}
        
        self.crop = False
        self.rescale = False
        self.remove_min = False

        self.max_num_of_cells = 0
    
    def load_track_file(self,track_file):
        self.track_file_orig = track_file
        
    def reset_track_file(self):
        self.track_file = self.track_file_orig.copy()
        self.crop_track_file = np.zeros_like(self.track_file)
        self.removed_cellnbs = []
        self.swap_cellnbs = {}
        self.max_cellnb = self.track_file[-1,0]

    def read_gts(self,fps):

        gts = []

        dataset_nb = fps[0].parts[-2]

        for fp in fps:

            gt_fp = fp.parents[1] / (dataset_nb + '_GT') / 'TRA' / (f'man_track{fp.name[1:]}')

            # Read inputs and outputs
            gts.append(cv2.imread(str(gt_fp),cv2.IMREAD_UNCHANGED).astype(np.uint16))

        self.gts = np.stack(gts)

    def get_slices(self,seg,shift):
    
        y,x = np.where(seg)
        y,x = int(np.mean(y)) + shift[0] , int(np.mean(x)) + shift[1]

        if y < 0:
            y = 0
        elif y > seg.shape[0]:
            y = seg.shape[0]

        if x < 0:
            x = 0
        elif x > seg.shape[1]:
            x = seg.shape[1]
            
        if (y - self.target_size[0]/2 > 0 and y + self.target_size[0]/2 - 1 < seg.shape[0]):
            y0 = int(y - self.target_size[0] / 2)
        elif y + self.target_size[0]/2 - 1 > seg.shape[0] and seg.shape[0] - self.target_size[0] > 0:
            y0 = seg.shape[0] - self.target_size[0]
        else:
            y0 = 0
            
        y1 = y0 + self.target_size[0] if y0 + self.target_size[0] < seg.shape[0] else seg.shape[0]
        
        if (x - self.target_size[1]/2 > 0 and x + self.target_size[1]/2 - 1 < seg.shape[1]):
            x0 = int(x - self.target_size[1] / 2)
        elif x + self.target_size[1]/2 - 1 > seg.shape[1] and seg.shape[1] - self.target_size[1] > 0:
            x0 = seg.shape[1] - self.target_size[1]
        else: 
            x0 = 0
            
        x1 = x0 + self.target_size[1] if x0 + self.target_size[1] < seg.shape[1] else seg.shape[1]   

        self.y = [y0,y1]
        self.x = [x0,x1]

    def read_image(self,fp):

        img = cv2.imread(str(fp),cv2.IMREAD_ANYDEPTH)

        if self.rescale:
            img = ((img - np.min(img)) / np.ptp(img))
            img.shape == self.target_size
        else:
            img = img / np.max(img)

        if self.crop:
            img = img[self.y[0]:self.y[1],self.x[0]:self.x[1]]

            if img.shape[0] < self.target_size[0]:
                img = np.pad(img,((0,self.target_size[0] - img.shape[0]),(0,0)))

            if img.shape[1] < self.target_size[1]:
                img = np.pad(img,((0,0),(0,self.target_size[1] - img.shape[1])))
                
        elif self.resize:
            img = cv2.resize(img,(self.target_size[1],self.target_size[0]))

        img = (255 * img).astype(np.uint8)

        return img

    def get_swapped_cellnb(self,gt,cellnb):
        if isinstance(self.swap_cellnbs[cellnb],list):
            mask = ((gt == cellnb)*255).astype(np.uint8)

            mask_label = label(mask)
            mask_cellnbs = np.unique(mask_label)
            mask_cellnbs = mask_cellnbs[mask_cellnbs!=0]  

            for mask_cellnb in mask_cellnbs:
                if (mask_label == mask_cellnb).sum() < self.min_area:
                    mask_cellnbs = mask_cellnbs[mask_cellnbs != mask_cellnb]          

            if len(mask_cellnbs) == 1:
                cell_1_area = (self.prev_gt == self.swap_cellnbs[cellnb][0]).sum() 
                cell_2_area = (self.prev_gt == self.swap_cellnbs[cellnb][1]).sum()

                if cell_1_area > cell_2_area:
                    new_cellnb = self.swap_cellnbs[cellnb][0]
                else:
                    new_cellnb = self.swap_cellnbs[cellnb][1]
            else:
                assert len(mask_cellnbs) == 2
                cellnb_1, cellnb_2 = self.swap_cellnbs[cellnb]

                if cellnb_1 in self.prev_gt and cellnb_1 in self.prev_gt:

                    y_1,x_1 = np.where(self.prev_gt == cellnb_1)
                    y_2,x_2 = np.where(self.prev_gt == cellnb_2)

                    y_1_prev,x_1_prev = int(np.mean(y_1)), int(np.mean(x_1))
                    y_2_prev,x_2_prev = int(np.mean(y_2)), int(np.mean(x_2))

                    y_1,x_1 = np.where(mask_label == mask_cellnbs[0])
                    y_2,x_2 = np.where(mask_label == mask_cellnbs[1])

                    y_1_cur,x_1_cur = int(np.mean(y_1)), int(np.mean(x_1))
                    y_2_cur,x_2_cur = int(np.mean(y_2)), int(np.mean(x_2))

                    dist_1_1_2_2 = np.power(y_1_prev - y_1_cur,2) + np.power(x_1_prev - x_1_cur,2) + np.power(y_2_prev - y_2_cur,2) + np.power(x_2_prev - x_2_cur,2) 
                    dist_1_2_1_2 = np.power(y_1_prev - y_2_cur,2) + np.power(x_1_prev - x_2_cur,2) + np.power(y_2_prev - y_1_cur,2) + np.power(x_2_prev - x_1_cur,2) 
                    
                    if dist_1_1_2_2 < dist_1_2_1_2:
                        cellnb_1, cellnb_2 = cellnb_2, cellnb_1

                    self.crop_track_file[cellnb_1-1,2] = self.framenb 
                    self.crop_track_file[cellnb_2-1,2] = self.framenb

                    gt[mask_label == mask_cellnbs[0]] = cellnb_1
                    gt[mask_label == mask_cellnbs[1]] = cellnb_2

                    return None

        else:
            new_cellnb = self.swap_cellnbs[cellnb]

        return new_cellnb

    def read_gt(self,fp,counter):

        nb = re.findall('\d+',fp.stem)[-1]
        framenb = int(nb)
        self.framenb = framenb
        dataset_nb = fp.parts[-2]
        gt_fp = fp.parents[1] / (dataset_nb + '_GT') / 'TRA' / (f'man_track{fp.name}')

        gt = self.gts[counter]

        skip = []

        # Crop or resize inputs and outputs to target_size
        if self.crop:
            gt = gt[self.y[0]:self.y[1],self.x[0]:self.x[1]]

            if gt.shape[0] < self.target_size[0]:
                gt = np.pad(gt,((0,self.target_size[0] - gt.shape[0]),(0,0)))

            if gt.shape[1] < self.target_size[1]:
                gt = np.pad(gt,((0,0),(0,self.target_size[1] - gt.shape[1])))

            cellnbs = np.unique(gt)
            cellnbs = cellnbs[cellnbs != 0]

            self.max_num_of_cells = max(self.max_num_of_cells,len(cellnbs))

            for cellnb in cellnbs:

                if cellnb in skip:
                    continue

                if cellnb in self.swap_cellnbs.keys():

                    new_cellnb = self.get_swapped_cellnb(gt,cellnb)

                    while new_cellnb in self.swap_cellnbs:
                        new_cellnb = self.get_swapped_cellnb(gt,new_cellnb)

                    if new_cellnb is None:
                        continue

                    gt[gt == cellnb] = new_cellnb
                    old_cellnb = cellnb
                    cellnb = new_cellnb
                else:
                    old_cellnb = cellnb

                mask = ((gt == cellnb)*255).astype(np.uint8)

                mask_label = label(mask)
                mask_cellnbs = np.unique(mask_label)
                mask_cellnbs = mask_cellnbs[mask_cellnbs!=0]

                for mask_cellnb in mask_cellnbs:

                    mask_sc = ((mask_label == mask_cellnb)*255).astype(np.uint8)
                    contours, _ = cv2.findContours(mask_sc, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if sum([contour.size >= 6 for contour in contours]) == 0 or (mask_sc > 127).sum() < self.min_area:
                        gt[mask_sc.astype(bool)] = 0
                        mask_label[mask_sc.astype(bool)] = 0

                mask_cellnbs = np.unique(mask_label)
                mask_cellnbs = mask_cellnbs[mask_cellnbs != 0]

                if len(mask_cellnbs) == 1:
                    if cellnb in self.crop_track_file[:,0]:
                        if self.crop_track_file[cellnb-1,2] == framenb - 1: 
                            self.crop_track_file[cellnb-1,2] = framenb
                        else:
                            new_cellnb = self.crop_track_file.shape[0]+1
                            self.crop_track_file = np.concatenate((self.crop_track_file,np.array([[new_cellnb,framenb,framenb,0]])),axis=0)
                            gt[mask_label == mask_cellnbs[0]] = new_cellnb
                            assert cellnb not in self.swap_cellnbs
                            self.swap_cellnbs[cellnb] = new_cellnb
                    else:
                        
                        if counter > 0 and self.track_file[old_cellnb-1,-1] != 0 and self.track_file[self.track_file[old_cellnb-1,-1]-1,2] == framenb-1:
                            mother_cellnb = self.track_file[cellnb-1,-1]
                            other_cellnb = self.track_file[(self.track_file[:,-1] == mother_cellnb) * (self.track_file[:,0] != cellnb),0][0]

                            if other_cellnb in gt and (gt == other_cellnb).sum() > self.min_area:
                                if mother_cellnb in self.swap_cellnbs:
                                    if isinstance(self.swap_cellnbs[mother_cellnb],list):
                                        cellnb_1, cellnb_2 = self.swap_cellnbs[mother_cellnb]
                                        y_1,x_1 = np.where(self.prev_gt == cellnb_1)
                                        y_2,x_2 = np.where(self.prev_gt == cellnb_2)

                                        y_1_prev,x_1_prev = int(np.mean(y_1)), int(np.mean(x_1))
                                        y_2_prev,x_2_prev = int(np.mean(y_2)), int(np.mean(x_2))

                                        y_1,x_1 = np.where(gt == cellnb)
                                        y_2,x_2 = np.where(gt == other_cellnb)

                                        y_1_cur,x_1_cur = int(np.mean(y_1)), int(np.mean(x_1))
                                        y_2_cur,x_2_cur = int(np.mean(y_2)), int(np.mean(x_2))

                                        dist_1_1_2_2 = np.power(y_1_prev - y_1_cur,2) + np.power(x_1_prev - x_1_cur,2) + np.power(y_2_prev - y_2_cur,2) + np.power(x_2_prev - x_2_cur,2) 
                                        dist_1_2_1_2 = np.power(y_1_prev - y_2_cur,2) + np.power(x_1_prev - x_2_cur,2) + np.power(y_2_prev - y_1_cur,2) + np.power(x_2_prev - x_1_cur,2) 
                                        
                                        if dist_1_1_2_2 < dist_1_2_1_2:
                                            cellnb_1, cellnb_2 = cellnb_2, cellnb_1

                                        self.crop_track_file[cellnb_1-1,2] = framenb 
                                        self.crop_track_file[other_cellnb-1,2] = framenb

                                        del self.swap_cellnbs[mother_cellnb]
                                        self.swap_cellnbs[cellnb] = cellnb_1
                                        self.swap_cellnbs[other_cellnb] = cellnb_2

                                        gt[gt == cellnb] = cellnb_1
                                        gt[gt == other_cellnb] = cellnb_2

                                    else:
                                        new_mother_cellnb = self.swap_cellnbs[mother_cellnb]

                                        if self.crop_track_file[new_mother_cellnb-1,2] != framenb-1:
                                            new_mother_cellnb = 0

                                        self.crop_track_file[cellnb-1] = np.array([cellnb,framenb,framenb,new_mother_cellnb])
                                        self.crop_track_file[other_cellnb-1] = np.array([other_cellnb,framenb,framenb,new_mother_cellnb])


                                elif self.crop_track_file[mother_cellnb-1,2] == framenb-1:
                                    assert self.crop_track_file[mother_cellnb-1,2] == framenb-1

                                    self.crop_track_file[cellnb-1] = np.array([cellnb,framenb,framenb,mother_cellnb])
                                    self.crop_track_file[other_cellnb-1] = np.array([other_cellnb,framenb,framenb,mother_cellnb])

                                else:
                                    self.crop_track_file[cellnb-1] = np.array([cellnb,framenb,framenb,0])
                                    self.crop_track_file[other_cellnb-1] = np.array([other_cellnb,framenb,framenb,0])                                    

                                skip.append(other_cellnb)

                            else:
                                if self.crop_track_file[mother_cellnb-1,2] != framenb-1:
                                    self.crop_track_file[cellnb-1] = np.array([cellnb,framenb,framenb,0])
                                else:
                                    self.crop_track_file[mother_cellnb-1,2] = framenb
                                    assert cellnb not in self.swap_cellnbs
                                    self.swap_cellnbs[cellnb] = mother_cellnb
                                    gt[gt == cellnb] = mother_cellnb
                        else:
                            self.crop_track_file[cellnb-1] = np.array([cellnb,framenb,framenb,0])


                elif len(mask_cellnbs) > 1:
                    if len(mask_cellnbs) > 2:
                        mask_areas = np.array([(mask_label == mask_cellnb).sum() for mask_cellnb in mask_cellnbs])
                        keep_masks = []
                        for mask_area in mask_areas:
                            keep_masks.append((mask_area > mask_areas[mask_areas != mask_area]).sum() > len(mask_cellnbs)-3)

                        mask_cellnbs = mask_cellnbs[keep_masks]
                        for mask_cellnb in mask_cellnbs:
                            if (mask_label == mask_cellnb).sum() < self.min_area:
                                mask_cellnbs = mask_cellnbs[mask_cellnbs != mask_cellnb] 

                    if self.crop_track_file[cellnb-1,2] == framenb-1 and self.crop_track_file[old_cellnb-1,2] == framenb-1: # cell was in previous frame
                        mother_cellnb = cellnb
                    else:
                        mother_cellnb = 0
                    cellnb_1 = self.crop_track_file.shape[0]+1
                    cellnb_2 = self.crop_track_file.shape[0]+2
                    self.crop_track_file = np.concatenate((self.crop_track_file,np.array([[cellnb_1,framenb,framenb,mother_cellnb]])),axis=0)
                    self.crop_track_file = np.concatenate((self.crop_track_file,np.array([[cellnb_2,framenb,framenb,mother_cellnb]])),axis=0)
                    gt[mask_label == mask_cellnbs[0]] = cellnb_1
                    gt[mask_label == mask_cellnbs[1]] = cellnb_2

                    assert cellnb not in self.swap_cellnbs
                    self.swap_cellnbs[cellnb] = [cellnb_1,cellnb_2]


                else:
                    pass


                # if len(mask_cellnbs) != 1:
                #     if self.temp_track_file[cellnb-1,1] == framenb and self.temp_track_file[cellnb-1,-1] != 0:
                #         mother_cellnb = self.temp_track_file[cellnb-1,-1]
                #         other_cellnb = self.temp_track_file[(self.temp_track_file[:,-1] == mother_cellnb) * (self.temp_track_file[:,0] != cellnb),0][0]

                #         if other_cellnb in gt or (other_cellnb in self.swap_cellnbs.keys() and self.swap_cellnbs[other_cellnb] in gt):
                #             self.temp_track_file[mother_cellnb-1,2] = self.temp_track_file[other_cellnb-1,2]
                #             self.temp_track_file[self.temp_track_file[:,-1] == other_cellnb,-1] = mother_cellnb 

                #         self.temp_track_file[other_cellnb-1] = -1                      
                #         self.removed_cellnbs.append(other_cellnb)

                #         self.temp_track_file[cellnb-1] = -1
                #         self.removed_cellnbs.append(cellnb)

                #     elif self.temp_track_file[cellnb-1,2] > framenb and self.temp_track_file[cellnb-1,1] == framenb:
                #         self.temp_track_file[cellnb-1,1] = framenb + 1

                #     elif self.temp_track_file[cellnb-1,1] < framenb:
                #         last_framenb = self.temp_track_file[cellnb-1,2]
                #         self.temp_track_file[cellnb-1,2] = framenb 

                #         new_row = np.array([[self.temp_track_file.shape[0],framenb + 1, last_framenb,0]])
                #         self.temp_track_file = np.concatenate((self.temp_track_file,new_row),axis=0)
                #         self.swap_cellnbs[cellnb] = self.temp_track_file[-1,0]

                #     else:
                #         self.temp_track_file[cellnb-1] = -1
                #         self.removed_cellnbs.append(cellnb)


        elif self.resize:      
            gt_resized = np.zeros((self.target_size),dtype=np.uint16)
            cellnbs = np.unique(gt)
            cellnbs = cellnbs[cellnbs != 0]
            for cellnb in cellnbs:
                mask_cellnb = ((gt == cellnb)*255).astype(np.uint8)
                mask_cellnb_resized = cv2.resize(mask_cellnb,(self.target_size[1],self.target_size[0]),interpolation= cv2.INTER_NEAREST)

                contours, _ = cv2.findContours(mask_cellnb_resized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                if sum([contour.size >= 6 for contour in contours]) > 0 and ((mask_cellnb_resized > 127).sum() >= self.min_area or not self.remove_min):
                    gt_resized[mask_cellnb_resized > 127] = cellnb
                else:
                    row = self.track_file_orig[self.track_file_orig[:,0] == cellnb][0]

                    if row[2] == framenb:
                        self.track_file_orig[self.track_file_orig[:,0] == cellnb,2] -= 1
                    elif row[1] == framenb:
                        self.track_file_orig[self.track_file_orig[:,0] == cellnb,1] += 1
                    else:
                        old_exit_framenb = row[2]
                        self.track_file_orig[self.track_file_orig[:,0] == cellnb,2] = framenb-1

                        max_cellnb = np.max(self.track_file_orig[:,0]) + 1
                        new_cell = np.array([[max_cellnb,framenb+1,old_exit_framenb,0]])

                        self.track_file_orig = np.concatenate((self.track_file_orig, new_cell))

                        for f in range(1,old_exit_framenb-framenb+1):
                            assert cellnb in self.gts[counter+f]
                            self.gts[counter+f, self.gts[counter+f] == cellnb] = max_cellnb

                    #TODO need to remove cell from man_track.txt if it's too small in terms of area or contour.size

            gt = gt_resized

        else:
            cellnbs = np.unique(gt)
            cellnbs = cellnbs[cellnbs != 0]
            for cellnb in cellnbs:
                mask_cellnb = ((gt == cellnb)*255).astype(np.uint8)

                contours, _ = cv2.findContours(mask_cellnb, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                if sum([contour.size >= 6 for contour in contours]) == 0 or (mask_cellnb > 127).sum() < self.min_area:
                    gt[mask_cellnb.astype(bool)] = 0

        self.prev_gt = gt

        return gt
    

    def clean_track_file(self,savepath,dataset_name,fps,CTC_coco_folder,f,ctc_counter,ann_length,annotations):
        if 0 in self.crop_track_file[:,0]:

            removed_ind = np.where(self.crop_track_file[:,0] == 0)[0]

            og_num = self.crop_track_file[self.crop_track_file[:,0] != 0,0]

            for ind in removed_ind:

                if (self.crop_track_file[ind:,0] < 1).all():
                    continue

                next_cellnb = self.crop_track_file[ind:,0][self.crop_track_file[ind:,0] > 0].min()
                self.crop_track_file[ind:,0] -= 1
                self.crop_track_file[self.crop_track_file[:,-1] >= next_cellnb,-1] -= 1
                self.crop_track_file[self.crop_track_file < 0] = 0

            self.crop_track_file = np.delete(self.crop_track_file,removed_ind,axis=0)

            annotation_copy = annotations[ann_length:].copy()
            cellnbs = self.crop_track_file[:,0]

            for a,ann in enumerate(annotation_copy):
                if isinstance(ann['track_id'],np.uint16) and ann['track_id'] in og_num:
                    annotations[ann_length+a]['track_id'] = cellnbs[og_num == ann['track_id']][0]

            for counter,fp in enumerate(fps):

                framenb = int(re.findall('\d+',fp.name)[-1])

                gt_fp = savepath / 'gt' / f'CTC_{dataset_name}_split_{f:02d}_frame_{framenb:03d}.tif'#_{counter:03d}.tif'
                # gt_fp = savepath / 'gt' / f'{dataset_name}_{f}_{counter:03d}_{fp.name}'
                TRA_fp = savepath.parent / CTC_coco_folder / f'{ctc_counter:03d}_GT' / 'TRA' / fp.name
                SEG_fp = savepath.parent / CTC_coco_folder / f'{ctc_counter:03d}_GT' / 'SEG' / fp.name

                assert TRA_fp.exists() and SEG_fp.exists() and gt_fp.exists()

                outputs = cv2.imread(str(gt_fp), cv2.IMREAD_ANYDEPTH)
                cellnbs = np.unique(outputs)
                cellnbs = cellnbs[cellnbs != 0]

                for cellnb in cellnbs:
                    outputs[outputs == cellnb] = self.crop_track_file[og_num == cellnb,0]

                cv2.imwrite(str(gt_fp),outputs)
                cv2.imwrite(str(TRA_fp),outputs)
                cv2.imwrite(str(SEG_fp),outputs)

        assert self.crop_track_file.shape[0] == self.crop_track_file[-1,0]

        mother_ids = np.unique(self.crop_track_file[:,-1])
        mother_ids = mother_ids[mother_ids != 0]

        for mother_id in mother_ids:
            assert (self.crop_track_file[:,-1] == mother_id).sum() == 2
            assert self.crop_track_file[mother_id-1,2] == self.crop_track_file[self.crop_track_file[:,-1] == mother_id,1][0] -1
            assert self.crop_track_file[mother_id-1,2] == self.crop_track_file[self.crop_track_file[:,-1] == mother_id,1][0] -1

        assert (self.crop_track_file[:,2] >= self.crop_track_file[:,1]).all()

        return annotations

def create_anno(mask,cellnb,image_id,annotation_id,dataset_name):
   
    mask_sc = mask == cellnb 

    mask_sc_label = label(mask_sc)
    mask_sc_ids = np.unique(mask_sc_label)[1:]

    if len(mask_sc_ids) > 0:

        mask_sc_area = [(mask_sc_label==mask_sc_id).sum() for mask_sc_id in mask_sc_ids]
        mask_sc_ind = np.argmax(np.array(mask_sc_area))
        mask_sc_id = mask_sc_ids[mask_sc_ind]

        mask_sc = mask_sc_label == mask_sc_id

        area = float((mask_sc > 0.0).sum())
        seg = polygonFromMask(mask_sc)
                                    
        y, x = np.where(mask_sc != 0)

        X = np.min(x) 
        Y = np.min(y)
        width = (np.max(x) - np.min(x)) 
        height = (np.max(y) - np.min(y)) 
        bbox = (X,Y,width,height)
        empty = False

        if X == 0 or Y == 0 or (X + width) == mask_sc.shape[1]-1 or (Y + height) == mask_sc.shape[0]-1:
            edge=True
        else:
            edge=False

    else: #empty frame
        area = 0
        seg = []
        bbox = []
        empty = True
        edge = None

        assert cellnb == -1

    annotation = {
        'id': annotation_id,
        'image_id': image_id,
        'bbox': bbox,
        'segmentation': seg,
        'area': area,
        'category_id': 1,
        'track_id': cellnb if cellnb > 0 else [],
        'dataset_name': dataset_name,
        'ignore': 0,
        'iscrowd': 0,
        'empty': empty,
        'edge': edge,
    }

    return annotation


