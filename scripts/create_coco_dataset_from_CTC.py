import cv2 
from pathlib import Path
import json
import numpy as np                                 
from tqdm import tqdm
import utils_coco as utils
import re
from skimage.measure import label

dataset = 'sim' # ['moma','DynamicNuclearNet-tracking-v1_0']
datapath = Path('/srv/home/chen/Cell-TRACTR/data') / dataset / 'COCO'

if dataset == 'moma':
    min_area = 0
    target_size = (256,32)
    resize = True
elif dataset == 'sim':
    min_area = 0
    target_size = (512,512)
    resize = True
else:
    min_area = 0
    target_size = None
    resize = False

if min_area > 0:
    raise NotImplementedError # 'Need to fix divisions if cell removed had just divided'

licenses = [{'id': 1,'name': 'MIT license'}]
categories = [{'id': 1, 'name': 'cell'}]
info = utils.get_info(dataset)

datapath.mkdir(exist_ok=True)
folders = ['train','val']
utils.create_folders(datapath,folders)

img_reader = utils.reader(dataset=dataset, target_size=target_size, resize=resize, min_area=min_area)

train_sets = sorted([x for x in (datapath.parent / 'CTC' / 'train').iterdir() if x.is_dir() and re.findall('\d\d$',x.name)])
val_sets = sorted([x for x in (datapath.parent / 'CTC' / 'val').iterdir() if x.is_dir() and re.findall('\d\d$',x.name)])

for folder,dataset_paths in zip(folders,[train_sets,val_sets]):
    
    image_id = 0   
    images = []
    annotation_id = 0
    annotations = []
    skip = 0

    for dataset_path in dataset_paths:

        fps = sorted(dataset_path.glob('*.tif'))

        if len(fps) < 4: # You need a minimum of 4 frames for flexible divisions while training. 3 frames if not using flexible divisons (prev_prev, prev, cur and fut target)
            continue

        dataset_name = dataset_path.name

        with open(dataset_path.parent / (dataset_name + '_GT') / 'TRA' / 'man_track.txt') as f:
            track_file = []
            for line in f:
                line = line.split() # to deal with blank 
                if line:            # lines (ie skip them)
                    line = [int(i) for i in line]
                    track_file.append(line)
            track_file = np.stack(track_file)

        img_reader.load_track_file(track_file)
        img_reader.read_gts(fps)

        for counter,fp in enumerate(tqdm(fps)):
            
            fn = fp.name
            fn_orig = fn
            framenb = int(re.findall('\d+',fn)[-1])
            
            img = img_reader.read_image(fp)
            gt = img_reader.read_gt(fp,counter)

            cellnbs = np.unique(gt)
            cellnbs = cellnbs[cellnbs != 0]

            track_file = img_reader.track_file_orig

            framenb = int(re.findall('\d+',fp.name)[-1])
            man_track = track_file[(track_file[:,1] <= framenb) * (track_file[:,2] >= framenb)]

            np.array_equal(sorted(cellnbs),man_track[:,0])
        
            if len(cellnbs) == 0:
                cellnbs = [-1]

            for cellnb in cellnbs:
                    mask_sc = label(gt == cellnb)
                    annotation = utils.create_anno(gt,cellnb,image_id,annotation_id,dataset_name)
                    annotations.append(annotation)
                    annotation_id += 1 

            fn = f'CTC_{dataset_name}_frame_{framenb:03d}.tif'

            image = {
                'license': 1,
                'man_track_id': dataset_name,
                'file_name': fn,
                'height': img.shape[0],
                'width': img.shape[1],
                'id': image_id,
                'ctc_id': fp.parent.name,
                'frame_id': 0,
                'seq_length': 1,
                }    

            images.append(image)

            image_id += 1

            cv2.imwrite(str(datapath / folder / 'img' / fn),img)
            cv2.imwrite(str(datapath / folder / 'gt' / fn),gt)

        np.savetxt(datapath / 'man_track' / folder / (f'{dataset_name}.txt'),track_file,fmt='%d')

    metadata = {
        'annotations': annotations,
        'images': images,
        'categories': categories,
        'licenses': licenses,
        'info': info,
        'sequences': 'cells',
        'max_num_of_cells': img_reader.max_num_of_cells,
    }
        
    with open(datapath / 'annotations' / folder / (f'anno.json'), 'w') as f:
        json.dump(metadata,f, cls=utils.NpEncoder)

    print(f'Max number of cells in all frames is {img_reader.max_num_of_cells}')
    print(f'{skip:03d} folders skipped!')
                
