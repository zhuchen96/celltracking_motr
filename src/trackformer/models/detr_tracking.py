from contextlib import nullcontext
import torch
import torch.nn as nn
import numpy as np

from ..util import box_ops
from ..util import misc as utils
from ..util.flex_div import update_early_or_late_track_divisions, update_object_detection
from .deformable_detr import DeformableDETR
from .deformable_transformer import DeformableTransformer
from .matcher import HungarianMatcher

class DETRTrackingBase(nn.Module):

    def __init__(self,
                 matcher: HungarianMatcher = None,
                 backprop_prev_frame=False,
                 dn_track = False,
                 dn_track_FPs=False,
                 dn_track_l1 = 0,
                 dn_track_l2 = 0,
                 dn_enc=False,
                 refine_div_track_queries = False,
                 no_data_aug=False,
                 flex_div=True,
                 use_prev_prev_frame=False,
                 num_queries = 30,
                 dn_track_group = False,
                 tgt_noise=1e-6):

        self._matcher = matcher
        self._backprop_prev_frame = backprop_prev_frame
        self.num_queries = num_queries
        self.dn_track = dn_track
        self.dn_track_FPs = dn_track_FPs
        self.dn_track_l1 = dn_track_l1
        self.dn_track_l2 = dn_track_l2
        self.dn_enc = dn_enc
        self.refine_div_track_queries = refine_div_track_queries
        self.no_data_aug = no_data_aug
        self.copy_dict_keys = ['labels','boxes','masks','track_ids','flexible_divisions','is_touching_edge','empty','framenb','labels_orig','boxes_orig','masks_orig','track_ids_orig','flexible_divisions_orig','is_touching_edge_orig']
        self.eval_prev_prev_frame = False
        self.dn_track_group = dn_track_group
        self.tgt_noise = tgt_noise

        self.use_prev_prev_frame = use_prev_prev_frame
        self.flex_div = flex_div

        if self.dn_track:
            self.dn_track_embedding = nn.Embedding(1,self.hidden_dim)

        self.train_model = False

    def train(self, mode: bool = True):
        """Sets the module in train mode."""
        return super().train(mode)

    def calc_num_FPs(self,targets,target_name):

        # Number of cells being tracked per batch
        num_cells = torch.tensor([len(target['main'][target_name]['prev_ind'][0]) for target in targets])
        num_cells_and_FPs = num_cells.clone()
        max_FPs = torch.ceil(torch.sqrt(num_cells)).long()

        for t in range(len(targets)):
            # 50% chance of adding a FP
            # Max 20% of number of cells in frame may be added
            if torch.rand(1) > 0.25: 
                num_cells_and_FPs[t] += torch.randint(0,max(max_FPs[t],1),(1,)).item()
                
        # Calculate the FPs necessary for batching to work. There needs to be an equal amount of track queries per batch so FPs are added to offset a sample with less cells
        max_track = max(num_cells_and_FPs)
        num_FPs = max_track - num_cells

        for t,target in enumerate(targets):
            target['main'][target_name]['num_FPs'] = num_FPs[t]
            target['main'][target_name]['max_FPs'] = max_FPs[t]

    def get_FP_boxes(self,target,target_name,i,prev_out):
        # Due to divisions, we do not want a repeat false positive which would occur with the current code below

        prev_out_ind_uni = torch.unique(target[target_name]['prev_ind'][0])
        prev_out_ind_uni_all =  torch.unique(target[target_name]['prev_ind_orig'][0]) # we do not want add FPs where FNs are. 

        not_prev_out_ind = torch.randperm(prev_out['pred_boxes'].shape[1])
        not_prev_out_ind = [ind.item() for ind in not_prev_out_ind if ind not in prev_out_ind_uni_all]

        random_false_out_ind = []
        FP_boxes = []
        FPs_hs = []

        for j in range(target[target_name]['num_FPs']):
            if j < len(prev_out_ind_uni) and j < target[target_name]['max_FPs'] and target_name == 'cur_target':
                
                box = prev_out['pred_boxes'][i,prev_out_ind_uni[j]].detach().clone()[:4]

                if target['training_method'] == 'main':
                    l_1, l_2 = 0.2, 0.1
                else:
                    l_1, l_2 = self.dn_track_l1, self.dn_track_l2
                box = box_ops.add_noise_to_boxes(box,l_1,l_2)
            else:
                box = torch.zeros((4)).to(self.device)
            
            if j < len(not_prev_out_ind): # Pick a random index that is not a TP
                random_false_out_ind.append(not_prev_out_ind[j])
            else:
                random_ind = torch.randint(0,len(not_prev_out_ind),(1,))[0]
                random_false_out_ind.append(int(not_prev_out_ind[random_ind]))

            if j < len(prev_out_ind_uni) and target['training_method'] == 'main' and box.sum() > 0 and torch.rand(1) > 0.5:
                FP_hs = prev_out['hs_embed'][i, prev_out_ind_uni[j]].detach().clone()
                FP_hs = torch.normal(0,0.25,size=FP_hs.shape,device=self.device) + FP_hs
                FPs_hs.append(FP_hs)
            else:
                FPs_hs.append(prev_out['hs_embed'][i,random_false_out_ind[-1]].detach().clone())

            FP_boxes.append(box)

        FP_boxes = torch.stack(FP_boxes)
        FPs_hs = torch.stack(FPs_hs)

        target[target_name]['prev_ind'][0] = torch.tensor(target[target_name]['prev_ind'][0].tolist() + random_false_out_ind).long()

        return FP_boxes, FPs_hs


    def update_target(self,target,target_name,index):

        cur_target = target[target_name]
        man_track = target['man_track']

        track_id = cur_target['track_ids'][index]
        track_id_1, track_id_2 = man_track[man_track[:,-1] == track_id,0]

        track_id_1_orig = cur_target['track_ids_orig'][cur_target['boxes'][index,:4].eq(cur_target['boxes_orig'][:,:4]).all(-1)][0]
        track_id_2_orig = cur_target['track_ids_orig'][cur_target['boxes'][index,4:].eq(cur_target['boxes_orig'][:,:4]).all(-1)][0]

        if track_id_1 != track_id_1_orig:
            track_id_1, track_id_2 = track_id_2, track_id_1

        cur_target['track_ids'] = torch.cat((cur_target['track_ids'][:index],torch.tensor([track_id_1]).to(self.device),torch.tensor([track_id_2]).to(self.device),cur_target['track_ids'][index+1:]))
        cur_target['boxes'] = torch.cat((cur_target['boxes'][:index],torch.cat((cur_target['boxes'][index,:4],torch.zeros(4,).to(self.device)))[None],torch.cat((cur_target['boxes'][index,4:],torch.zeros(4,).to(self.device)))[None],cur_target['boxes'][index+1:]))
        cur_target['labels'] = torch.cat((cur_target['labels'][:index],torch.tensor([0,1]).long().to(self.device)[None],torch.tensor([0,1]).long().to(self.device)[None],cur_target['labels'][index+1:]))
        cur_target['flexible_divisions'] = torch.cat((cur_target['flexible_divisions'][:index],torch.tensor([False]).to(self.device),cur_target['flexible_divisions'][index:]))
        edge_1 = cur_target['is_touching_edge_orig'][cur_target['track_ids_orig'] == track_id_1_orig]
        edge_2 = cur_target['is_touching_edge_orig'][cur_target['track_ids_orig'] == track_id_2_orig]
        cur_target['is_touching_edge'] = torch.cat((cur_target['is_touching_edge'][:index],edge_1,edge_2,cur_target['is_touching_edge'][index+1:]))

        assert cur_target['is_touching_edge'].shape[0] == cur_target['flexible_divisions'].shape[0] == cur_target['boxes'].shape[0] == cur_target['track_ids'].shape[0]
        
        if 'masks' in cur_target:
            N,_,H,W = cur_target['masks'].shape
            cur_target['masks'] = torch.cat((cur_target['masks'][:index],torch.cat((cur_target['masks'][index,:1],torch.zeros((1,H,W)).to(self.device,dtype=torch.uint8)))[None],torch.cat((cur_target['masks'][index,1:],torch.zeros((1,H,W)).to(self.device,dtype=torch.uint8)))[None],cur_target['masks'][index+1:]))

    def get_random_indices(self,targets, target_name,prev_indices):

        rand_num = torch.rand(1)

        # We need to access the number of divisions before we can add the track queries
        for target, prev_ind in zip(targets, prev_indices):
            prev_out_ind, prev_target_ind = prev_ind

            random_subset_mask_orig =  torch.randperm(len(prev_target_ind))

            if target['main'][target_name]['empty'] or target_name == 'prev_target':
                target['main'][target_name]['prev_ind'] = [prev_ind[0][random_subset_mask_orig],prev_ind[1][random_subset_mask_orig]]
                target['main'][target_name]['prev_ind_orig'] = [prev_ind[0][random_subset_mask_orig],prev_ind[1][random_subset_mask_orig]]
                target['main'][target_name]['random_subset_mask'] = random_subset_mask_orig
                target['main'][target_name]['random_subset_mask_orig'] = random_subset_mask_orig
                
                continue

            if self.no_data_aug:
                random_subset_mask = random_subset_mask_orig
            elif rand_num > 0.5:  # max number to be tracked since 
                random_subset_mask = random_subset_mask_orig[:len(prev_target_ind)]
            else:
                random_subset_mask = random_subset_mask_orig[:max(len(prev_target_ind) - torch.randint(0,max(len(prev_target_ind)//5,2),(1,)),1)]

            target['main'][target_name]['prev_ind'] = [prev_out_ind[random_subset_mask],prev_target_ind[random_subset_mask]]
            target['main'][target_name]['prev_ind_orig'] = [prev_out_ind[random_subset_mask_orig],prev_target_ind[random_subset_mask_orig]]
            target['main'][target_name]['random_subset_mask'] = random_subset_mask
            target['main'][target_name]['random_subset_mask_orig'] = random_subset_mask_orig

            if target_name == 'cur_target' and torch.tensor([target['main']['cur_target']['empty'] for target in targets]).sum() == 0 and torch.tensor([target['main']['prev_target']['empty'] for target in targets]).sum() == 0:

                if self.dn_track:
                    target['dn_track'] = {'cur_target': {}}
                    for copy_dict_key in self.copy_dict_keys: #  ['labels','boxes','masks','track_ids','flexible_divisions','is_touching_edge,'empty','framenb']
                        if copy_dict_key in target['main']['cur_target']:
                            target['dn_track']['cur_target'][copy_dict_key] = target['main']['cur_target'][copy_dict_key].clone()

                    target['dn_track']['prev_target'] = target['main']['prev_target'].copy()
                    if 'fut_target' in target['main']:
                        target['dn_track']['fut_target'] = target['main']['fut_target'].copy()
                    target['dn_track']['training_method'] = 'dn_track'

                    # Use only track queries that the main group is using so dn_track queries can also attend to all object queries
                    target['dn_track']['cur_target']['prev_ind'] = [prev_out_ind[random_subset_mask_orig],prev_target_ind[random_subset_mask_orig]]
                    target['dn_track']['cur_target']['prev_ind_orig'] = [prev_out_ind[random_subset_mask_orig],prev_target_ind[random_subset_mask_orig]]
                    target['dn_track']['man_track'] = target['main']['man_track'].clone()

                if self.dn_track_group:

                    target['dn_track_group'] = {'cur_target': {}}
                    for copy_dict_key in self.copy_dict_keys: #  ['labels','boxes','masks','track_ids','flexible_divisions','is_touching_edge,'empty','framenb']
                        if copy_dict_key in target['main']['cur_target']:
                            target['dn_track_group']['cur_target'][copy_dict_key] = target['main']['cur_target'][copy_dict_key].clone()

                    target['dn_track_group']['prev_target'] = target['main']['prev_target'].copy()
                    if 'fut_target' in target['main']:
                        target['dn_track_group']['fut_target'] = target['main']['fut_target'].copy()
                    target['dn_track_group']['training_method'] = 'dn_track_group'
                    
                    # Use only track queries that the main group is using so dn_track_group queries can also attend to all object queries
                    target['dn_track_group']['cur_target']['prev_ind'] = [prev_out_ind[random_subset_mask],prev_target_ind[random_subset_mask]]
                    target['dn_track_group']['cur_target']['prev_ind_orig'] = [prev_out_ind[random_subset_mask_orig],prev_target_ind[random_subset_mask_orig]]
                    target['dn_track_group']['cur_target']['random_subset_mask'] = random_subset_mask
                    target['dn_track_group']['man_track'] = target['main']['man_track'].clone()

    def separate_divided_cells(self,target,training_method,target_name,boxes=None,masks=None):
        
        for prev_ind_out in target[training_method][target_name]['prev_ind'][0]:
            if (target[training_method][target_name]['prev_ind'][0] == prev_ind_out).sum() == 2:
                inds = torch.where(target[training_method][target_name]['prev_ind'][0] == prev_ind_out)[0]
                orig_inds = target[training_method][target_name]['random_subset_mask'][inds]
                assert len(orig_inds) == 2 and (orig_inds[0] == orig_inds[1]+1 or orig_inds[0]+1 == orig_inds[1])
                ind = inds[torch.argmax(orig_inds)]
                # ind = inds[torch.argmax(target[training_method][target_name]['prev_ind'][1][inds])] # whichever box has the higher prev_ind for targets will be the division box
                if boxes is not None: # boxes
                    boxes[ind,:4] = boxes[ind,4:]
                else: # masks
                    masks[ind,0] = masks[ind,1]
            elif (target[training_method][target_name]['prev_ind'][0] == prev_ind_out).sum() == 1:
                pass
            else:
                NotImplementedError
        if boxes is not None:
            return boxes
        else:
            return masks

    def remove_new_cells(self,target,training_method,target_name,target_ind_match_matrix,prev_track_ids):

        new_cell_indices = sorted((target_ind_match_matrix.sum(0) == 0).nonzero(),reverse=True)
        man_track = target[training_method]['man_track']
        
        output_target = target[training_method][target_name]
        output_target['new_cell_ids'] = []
        framenb = output_target['framenb']

        for new_cell_ind in new_cell_indices:

            track_id = output_target['track_ids'][new_cell_ind]

            if track_id in man_track[:,-1]:
                dau_cells = man_track[man_track[:,-1] == track_id,0]

                if man_track[man_track[:,0]==dau_cells[0],1] == framenb:
                    output_target['new_cell_ids'].extend([dau_cells[0].item(),dau_cells[1].item()])
                    assert training_method == 'dn_track_group'
                else:
                    output_target['new_cell_ids'].append(output_target['track_ids'][new_cell_ind[0]].item())
            else:
                output_target['new_cell_ids'].append(output_target['track_ids'][new_cell_ind[0]].item())
                
            if new_cell_ind == output_target['boxes'].shape[0] - 1:
                output_target['boxes'] = output_target['boxes'][:new_cell_ind]
                output_target['labels'] = output_target['labels'][:new_cell_ind]
                output_target['track_ids'] = output_target['track_ids'][:new_cell_ind]
                output_target['flexible_divisions'] = output_target['flexible_divisions'][:new_cell_ind]
                output_target['is_touching_edge'] = output_target['is_touching_edge'][:new_cell_ind]

                if 'masks' in output_target:
                    output_target['masks'] = output_target['masks'][:new_cell_ind]

            else:
                output_target['boxes'] = torch.cat((output_target['boxes'][:new_cell_ind], output_target['boxes'][new_cell_ind+1:]),axis=0)
                output_target['labels'] = torch.cat((output_target['labels'][:new_cell_ind], output_target['labels'][new_cell_ind+1:]),axis=0)
                output_target['track_ids'] = torch.cat((output_target['track_ids'][:new_cell_ind], output_target['track_ids'][new_cell_ind+1:]),axis=0)
                output_target['flexible_divisions'] = torch.cat((output_target['flexible_divisions'][:new_cell_ind], output_target['flexible_divisions'][new_cell_ind+1:]),axis=0)
                output_target['is_touching_edge'] = torch.cat((output_target['is_touching_edge'][:new_cell_ind], output_target['is_touching_edge'][new_cell_ind+1:]),axis=0)

                if 'masks' in output_target:
                    output_target['masks'] = torch.cat((output_target['masks'][:new_cell_ind], output_target['masks'][new_cell_ind+1:]),axis=0)

        target_ind_match_matrix = prev_track_ids.unsqueeze(dim=1).eq(output_target['track_ids'])
        output_target['target_ind_matching'] = target_ind_match_matrix.any(dim=1)
        output_target['cells_leaving_mask'] = ~target_ind_match_matrix.any(dim=1)
        if training_method == 'main':
            output_target['cells_leaving_mask'] = torch.cat((output_target['cells_leaving_mask'],(torch.tensor([False, ] * output_target['num_FPs'])).bool().to(self.device)))
        output_target['track_query_match_ids'] = target_ind_match_matrix.nonzero()[:, 1]

        return target

    def add_track_queries_to_targets(self, targets, target_name, prev_target_name, prev_out):

        # Due to transformers needing the same number of track queries per batch (object queries is always the same), we need to add FPs to offset divisions or number of track queries
        # Only matters if we are using the prev prev frame when a division is tracked from prev prev frame to prev frame. This is because a single decoder output embedding can predict a cell division
        
        training_methods = ['main']

        if 'dn_track' in targets[0]:
            training_methods += ['dn_track']
            # Add in empty (zero boxes / embeddings) FPs if batch size is greater than 1 so the number of total queries are equal between batches
            num_cells = torch.tensor([len(target['dn_track'][target_name]['prev_ind'][0]) for target in targets])
            for t,target in enumerate(targets):
                target['dn_track'][target_name]['num_FPs'] = max(num_cells) - num_cells[t]

        if 'dn_track_group' in targets[0]:
            training_methods += ['dn_track_group']
            # Add in empty (zero boxes / embeddings) FPs if batch size is greater than 1 so the number of total queries are equal between batches
            num_cells = torch.tensor([len(target['dn_track_group'][target_name]['prev_ind'][0]) for target in targets])
            for t,target in enumerate(targets):
                target['dn_track_group'][target_name]['num_FPs'] = max(num_cells) - num_cells[t]

        # Get FPs for main training method only
        self.calc_num_FPs(targets,target_name)

        for i, target in enumerate(targets):

            # detected prev frame tracks
            track_ids = target['main'][prev_target_name]['track_ids']

            # If 'dn_track' or 'dn_track_group' then target_name == 'cur_target'
            if 'dn_track' in target:
                dn_track = target['dn_track']['cur_target']
                prev_track_ids = track_ids[dn_track['prev_ind'][1]]

                # match track ids between frames
                target_ind_match_matrix = prev_track_ids.unsqueeze(dim=1).eq(dn_track['track_ids'])
                dn_track['target_ind_matching'] = target_ind_match_matrix.any(dim=1)
                dn_track['cells_leaving_mask'] = ~target_ind_match_matrix.any(dim=1).bool().to(self.device)
                dn_track['track_query_match_ids'] = target_ind_match_matrix.nonzero()[:, 1]

                # For cells that enter the frame, we need to drop these cells during denoised training because no object queries are used. so the matcher will fail
                # Unless object queries are add to dn_track, we need to get rid of them
                target = self.remove_new_cells(target,'dn_track','cur_target',target_ind_match_matrix,prev_track_ids)

            if 'dn_track_group' in target:

                dn_track_group = target['dn_track_group']['cur_target']
                prev_track_ids = track_ids[dn_track_group['prev_ind'][1]]

                # match track ids between frames
                target_ind_match_matrix = prev_track_ids.unsqueeze(dim=1).eq(dn_track_group['track_ids'])
                dn_track_group['target_ind_matching'] = target_ind_match_matrix.any(dim=1)
                dn_track_group['cells_leaving_mask'] = ~target_ind_match_matrix.any(dim=1)
                dn_track_group['track_query_match_ids'] = target_ind_match_matrix.nonzero()[:, 1]

                # For cells that enter the frame, we need to drop these cells during denoised training because no object queries are used. so the matcher will fail
                # Unless object queries are add to dn_track_group, we need to get rid of them
                target = self.remove_new_cells(target,'dn_track_group','cur_target',target_ind_match_matrix,prev_track_ids)

            prev_track_ids = track_ids[target['main'][target_name]['prev_ind'][1]]

            # match track ids between frames
            target_ind_match_matrix = prev_track_ids.unsqueeze(dim=1).eq(target['main'][target_name]['track_ids'])
            target['main'][target_name]['target_ind_matching'] = target_ind_match_matrix.any(dim=1)
            target['main'][target_name]['track_query_match_ids'] = target_ind_match_matrix.nonzero()[:, 1]
            target_ind_not_matched_idx = (1 - target_ind_match_matrix.sum(dim=0)).nonzero()[:,0] # cells in cur_target that don't match to anything in prev_traget (could be FN or cell entering frame)

            # For images with no cells in them, we reformat target_ind_matching so torch.cat works properly with zero cells
            if target['main'][target_name]['target_ind_matching'].shape[0] == 0:
                target['main'][target_name]['target_ind_matching'] = torch.tensor([],device=self.device).bool()

            # If there is a FN for a cell in the previous frame that divides, then the labels/boxes/masks need to be adjusted for object detection to detect the divided cells separately
            if len(target_ind_not_matched_idx) > 0:
                count = 0
                for nidx in range(len(target_ind_not_matched_idx)):
                    target_ind_not_matched_i = target_ind_not_matched_idx[nidx] + count
                    if target['main'][target_name]['boxes'][target_ind_not_matched_i][-1] > 0:
                        self.update_target(target['main'],target_name,target_ind_not_matched_i)
                        count += 1

                # target['track_ids'] has change since FP divided cells have been separated into two boxes
                target_ind_match_matrix = prev_track_ids.unsqueeze(dim=1).eq(target['main'][target_name]['track_ids'])
                target['main'][target_name]['target_ind_matching'] = target_ind_match_matrix.any(dim=1)
                target['main'][target_name]['track_query_match_ids'] = target_ind_match_matrix.nonzero()[:, 1]

            # Division-ahead GT: 1 if the matched cur cell is dividing at this frame, 0 otherwise.
            # Supervised at the prev frame so the model learns pre-division morphology 1 frame early.
            match_ids = target['main'][target_name]['track_query_match_ids']
            if len(match_ids) > 0:
                target['main'][target_name]['track_query_div_ahead_gt'] = (
                    target['main'][target_name]['labels'][match_ids, 1] == 1
                ).float()
            else:
                target['main'][target_name]['track_query_div_ahead_gt'] = torch.tensor([], device=self.device)

            target['main'][target_name]['target_ind_matching'] = torch.cat([
                target['main'][target_name]['target_ind_matching'],
                torch.tensor([False, ] * target['main'][target_name]['num_FPs']).bool().to(self.device)
                ])

            target['main'][target_name]['track_queries_TP_mask'] = torch.cat([
                target['main'][target_name]['target_ind_matching'],
                torch.tensor([False, ] * (self.num_queries)).to(self.device)
                ]).bool()

            # track query masks
            track_queries_mask = torch.ones_like(target['main'][target_name]['target_ind_matching']).bool()
            track_queries_fal_pos_mask = torch.zeros_like(target['main'][target_name]['target_ind_matching']).bool()
            if len(target['main'][target_name]['target_ind_matching']) > 0:
                track_queries_fal_pos_mask[~target['main'][target_name]['target_ind_matching']] = True

            if 'dn_track' in target:

                dn_track['target_ind_matching'] = torch.cat([
                    dn_track['target_ind_matching'],
                    torch.tensor([False, ] * dn_track['num_FPs']).bool().to(self.device)
                    ])

                dn_track['track_queries_TP_mask'] = dn_track['target_ind_matching']
                dn_track['track_queries_mask'] = torch.ones_like(dn_track['target_ind_matching']).to(self.device).bool()
                dn_track['track_queries_fal_pos_mask'] = torch.zeros_like(dn_track['target_ind_matching']).to(self.device).bool()
                dn_track['track_queries_fal_pos_mask'][~dn_track['target_ind_matching']] = True      

                boxes = target['main']['prev_target']['boxes'][dn_track['prev_ind'][1],:4].clone()
                dn_track['track_query_boxes_gt'] = boxes.clone()

                l_1 = self.dn_track_l1
                l_2 = self.dn_track_l2

                boxes = box_ops.add_noise_to_boxes(boxes,l_1,l_2)

                if dn_track['num_FPs'] > 0:
                    FP_boxes = torch.zeros((dn_track['num_FPs'],4),device=self.device,dtype=boxes.dtype)
                    boxes = torch.cat((boxes,FP_boxes),axis=0)

                assert torch.sum(boxes < 0) == 0, 'Bboxes need to have positive values'

                dn_track['track_query_boxes'] = boxes
                dn_track['num_queries'] = len(dn_track['track_queries_mask'])
                dn_track['track_query_hs_embeds'] = self.dn_track_embedding.weight.repeat(len(boxes),1)

                assert torch.sum(dn_track['track_query_boxes'] < 0) == 0, 'Bboxes need to have positive values'
                
            if 'dn_track_group' in target:

                dn_track_group['target_ind_matching'] = torch.cat([
                    dn_track_group['target_ind_matching'],
                    torch.tensor([False, ] * dn_track_group['num_FPs']).bool().to(self.device)
                    ])
                
                dn_track_group['track_queries_TP_mask'] = dn_track_group['target_ind_matching']
                dn_track_group['track_queries_mask'] = torch.ones_like(dn_track_group['target_ind_matching']).to(self.device).bool()
                dn_track_group['track_queries_fal_pos_mask'] = torch.zeros_like(dn_track_group['target_ind_matching']).to(self.device).bool()
                dn_track_group['track_queries_fal_pos_mask'][~dn_track_group['target_ind_matching']] = True      

                dn_track_group_boxes = prev_out['pred_boxes'][i, dn_track_group['prev_ind'][0]].clone().detach()
                dn_track_group_hs = prev_out['hs_embed'][i, dn_track_group['prev_ind'][0]].clone()

                if self.last_frame_tracked == True:
                    dn_track_group_boxes = self.separate_divided_cells(target,'dn_track_group','cur_target',boxes = dn_track_group_boxes)

                dn_track_group_boxes = dn_track_group_boxes[:,:4]

                dn_track_group_hs += torch.normal(0,self.tgt_noise,size=dn_track_group_hs.shape,device=self.device)

                dn_track_group['num_queries'] = len(dn_track_group['track_queries_mask'])

                if dn_track_group['num_FPs'] > 0:
                    FP_boxes = torch.zeros((dn_track_group['num_FPs'],4),device=self.device,dtype=boxes.dtype)
                    dn_track_group_boxes = torch.cat((dn_track_group_boxes,FP_boxes),axis=0)
                    dn_track_group_hs = torch.cat((dn_track_group_hs,torch.zeros((dn_track_group['num_FPs'],dn_track_group_hs.shape[-1]),device=self.device)))

                dn_track_group['track_query_boxes_gt'] = dn_track_group_boxes
                dn_track_group_noised_boxes = box_ops.add_noise_to_boxes(dn_track_group_boxes.clone(),l_1,l_2)
                dn_track_group['track_query_boxes'] = dn_track_group_noised_boxes
                dn_track_group['track_query_hs_embeds'] = dn_track_group_hs

                assert dn_track_group_boxes.shape[0] == dn_track_group_hs.shape[0]

            prev_indices = target['main'][target_name]['prev_ind']
            
            boxes = prev_out['pred_boxes'][i, prev_indices[0]].detach().clone()
            hs = prev_out['hs_embed'][i, prev_indices[0]]

            if self.masks and self.init_boxes_from_masks and boxes.shape[0] > 0:

                masks = prev_out['pred_masks'][i, prev_indices[0]]

                if self.last_frame_tracked == True:
                    masks = self.separate_divided_cells(target,'main',target_name,masks=masks)

                masks = masks[:,0]

                masks_filt = torch.zeros((masks.shape),device=self.device)
                argmax = torch.argmax(masks,axis=0)
                
                for m in range(masks.shape[0]):
                    masks_filt[m,argmax==m] = masks[m,argmax==m]
                                   
                mask_boxes = box_ops.masks_to_boxes(masks_filt > 0,cxcywh=True) # bbox are 0 to 1 so we don't have to resize masks before getting bboxes
                boxes = mask_boxes
                assert torch.sum(boxes < 0) == 0, 'Bboxes need to have positive values'
            else:
                if self.last_frame_tracked == True:
                    boxes = self.separate_divided_cells(target,'main',target_name,boxes=boxes)
                boxes = boxes[:,:4]

            # Velocity-offset reference points: shift track query boxes by estimated
            # cell velocity (prev_gt_pos - prev_prev_gt_pos) so the reference points
            # start closer to where the cell will be in the current frame.
            if (target_name == 'cur_target' and boxes.shape[0] > 0 and
                    'prev_prev_target' in target['main'] and
                    not target['main']['prev_prev_target'].get('empty', False)):
                pprev_target = target['main']['prev_prev_target']
                if 'track_ids' in pprev_target and len(pprev_target['track_ids']) > 0:
                    pprev_ids = pprev_target['track_ids']
                    # prev_track_ids contains the GT track IDs for the N tracked cells
                    # prev_indices[1] indexes into prev_target GT boxes for those cells
                    gt_prev_centers = target['main'][prev_target_name]['boxes'][prev_indices[1], :2]  # [N, 2]
                    match_pprev = prev_track_ids.unsqueeze(1).eq(pprev_ids.unsqueeze(0))  # [N, M]
                    has_pprev = match_pprev.any(1)  # [N]
                    if has_pprev.any():
                        pprev_idx = match_pprev.float().argmax(1)  # [N]
                        gt_pprev_centers = pprev_target['boxes'][pprev_idx, :2]  # [N, 2]
                        vel = torch.zeros_like(gt_prev_centers)
                        vel[has_pprev] = gt_prev_centers[has_pprev] - gt_pprev_centers[has_pprev]
                        boxes[:, :2] = (boxes[:, :2] + vel).clamp(0.0, 1.0)

            num_FPs = target['main'][target_name]['num_FPs']
            if num_FPs > 0:
                FP_boxes, FP_hs = self.get_FP_boxes(target['main'],target_name,i,prev_out)
                boxes = torch.cat((boxes,FP_boxes))
                hs = torch.cat((hs,FP_hs))

            assert torch.sum(boxes < 0) == 0, 'Bboxes need to have positive values'
                    
            if boxes.shape[0] > 0:
                target['main'][target_name]['track_query_boxes'] = boxes
                target['main'][target_name]['track_query_hs_embeds'] = hs

                assert torch.sum(target['main'][target_name]['track_query_boxes'] < 0) == 0, 'Bboxes need to have positive values'
                assert target['main'][target_name]['track_query_hs_embeds'].shape[0] + self.num_queries == len(target['main'][target_name]['track_queries_TP_mask'])
                assert target['main'][target_name]['track_query_boxes'].shape[0] + self.num_queries == len(target['main'][target_name]['track_queries_TP_mask'])

            target['main'][target_name]['track_queries_mask'] = torch.cat([track_queries_mask, torch.tensor([False, ] * self.num_queries).to(self.device)]).bool()
            target['main'][target_name]['track_queries_fal_pos_mask'] = torch.cat([track_queries_fal_pos_mask, torch.tensor([False, ] * self.num_queries).to(self.device)]).bool()
            target['main'][target_name]['num_queries'] = len(track_queries_mask) + self.num_queries

    def forward(self, samples: utils.NestedTensor, targets: list = None):  

        if self.train_model:

            # Determines how many frames the model was train / backpropgate on at once
            track =  torch.rand(1) > 0.1

            if targets is None:
                raise NotImplementedError

            backprop_context = torch.no_grad
            if self._backprop_prev_frame:
                backprop_context = nullcontext

            if track:
                with backprop_context():

                    if  max([target['main']['cur_target']['boxes'].shape[0] for target in targets]) > self.num_queries: # checks if there are more objects in the image than there are object queries
                        raise NotImplementedError
                    
                    track_two_frames =  torch.rand(1) > 1/9

                    if track_two_frames:
                        # PREV PREV
                        self.last_frame_tracked = False

                        prev_prev_out, _, _, _, hs = super().forward([t['prev_prev_image'] for t in targets])

                        del _

                        if self.masks and self.init_boxes_from_masks:
                            prev_prev_out['pred_masks'] = self.forward_prediction_heads(hs[-1],self.final_mask_embed_index)

                        prev_prev_indices, targets = self._matcher(prev_prev_out, targets, 'main', 'prev_prev_target')

                        # if self.flex_div: Need to prev_prev_prev_target if to use this
                        #     targets, prev_prev_indices = update_object_detection(prev_prev_out,targets,prev_prev_indices,self.num_queries,'main','prev_prev_target','prev_target','cur_target') 

                        for t,target in enumerate(targets):
                            target['main']['prev_prev_target']['track_queries_mask'] = torch.zeros((self.num_queries)).bool().to(self.device)
                            target['main']['prev_prev_target']['indices'] = prev_prev_indices[t]

                        targets = utils.man_track_ids(targets,'main','prev_prev_target','prev_target')

                        self.get_random_indices(targets, 'prev_target', prev_prev_indices)
                        self.add_track_queries_to_targets(targets, 'prev_target', 'prev_prev_target', prev_prev_out)

                        # PREV
                        prev_out, _, _,_, hs = super().forward([t['prev_image'] for t in targets], targets, 'prev_target')
                        del _
                        
                        if self.masks and self.init_boxes_from_masks:
                            prev_out['pred_masks'] = self.forward_prediction_heads(hs[-1],self.final_mask_embed_index)

                        self.last_frame_tracked = True

                        prev_indices, targets = self._matcher(prev_out, targets, 'main', 'prev_target')

                        if self.flex_div:
                            targets = update_early_or_late_track_divisions(
                                prev_out,
                                targets,
                                'main',
                                'prev_prev_target',
                                'prev_target',
                                'cur_target',
                            )

                        for t,target in enumerate(targets):
                            target['main']['prev_target']['track_ids_track'] = target['main']['prev_target']['track_ids'].clone()

                        # When prev_prev frame is used, prev_out prediction make tracking predictions
                        # Predcitions with divisions have two cells, these need to be separate before predicting cur frame
                        # First step is to duplicate the indices for division boxes
                        # The function add_track_queries_to_targets will actually separate the boxes based on duplicated indexes
                        new_prev_indices = []
                        for target, (prev_ind_out, prev_ind_tgt) in zip(targets,prev_indices):

                            if target['main']['prev_target']['empty']:
                                new_prev_indices.append((prev_ind_out,prev_ind_tgt))
                            else:
                                boxes = target['main']['prev_target']['boxes'][prev_ind_tgt]
                                boxes_orig = target['main']['prev_target']['boxes_orig']

                                new_prev_ind_tgt = torch.ones((len(boxes_orig)),dtype=torch.int64) * -1
                                new_prev_ind_out = torch.ones((len(boxes_orig)),dtype=torch.int64) * -1

                                count = 0
                                for b,box in enumerate(boxes):
                                    if box[-1] > 0:
                                        new_prev_ind_out[b+count] = prev_ind_out[b]
                                        new_prev_ind_out[b+count+1] = prev_ind_out[b]
                                        new_ind_1 = boxes_orig[:,:4].eq(box[:4]).all(-1).nonzero()[0][0]
                                        new_ind_2 = boxes_orig[:,:4].eq(box[4:]).all(-1).nonzero()[0][0]
                                        new_prev_ind_tgt[b+count] = new_ind_1
                                        new_prev_ind_tgt[b+count+1] = new_ind_2
                                        count += 1
                                    else:
                                        new_ind = boxes_orig[:,:4].eq(box[:4]).all(-1).nonzero()[0][0]
                                        new_prev_ind_tgt[b+count] = new_ind         
                                        new_prev_ind_out[b+count] = prev_ind_out[b]  

                                assert -1 not in new_prev_ind_out and -1 not in new_prev_ind_tgt

                                new_prev_indices.append((new_prev_ind_out,new_prev_ind_tgt))

                        prev_indices = new_prev_indices

                        for t,target in enumerate(targets):
                            target['main']['prev_target']['updated_indices'] = prev_indices[t]

                    else:
                        prev_prev_out = None
                        prev_out, _, _, _, hs = super().forward([t['prev_image'] for t in targets])

                        del _

                        self.last_frame_tracked = False

                        if self.masks and self.init_boxes_from_masks:
                            prev_out['pred_masks'] = self.forward_prediction_heads(hs[-1],self.final_mask_embed_index)

                        prev_indices, targets = self._matcher(prev_out, targets, 'main', 'prev_target')

                        if self.flex_div:
                            targets, prev_indices = update_object_detection(prev_out,targets,prev_indices,self.num_queries,'main','prev_prev_target','prev_target','cur_target') 

                        for t,target in enumerate(targets):
                            target['main']['prev_target']['track_queries_mask'] = torch.zeros((self.num_queries)).bool().to(self.device)
                            target['main']['prev_target']['indices'] = prev_indices[t]

                    targets = utils.man_track_ids(targets,'main','prev_target','cur_target')

                    self.get_random_indices(targets, 'cur_target', prev_indices)
                    self.add_track_queries_to_targets(targets, 'cur_target', 'prev_target', prev_out)

                    for target in targets:
                        target['track'] = True

            else:
                prev_prev_out = None
                prev_out = None

                for target in targets:
                    target['main']['cur_target']['track_queries_mask'] = torch.zeros((self.num_queries)).bool().to(self.device)
                    target['main']['cur_target']['track_queries_TP_mask'] = torch.zeros((self.num_queries)).bool().to(self.device)
                    target['main']['cur_target']['num_queries'] = self.num_queries
                    target['track'] = False

        out, targets, features, memory, hs  = super().forward(samples, targets, 'cur_target')

        if self.train_model:               
            out['prev_outputs'] = prev_out
            out['prev_prev_outputs'] = prev_prev_out

        return out, targets, features, memory, hs
    


class DeformableDETRTracking(DETRTrackingBase, DeformableDETR, DeformableTransformer):
    def __init__(self, tracking_kwargs, detr_kwargs, transformer_kwargs):
        DeformableTransformer.__init__(self, **transformer_kwargs)
        DeformableDETR.__init__(self, **detr_kwargs)
        DETRTrackingBase.__init__(self, **tracking_kwargs)
