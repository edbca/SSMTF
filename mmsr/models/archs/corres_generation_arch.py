import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.archs.arch_util import tensor_shift
from models.archs.ref_map_util import feature_match_index
from models.archs.vgg_arch import VGGFeatureExtractor

logger = logging.getLogger('base')


class CorrespondenceGenerationArch(nn.Module):

    def __init__(self,
                 patch_size=3,
                 stride=1,
                 vgg_layer_list=['relu3_1', 'relu2_1', 'relu1_1'],
                 vgg_type='vgg19'):
        super(CorrespondenceGenerationArch, self).__init__()
        self.patch_size = patch_size
        self.stride = stride

        self.vgg_layer_list = vgg_layer_list
        self.vgg = VGGFeatureExtractor(layer_name_list=vgg_layer_list, vgg_type=vgg_type)

    def index_to_flow(self, max_idx):
        device = max_idx.device
        # max_idx to flow
        h, w = max_idx.size()
        flow_w = max_idx % w
        flow_h = max_idx // w

        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, h).to(device),
            torch.arange(0, w).to(device))
        grid = torch.stack((grid_x, grid_y), 2).unsqueeze(0).float().to(device)
        grid.requires_grad = False
        flow = torch.stack((flow_w, flow_h),
                           dim=2).unsqueeze(0).float().to(device)
        flow = flow - grid  # shape:(1, w, h, 2)
        flow = torch.nn.functional.pad(flow, (0, 0, 0, 2, 0, 2))

        return flow

    def forward(self, dense_features, img_ref_hr):
        batch_offset_relu3 = []
        batch_offset_relu3_down =[]
        batch_offset_relu3_up =[]
        mask3=[]
        batch_offset_relu2 = []
        batch_offset_relu2_down =[]
        batch_offset_relu2_up =[]
        mask2=[]
        batch_offset_relu1 = []
        batch_offset_relu1_down =[]
        batch_offset_relu1_up =[]
        mask1=[]

        for ind in range(img_ref_hr.size(0)):
            feat_in = dense_features['dense_features1'][ind]
            feat_ref = dense_features['dense_features2'][ind]
            c, h, w = feat_ref.size()
            feat_ref_down = F.interpolate(feat_ref.unsqueeze(0), size=(h//2,w//2), mode='bilinear', align_corners=False).squeeze(0) #1
            feat_ref_down = F.pad(feat_ref_down, (0, w-w//2, 0, h-h//2))#1
            
            feat_ref_up = F.interpolate(feat_ref.unsqueeze(0), scale_factor=1.5, mode='bilinear', align_corners=False).squeeze(0) #1
            _, n_h, n_w = feat_ref_up.size()
            feat_ref_up = feat_ref_up[ :, (n_h-h)//2:(n_h-h)//2 + h, (n_w-w)//2:(n_w-w)//2 + w]


            feat_in = F.normalize(feat_in.reshape(c, -1), dim=0).view(c, h, w)
            feat_ref = F.normalize(feat_ref.reshape(c, -1), dim=0).view(c, h, w)
            feat_ref_down = F.normalize(feat_ref_down.reshape(c, -1), dim=0).view(c, h, w) #1
            feat_ref_up = F.normalize(feat_ref_up.reshape(c, -1), dim=0).view(c, h, w) #1

            ###########
            #same scale
            _max_idx, _max_val = feature_match_index(
                feat_in,
                feat_ref,
                patch_size=self.patch_size,
                input_stride=self.stride,
                ref_stride=self.stride,
                is_norm=True,
                norm_input=True)
            # offset map for relu3_1
            offset_relu3 = self.index_to_flow(_max_idx)
            # shift offset relu3
            shifted_offset_relu3 = []
            for i in range(0, 3):
                for j in range(0, 3):
                    flow_shift = tensor_shift(offset_relu3, (i, j))
                    shifted_offset_relu3.append(flow_shift)
            shifted_offset_relu3 = torch.cat(shifted_offset_relu3, dim=0)
            batch_offset_relu3.append(shifted_offset_relu3)
            #val shift  
            val_re3 = torch.nn.functional.pad(_max_val, (0, 2, 0, 2))
            val_relu3 = [] #1
            for i in range(0, 3): #1
                for j in range(0, 3): #1
                    val_shift = tensor_shift(val_re3, (i, j)).unsqueeze(0) #1
                    val_relu3.append(val_shift) #1
            val_relu3 = torch.mean(torch.cat(val_relu3, dim=0),dim=0) #1 [9,40,40]-->[40,40],mean

            ###########
            #down scale
            _max_idx_down, _max_val_down = feature_match_index(
                feat_in,
                feat_ref_down,
                patch_size=self.patch_size,
                input_stride=self.stride,
                ref_stride=self.stride,
                is_norm=True,
                norm_input=True)
            # offset map for relu3_1
            offset_relu3_down = self.index_to_flow(_max_idx_down)
            # shift offset relu3
            shifted_offset_relu3_down = []
            for i in range(0, 3):
                for j in range(0, 3):
                    flow_shift_down = tensor_shift(offset_relu3_down, (i, j))
                    shifted_offset_relu3_down.append(flow_shift_down)
            shifted_offset_relu3_down = torch.cat(shifted_offset_relu3_down, dim=0)
            batch_offset_relu3_down.append(shifted_offset_relu3_down)
            #val shift  
            val_re3_down = torch.nn.functional.pad(_max_val_down, ( 0, 2, 0, 2))
            val_relu3_down = [] #1
            for i in range(0, 3): #1
                for j in range(0, 3): #1
                    val_shift_down = tensor_shift(val_re3_down, (i, j)).unsqueeze(0) #1
                    val_relu3_down.append(val_shift_down) #1
            val_relu3_down = torch.mean(torch.cat(val_relu3_down, dim=0),dim=0) #1 [9,40,40]-->[40,40],mean

            ###########
            #up scale
            _max_idx_up, _max_val_up = feature_match_index(
                feat_in,
                feat_ref_up,
                patch_size=self.patch_size,
                input_stride=self.stride,
                ref_stride=self.stride,
                is_norm=True,
                norm_input=True)
            # offset map for relu3_1
            offset_relu3_up = self.index_to_flow(_max_idx_up)
            # shift offset relu3
            shifted_offset_relu3_up = []
            for i in range(0, 3):
                for j in range(0, 3):
                    flow_shift_up = tensor_shift(offset_relu3_up, (i, j))
                    shifted_offset_relu3_up.append(flow_shift_up)
            shifted_offset_relu3_up = torch.cat(shifted_offset_relu3_up, dim=0)
            batch_offset_relu3_up.append(shifted_offset_relu3_up)
            #val shift  
            val_re3_up = torch.nn.functional.pad(_max_val_up, ( 0, 2, 0, 2))
            val_relu3_up = [] #1
            for i in range(0, 3): #1
                for j in range(0, 3): #1
                    val_shift_up = tensor_shift(val_re3_up, (i, j)).unsqueeze(0) #1
                    val_relu3_up.append(val_shift_up) #1
            val_relu3_up = torch.mean(torch.cat(val_relu3_up, dim=0),dim=0) #1 [9,40,40]-->[40,40],mean

            mask_3 =torch.stack([val_relu3, val_relu3_down, val_relu3_up],dim=0) #1
            _, indices = torch.max(mask_3, dim=0) #1
            mask3_same = (indices==0).float() #1
            mask2_same = torch.repeat_interleave(mask3_same,2,0)
            mask2_same = torch.repeat_interleave(mask2_same,2,1)
            mask1_same = torch.repeat_interleave(mask3_same,4,0)
            mask1_same = torch.repeat_interleave(mask1_same,4,1)
            mask3_down = (indices==1).float() #1 
            mask2_down = torch.repeat_interleave(mask3_down,2,0)
            mask2_down = torch.repeat_interleave(mask2_down,2,1)
            mask1_down = torch.repeat_interleave(mask3_down,4,0)
            mask1_down = torch.repeat_interleave(mask1_down,4,1)
            mask3_up = (indices==2).float() #1 
            mask2_up = torch.repeat_interleave(mask3_up,2,0)
            mask2_up = torch.repeat_interleave(mask2_up,2,1)
            mask1_up = torch.repeat_interleave(mask3_up,4,0)
            mask1_up = torch.repeat_interleave(mask1_up,4,1)            

            mask3.append(mask3_same.unsqueeze(0)) #1
            mask3.append(mask3_down.unsqueeze(0)) #1 
            mask3.append(mask3_up.unsqueeze(0)) #1 
            mask2.append(mask2_same.unsqueeze(0)) #1
            mask2.append(mask2_down.unsqueeze(0)) #1 
            mask2.append(mask2_up.unsqueeze(0)) #1 
            mask1.append(mask1_same.unsqueeze(0)) #1
            mask1.append(mask1_down.unsqueeze(0)) #1 
            mask1.append(mask1_up.unsqueeze(0)) #1 

            # offset map for relu2_1_same
            offset_relu2 = torch.repeat_interleave(offset_relu3, 2, 1)
            offset_relu2 = torch.repeat_interleave(offset_relu2, 2, 2)
            offset_relu2 *= 2
            # shift offset relu2_1_same
            shifted_offset_relu2 = []
            for i in range(0, 3):
                for j in range(0, 3):
                    flow_shift = tensor_shift(offset_relu2, (i * 2, j * 2))
                    shifted_offset_relu2.append(flow_shift)
            shifted_offset_relu2 = torch.cat(shifted_offset_relu2, dim=0)
            batch_offset_relu2.append(shifted_offset_relu2)

            # offset map for relu2_1_down
            offset_relu2_down = torch.repeat_interleave(offset_relu3_down, 2, 1)#1
            offset_relu2_down = torch.repeat_interleave(offset_relu2_down, 2, 2)#1
            offset_relu2_down *= 2 #1
            # shift offset relu2_1_down
            shifted_offset_relu2_down = []#1
            for i in range(0, 3):#1
                for j in range(0, 3):#1
                    flow_shift = tensor_shift(offset_relu2_down, (i * 2, j * 2))#1
                    shifted_offset_relu2_down.append(flow_shift)#1
            shifted_offset_relu2_down = torch.cat(shifted_offset_relu2_down, dim=0)#1
            batch_offset_relu2_down.append(shifted_offset_relu2_down)#1

            # offset map for relu2_1_up
            offset_relu2_up = torch.repeat_interleave(offset_relu3_up, 2, 1)#1
            offset_relu2_up = torch.repeat_interleave(offset_relu2_up, 2, 2)#1
            offset_relu2_up *= 2 #1
            # shift offset relu2_1_up
            shifted_offset_relu2_up = []#1
            for i in range(0, 3):#1
                for j in range(0, 3):#1
                    flow_shift = tensor_shift(offset_relu2_up, (i * 2, j * 2))#1
                    shifted_offset_relu2_up.append(flow_shift)#1
            shifted_offset_relu2_up = torch.cat(shifted_offset_relu2_up, dim=0)#1
            batch_offset_relu2_up.append(shifted_offset_relu2_up)#1


            # offset map for relu1_1_same
            offset_relu1 = torch.repeat_interleave(offset_relu3, 4, 1)
            offset_relu1 = torch.repeat_interleave(offset_relu1, 4, 2)
            offset_relu1 *= 4
            # shift offset relu1_1_same
            shifted_offset_relu1 = []
            for i in range(0, 3):
                for j in range(0, 3):
                    flow_shift = tensor_shift(offset_relu1, (i * 4, j * 4))
                    shifted_offset_relu1.append(flow_shift)
            shifted_offset_relu1 = torch.cat(shifted_offset_relu1, dim=0)
            batch_offset_relu1.append(shifted_offset_relu1)

            # offset map for relu1_1_down
            offset_relu1_down = torch.repeat_interleave(offset_relu3_down, 4, 1)#1
            offset_relu1_down = torch.repeat_interleave(offset_relu1_down, 4, 2)#1
            offset_relu1_down *= 4 #1
            # shift offset relu1_1_down
            shifted_offset_relu1_down = []#1
            for i in range(0, 3):#1
                for j in range(0, 3):#1
                    flow_shift = tensor_shift(offset_relu1_down, (i * 2, j * 2))#1
                    shifted_offset_relu1_down.append(flow_shift)#1
            shifted_offset_relu1_down = torch.cat(shifted_offset_relu1_down, dim=0)#1
            batch_offset_relu1_down.append(shifted_offset_relu1_down)#1

            # offset map for relu1_1_up
            offset_relu1_up = torch.repeat_interleave(offset_relu3_up, 4, 1)#1
            offset_relu1_up = torch.repeat_interleave(offset_relu1_up, 4, 2)#1
            offset_relu1_up *= 4 #1
            # shift offset relu1_1_up
            shifted_offset_relu1_up = []#1
            for i in range(0, 3):#1
                for j in range(0, 3):#1
                    flow_shift = tensor_shift(offset_relu1_up, (i * 2, j * 2))#1
                    shifted_offset_relu1_up.append(flow_shift)#1
            shifted_offset_relu1_up = torch.cat(shifted_offset_relu1_up, dim=0)#1
            batch_offset_relu1_up.append(shifted_offset_relu1_up)#1

        # size: [b, 9, h, w, 2], the order of the last dim: [x, y]
        batch_offset_relu3 = torch.stack(batch_offset_relu3, dim=0)
        batch_offset_relu2 = torch.stack(batch_offset_relu2, dim=0)
        batch_offset_relu1 = torch.stack(batch_offset_relu1, dim=0)

        batch_offset_relu3_down = torch.stack(batch_offset_relu3_down, dim=0)
        batch_offset_relu2_down = torch.stack(batch_offset_relu2_down, dim=0)
        batch_offset_relu1_down = torch.stack(batch_offset_relu1_down, dim=0)

        batch_offset_relu3_up = torch.stack(batch_offset_relu3_up, dim=0)
        batch_offset_relu2_up = torch.stack(batch_offset_relu2_up, dim=0)
        batch_offset_relu1_up = torch.stack(batch_offset_relu1_up, dim=0)

        pre_offset = {}
        pre_offset['relu1_1'] = batch_offset_relu1
        pre_offset['relu1_1_down'] = batch_offset_relu1_down
        pre_offset['relu1_1_up'] = batch_offset_relu1_up      
        pre_offset['relu2_1'] = batch_offset_relu2
        pre_offset['relu2_1_down'] = batch_offset_relu2_down
        pre_offset['relu2_1_up'] = batch_offset_relu2_up
        pre_offset['relu3_1'] = batch_offset_relu3
        pre_offset['relu3_1_down'] = batch_offset_relu3_down
        pre_offset['relu3_1_up'] = batch_offset_relu3_up

        mask = {}
        mask['relu1_1'] = mask1
        mask['relu2_1'] = mask2
        mask['relu3_1'] = mask3

        img_ref_feat = self.vgg(img_ref_hr)

        return pre_offset, img_ref_feat, mask




