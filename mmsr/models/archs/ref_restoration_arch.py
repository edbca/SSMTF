import models.archs.arch_util as arch_util
import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import DCN_sep_pre_multi_offset as DynAgg


class PixelUnshuffle(nn.Module):
    def __init__(self, downscale_factor):
        super(PixelUnshuffle, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input):
        """
        input: (N, C, H, W)
        output: (N, C * r^2, H / r, W / r)
        """
        batch_size, channels, height, width = input.size()
        r = self.downscale_factor

        out_channels = channels * (r ** 2)
        out_height = height // r
        out_width = width // r

        input_view = input.contiguous().view(
            batch_size, channels, out_height, r, out_width, r)

        unshuffled_output = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
        return unshuffled_output.view(batch_size, out_channels, out_height, out_width)


class ContentExtractor(nn.Module):

    def __init__(self, in_nc=3, out_nc=3, nf=64, n_blocks=16):
        super(ContentExtractor, self).__init__()

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.body = arch_util.make_layer(arch_util.ResidualBlockNoBN, n_blocks, nf=nf)
        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # initialization
        arch_util.default_init_weights([self.conv_first], 0.1)

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        feat = self.body(feat)
        return feat

class RestorationNet(nn.Module):

    def __init__(self, ngf=64, n_blocks=16, groups=8):
        super(RestorationNet, self).__init__()
        self.content_extractor = ContentExtractor(in_nc=3, out_nc=3, nf=ngf, n_blocks=n_blocks)
        self.dyn_agg_restore = DynamicAggregationRestoration(ngf, n_blocks, groups)

        arch_util.srntt_init_weights(self, init_type='normal', init_gain=0.02)
        self.re_init_dcn_offset()

    def re_init_dcn_offset(self):
        self.dyn_agg_restore.small_dyn_agg.conv_offset_mask.weight.data.zero_()
        self.dyn_agg_restore.small_dyn_agg.conv_offset_mask.bias.data.zero_()
        self.dyn_agg_restore.medium_dyn_agg.conv_offset_mask.weight.data.zero_()
        self.dyn_agg_restore.medium_dyn_agg.conv_offset_mask.bias.data.zero_()
        self.dyn_agg_restore.large_dyn_agg.conv_offset_mask.weight.data.zero_()
        self.dyn_agg_restore.large_dyn_agg.conv_offset_mask.bias.data.zero_()

    def forward(self, x, pre_offset, img_ref_feat, mask):
        """
        Args:
            x (Tensor): the input image of SRNTT.
            maps (dict[Tensor]): the swapped feature maps on relu3_1, relu2_1
                and relu1_1. depths of the maps are 256, 128 and 64
                respectively.
        """

        base = F.interpolate(x, None, 4, 'bilinear', False)
        content_feat = self.content_extractor(x)

        upscale_restore = self.dyn_agg_restore(content_feat, pre_offset, img_ref_feat, mask)
        return upscale_restore + base


class DynamicAggregationRestoration(nn.Module):

    def __init__(self, ngf=64, n_blocks=16, groups=8):
        super(DynamicAggregationRestoration, self).__init__()

        # dynamic aggregation module for relu3_1 reference feature
        self.small_offset_conv1 = nn.Conv2d(ngf + 256, 256, 3, 1, 1, bias=True)  # concat for diff
        self.small_offset_conv2 = nn.Conv2d(256, 256, 3, 1, 1, bias=True)
        self.small_dyn_agg = DynAgg(256,256,3,stride=1,padding=1,dilation=1,deformable_groups=groups,extra_offset_mask=True)
        self.small_dyn_agg_down = DynAgg(256,256,3,stride=1,padding=1,dilation=1,deformable_groups=groups,extra_offset_mask=True)
        self.small_dyn_agg_up = DynAgg(256,256,3,stride=1,padding=1,dilation=1,deformable_groups=groups,extra_offset_mask=True)

        # for small scale restoration
        self.head_small = nn.Sequential(
            nn.Conv2d(ngf + 256, ngf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True))
        self.body_small = arch_util.make_layer(
            arch_util.ResidualBlockNoBN, n_blocks, nf=ngf)
        self.tail_small = nn.Sequential(
            nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2), nn.LeakyReLU(0.1, True))

        # dynamic aggregation module for relu2_1 reference feature
        self.medium_offset_conv1 = nn.Conv2d(
            ngf + 128, 128, 3, 1, 1, bias=True)
        self.medium_offset_conv2 = nn.Conv2d(128, 128, 3, 1, 1, bias=True)
        self.medium_dyn_agg = DynAgg(128,128,3,stride=1,padding=1,dilation=1,deformable_groups=groups,extra_offset_mask=True)
        self.medium_dyn_agg_down = DynAgg(128,128,3,stride=1,padding=1,dilation=1,deformable_groups=groups,extra_offset_mask=True)#
        self.medium_dyn_agg_up = DynAgg(128,128,3,stride=1,padding=1,dilation=1,deformable_groups=groups,extra_offset_mask=True)#

        # for medium scale restoration
        self.head_medium = nn.Sequential(
            nn.Conv2d(ngf + 128, ngf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True))
        self.body_medium = arch_util.make_layer(
            arch_util.ResidualBlockNoBN, n_blocks, nf=ngf)
        self.tail_medium = nn.Sequential(
            nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2), nn.LeakyReLU(0.1, True))

        # dynamic aggregation module for relu1_1 reference feature
        self.large_offset_conv1 = nn.Conv2d(ngf + 64, 64, 3, 1, 1, bias=True)
        self.large_offset_conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.large_dyn_agg = DynAgg(64,64,3,stride=1,padding=1,dilation=1,deformable_groups=groups,extra_offset_mask=True)
        self.large_dyn_agg_down = DynAgg(64,64,3,stride=1,padding=1,dilation=1,deformable_groups=groups,extra_offset_mask=True)
        self.large_dyn_agg_up = DynAgg(64,64,3,stride=1,padding=1,dilation=1,deformable_groups=groups,extra_offset_mask=True)

        # for large scale
        self.head_large = nn.Sequential(
            nn.Conv2d(ngf + 64, ngf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True))
        self.body_large = arch_util.make_layer(
            arch_util.ResidualBlockNoBN, n_blocks, nf=ngf)
        self.tail_large = nn.Sequential(
            nn.Conv2d(ngf, ngf // 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(ngf // 2, 3, kernel_size=3, stride=1, padding=1))

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, pre_offset, img_ref_feat, mask):
        # dynamic aggregation for relu3_1 reference feature
        _,_,h1,w1 =img_ref_feat['relu3_1'].size()
        relu3_offset = torch.cat([x, img_ref_feat['relu3_1']], 1)
        relu3_offset = self.lrelu(self.small_offset_conv1(relu3_offset))
        relu3_offset = self.lrelu(self.small_offset_conv2(relu3_offset))
        
        relu3_swapped_feat = self.lrelu(self.small_dyn_agg([img_ref_feat['relu3_1'], relu3_offset],pre_offset['relu3_1'])) #[9,256,40,40]
        # x1 same scale   
        img_ref_feat3_down = F.interpolate(img_ref_feat['relu3_1'], size=(h1//2, w1//2), mode='bilinear', align_corners=False)
        img_ref_feat3_down = F.pad(img_ref_feat3_down, (0, w1-w1//2, 0, h1-h1//2)) 
        relu3_swapped_feat_down = self.lrelu(self.small_dyn_agg_down([img_ref_feat3_down, relu3_offset],pre_offset['relu3_1_down'])) #[9,256,40,40]
        # x1 up scale
        img_ref_feat3_up = F.interpolate(img_ref_feat['relu3_1'], scale_factor=1.5, mode='bilinear', align_corners=False)
        _,_,n_h,n_w = img_ref_feat3_up.size()
        img_ref_feat3_up = img_ref_feat3_up[:,:,(n_h-h1)//2:(n_h-h1)//2 + h1, (n_w-w1)//2:(n_w-w1)//2 + w1]        
        relu3_swapped_feat_up = self.lrelu(self.small_dyn_agg_up([img_ref_feat3_up, relu3_offset], pre_offset['relu3_1_up'])) #[9,256,40,40]
        # x1 swapped_feat
        relu3_swapped_feat = relu3_swapped_feat*mask['relu3_1'][0].unsqueeze(0) + relu3_swapped_feat_down*mask['relu3_1'][1].unsqueeze(0) + relu3_swapped_feat_up*mask['relu3_1'][2].unsqueeze(0)

        h = torch.cat([x, relu3_swapped_feat], 1)
        h = self.head_small(h)
        h = self.body_small(h) + x
        x = self.tail_small(h)
        # dynamic aggregation for relu2_1 reference feature
        relu2_offset = torch.cat([x, img_ref_feat['relu2_1']], 1)
        relu2_offset = self.lrelu(self.medium_offset_conv1(relu2_offset))
        relu2_offset = self.lrelu(self.medium_offset_conv2(relu2_offset))
        # x2 same scale
        relu2_swapped_feat = self.lrelu(self.medium_dyn_agg([img_ref_feat['relu2_1'], relu2_offset],pre_offset['relu2_1']))
        # x2 down scale
        img_ref_feat2_down = F.interpolate(img_ref_feat['relu2_1'], size=(h1, w1), mode='bilinear', align_corners=False)
        img_ref_feat2_down = F.pad(img_ref_feat2_down, (0, w1*2-w1, 0, h1*2-h1)) 
        relu2_swapped_feat_down = self.lrelu(self.medium_dyn_agg_down([img_ref_feat2_down, relu2_offset],pre_offset['relu2_1_down'])) #[9,128,80,80]
        # x2 up scale
        img_ref_feat2_up = F.interpolate(img_ref_feat['relu2_1'], scale_factor=1.5, mode='bilinear', align_corners=False)
        _,_,n_h,n_w = img_ref_feat2_up.size()
        img_ref_feat2_up = img_ref_feat2_up[:,:,(n_h-h1*2)//2:(n_h-h1*2)//2 + h1*2, (n_w-w1*2)//2:(n_w-w1*2)//2 + w1*2]        
        relu2_swapped_feat_up = self.lrelu(self.medium_dyn_agg_up([img_ref_feat2_up, relu2_offset], pre_offset['relu2_1_up'])) #[9,256,40,40]
        # x2 swapped_feat
        relu2_swapped_feat = relu2_swapped_feat*mask['relu2_1'][0].unsqueeze(0) + relu2_swapped_feat_down*mask['relu2_1'][1].unsqueeze(0) + relu2_swapped_feat_up*mask['relu2_1'][2].unsqueeze(0)

        # medium scale
        h = torch.cat([x, relu2_swapped_feat], 1)
        h = self.head_medium(h)
        h = self.body_medium(h) + x
        x = self.tail_medium(h)

        # dynamic aggregation for relu1_1 reference feature
        relu1_offset = torch.cat([x, img_ref_feat['relu1_1']], 1)
        relu1_offset = self.lrelu(self.large_offset_conv1(relu1_offset))
        relu1_offset = self.lrelu(self.large_offset_conv2(relu1_offset))
        # x4 same scale        
        relu1_swapped_feat = self.lrelu(self.large_dyn_agg([img_ref_feat['relu1_1'], relu1_offset],pre_offset['relu1_1']))
        # x4 down scale
        img_ref_feat1_down = F.interpolate(img_ref_feat['relu1_1'], size=(h1*2, w1*2), mode='bilinear', align_corners=False)
        img_ref_feat1_down = F.pad(img_ref_feat1_down, (0, w1*4-w1*2, 0, h1*4-h1*2)) 
        relu1_swapped_feat_down = self.lrelu(self.large_dyn_agg_down([img_ref_feat1_down, relu1_offset],pre_offset['relu1_1_down'])) #[9,64,160,160]
        # x4 up scale
        img_ref_feat1_up = F.interpolate(img_ref_feat['relu1_1'], scale_factor=1.5, mode='bilinear', align_corners=False)
        _,_,n_h,n_w = img_ref_feat1_up.size()
        img_ref_feat1_up = img_ref_feat1_up[:,:,(n_h-h1*4)//2:(n_h-h1*4)//2 + h1*4, (n_w-w1*4)//2:(n_w-w1*4)//2 + w1*4]        
        relu1_swapped_feat_up = self.lrelu(self.large_dyn_agg_up([img_ref_feat1_up, relu1_offset], pre_offset['relu1_1_up'])) #[9,256,40,40]
        # x4 swapped_feat
        relu1_swapped_feat = relu1_swapped_feat*mask['relu1_1'][0].unsqueeze(0) + relu1_swapped_feat_down*mask['relu1_1'][1].unsqueeze(0) + relu1_swapped_feat_up*mask['relu1_1'][2].unsqueeze(0)

        # large scale
        h = torch.cat([x, relu1_swapped_feat], 1)
        h = self.head_large(h)
        h = self.body_large(h) + x
        x = self.tail_large(h)

        return x














