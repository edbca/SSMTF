import models.archs.arch_util as arch_util
import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import PatchEmbed, PatchUnEmbed, ResidualGroup
from .blocks import DCN_sep_pre_multi_offset as DynAgg

class MamabaBlock(nn.Module):
    def __init__(self, num_features,layers,depths):
        super(MamabaBlock, self).__init__()

        self.patch_embed = PatchEmbed(embed_dim=num_features,norm_layer=nn.LayerNorm)
        
        self.patch_unembed = PatchUnEmbed(embed_dim=num_features,norm_layer=nn.LayerNorm)

        self.pos_drop = nn.Dropout(p=0.1)

        self.mamaba = nn.ModuleList([
            ResidualGroup(
                dim=num_features,
                depth=depths,#6,
                d_state = 16,
                mlp_ratio=2,
                norm_layer=nn.LayerNorm,
                use_checkpoint=False,
                resi_connection='1conv',
                is_light_sr = False
            )
            for i in range(layers)])#3
        self.norm = nn.LayerNorm(num_features)

    def forward(self, x):
        x_size = (x.shape[2], x.shape[3])#(48,48)
        x = self.patch_embed(x) # N,L,C #(8,64,48,48)-->(8,48*48,64)
        x = self.pos_drop(x) #(8,48*48,64)-->(8,48*48,64)
        for blk in self.mamaba:
            x = blk(x,x_size)
        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)
        return x


class ContentExtractor(nn.Module):

    def __init__(self, in_nc=3, out_nc=3, nf=64, n_blocks=16):
        super(ContentExtractor, self).__init__()

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.body = arch_util.make_layer(
            arch_util.ResidualBlockNoBN, n_blocks, nf=nf)

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
        self.content_extractor = ContentExtractor(
            in_nc=3, out_nc=3, nf=ngf, n_blocks=n_blocks)
        self.dyn_agg_restore = DynamicAggregationRestoration(
            ngf, n_blocks, groups)

        arch_util.srntt_init_weights(self, init_type='normal', init_gain=0.02)
        self.re_init_dcn_offset()

    def re_init_dcn_offset(self):
        self.dyn_agg_restore.small_dyn_agg.conv_offset_mask.weight.data.zero_()
        self.dyn_agg_restore.small_dyn_agg.conv_offset_mask.bias.data.zero_()
        self.dyn_agg_restore.medium_dyn_agg.conv_offset_mask.weight.data.zero_()
        self.dyn_agg_restore.medium_dyn_agg.conv_offset_mask.bias.data.zero_()
        self.dyn_agg_restore.large_dyn_agg.conv_offset_mask.weight.data.zero_()
        self.dyn_agg_restore.large_dyn_agg.conv_offset_mask.bias.data.zero_()

    def forward(self, x, pre_offset, img_ref_feat):
        """
        Args:
            x (Tensor): the input image of SRNTT.
            maps (dict[Tensor]): the swapped feature maps on relu3_1, relu2_1
                and relu1_1. depths of the maps are 256, 128 and 64
                respectively.
        """

        base = F.interpolate(x, None, 4, 'bilinear', False)
        content_feat = self.content_extractor(x)

        upscale_restore = self.dyn_agg_restore(content_feat, pre_offset,
                                               img_ref_feat)
        return upscale_restore + base


class DynamicAggregationRestoration(nn.Module):

    def __init__(self, ngf=64, n_blocks=16, groups=8):
        super(DynamicAggregationRestoration, self).__init__()

        # dynamic aggregation module for relu3_1 reference feature
        self.small_offset_conv1 = nn.Conv2d(ngf + 256, 256, 3, 1, 1, bias=True)  # concat for diff
        self.small_offset_conv2 = nn.Conv2d(256, 256, 3, 1, 1, bias=True)
        self.small_dyn_agg = DynAgg(
            256,
            256,
            3,
            stride=1,
            padding=1,
            dilation=1,
            deformable_groups=groups,
            extra_offset_mask=True)

        # for small scale restoration
        self.head_small = nn.Sequential(
            nn.Conv2d(ngf + 256, ngf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True))

        #self.body_small = arch_util.make_layer(arch_util.ResidualBlockNoBN, n_blocks, nf=ngf)
        self.body_small = MamabaBlock(ngf,3,6)##40,40

        self.tail_small = nn.Sequential(
            nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2), nn.LeakyReLU(0.1, True))

        # dynamic aggregation module for relu2_1 reference feature
        self.medium_offset_conv1 = nn.Conv2d(ngf + 128, 128, 3, 1, 1, bias=True)
        self.medium_offset_conv2 = nn.Conv2d(128, 128, 3, 1, 1, bias=True)
        self.medium_dyn_agg = DynAgg(
            128,
            128,
            3,
            stride=1,
            padding=1,
            dilation=1,
            deformable_groups=groups,
            extra_offset_mask=True)

        # for medium scale restoration
        self.head_medium = nn.Sequential(
            nn.Conv2d(ngf + 128, ngf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True))
        #self.body_medium = arch_util.make_layer(arch_util.ResidualBlockNoBN, n_blocks, nf=ngf)
        self.body_medium = MamabaBlock(ngf,3,6)##80,80

        self.tail_medium = nn.Sequential(
            nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2), nn.LeakyReLU(0.1, True))

        # dynamic aggregation module for relu1_1 reference feature
        self.large_offset_conv1 = nn.Conv2d(ngf + 64, 64, 3, 1, 1, bias=True)
        self.large_offset_conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.large_dyn_agg = DynAgg(
            64,
            64,
            3,
            stride=1,
            padding=1,
            dilation=1,
            deformable_groups=groups,
            extra_offset_mask=True)

        # for large scale
        self.head_large = nn.Sequential(
            nn.Conv2d(ngf + 64, ngf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True))
        #self.body_large = arch_util.make_layer(arch_util.ResidualBlockNoBN, n_blocks, nf=ngf)
        self.body_large = MamabaBlock(ngf,3,6)

        self.tail_large = nn.Sequential(
            nn.Conv2d(ngf, ngf // 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(ngf // 2, 3, kernel_size=3, stride=1, padding=1))

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def forward(self, x, pre_offset, img_ref_feat):
        #self.output = self.net_g(self.img_in_lq, self.pre_offset,self.img_ref_feat)
        #self.img_in_lq [b,64,40,40]
        #pre_offset['relu1_1'] = batch_offset_relu1 [b,9,160,160]
        #pre_offset['relu2_1'] = batch_offset_relu2 [b,9,80,80]
        #pre_offset['relu3_1'] = batch_offset_relu3 [b,9,40,40]

        # dynamic aggregation for relu3_1 reference feature #[b,64,40,40]+[b,256,40,40]
        relu3_offset = torch.cat([x, img_ref_feat['relu3_1']], 1)
        relu3_offset = self.lrelu(self.small_offset_conv1(relu3_offset)) #[b,54+256,40,40]-->[b,256,40,40]
        relu3_offset = self.lrelu(self.small_offset_conv2(relu3_offset)) #[b,256,40,40]-->[b,256,40,40]
        #([[b,256,40,40],[b,256,40,40]],[b,9,40,40])-->
        relu3_swapped_feat = self.lrelu(self.small_dyn_agg([img_ref_feat['relu3_1'], relu3_offset],pre_offset['relu3_1']))
        # small scale
        h = torch.cat([x, relu3_swapped_feat], 1)
        h = self.head_small(h)
        h = self.body_small(h) + x
        x = self.tail_small(h)

        # dynamic aggregation for relu2_1 reference feature
        relu2_offset = torch.cat([x, img_ref_feat['relu2_1']], 1)
        relu2_offset = self.lrelu(self.medium_offset_conv1(relu2_offset))
        relu2_offset = self.lrelu(self.medium_offset_conv2(relu2_offset))
        relu2_swapped_feat = self.lrelu(self.medium_dyn_agg([img_ref_feat['relu2_1'], relu2_offset],pre_offset['relu2_1']))
        # medium scale
        h = torch.cat([x, relu2_swapped_feat], 1)
        h = self.head_medium(h)
        h = self.body_medium(h) + x
        x = self.tail_medium(h)

        # dynamic aggregation for relu1_1 reference feature
        relu1_offset = torch.cat([x, img_ref_feat['relu1_1']], 1)
        relu1_offset = self.lrelu(self.large_offset_conv1(relu1_offset))
        relu1_offset = self.lrelu(self.large_offset_conv2(relu1_offset))
        relu1_swapped_feat = self.lrelu(self.large_dyn_agg([img_ref_feat['relu1_1'], relu1_offset],pre_offset['relu1_1']))
        #relu1_swapped_feat = self.pAda(x, relu1_swapped_feat)
        # large scale
        h = torch.cat([x, relu1_swapped_feat], 1)
        h = self.head_large(h)
        h = self.body_large(h) + x
        x = self.tail_large(h)

        return x
