import torch
import torch.nn as nn
import torch.nn.functional as F

from misc.quant_convs_bbcu import BinaryBlock, BinaryConv2d, BinaryUpConv2d
from basicsr.archs.arch_util import default_init_weights, make_layer
from basicsr.utils.registry import ARCH_REGISTRY
from misc.attention import SqueezeExcitationAttentionBlock

@ARCH_REGISTRY.register()
class BinEDSRSESE(nn.Module):

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4, img_range=1., rgb_mean=(0.4488, 0.4371, 0.4040)):
      super().__init__()
      self.upscale = upscale
      self.img_range = img_range

      self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
      self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
      self.body = make_layer(BinaryBlock, num_block, conv=BinaryConv2d,
                  n_feats=num_feat,
                  kernel_size=3,
                  bias=False,
                  bn=False,
                  resid_attn=SqueezeExcitationAttentionBlock(num_feat, 32),
                  bin_attn=SqueezeExcitationAttentionBlock(num_feat, 32))

      # upsampling
      self.upconv = BinaryUpConv2d(num_feat, num_feat * self.upscale * self.upscale, 3, False,upscale=upscale)
      
      self.conv_after_body = BinaryConv2d(num_feat, num_feat, 3, resid_attn=SqueezeExcitationAttentionBlock(num_feat, 32),
                  bin_attn=SqueezeExcitationAttentionBlock(num_feat, 32))
      self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

      # activation function
      self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

      # initialization
      default_init_weights([self.conv_first, self.upconv, self.conv_after_body, self.conv_last], 0.1)

    def forward(self, x):
      self.mean = self.mean.type_as(x)

      x = (x - self.mean) * self.img_range
      x = self.conv_first(x)

      res = self.conv_after_body(self.body(self.lrelu(x)))
      res += x

      out = self.conv_last(self.upconv(res))
      out = out / self.img_range + self.mean

      return out
