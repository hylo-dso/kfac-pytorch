import math
import torch

from kfac.layers.base import KFACLayer
from kfac.utils import try_contiguous

class Conv2dLayer(KFACLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_bias = self.module.bias is not None

    def get_gradient(self):
        grad = self.module.weight.grad.data.view(
                self.module.weight.grad.data.size(0), -1)  
        if self.module.bias is not None:
            grad = torch.cat([grad, module.bias.grad.data.view(-1, 1)], 1)
        return grad

    def get_diag_blocks(self, diag_blocks):
        return diag_blocks

    # TODO: refactor extract_params to not reuire x arg
    def _extract_patches(self, x):
        """Extract patches from convolutional layer
        Args:
          x: The input feature maps.  (batch_size, in_c, h, w)
    
        Returns:
          Tensor of shape (batch_size, out_h, out_w, in_c*kh*kw)
        """
        padding = self.module.padding
        kernel_size = self.module.kernel_size
        stride = self.module.stride
        if padding[0] + padding[1] > 0:
            x = torch.nn.functional.pad(x, 
                    (padding[1], padding[1], padding[0], padding[0])).data
        x = x.unfold(2, kernel_size[0], stride[0])
        x = x.unfold(3, kernel_size[1], stride[1])
        x = x.transpose_(1, 2).transpose_(2, 3).contiguous()
        x = x.view(
            x.size(0), x.size(1), x.size(2),
            x.size(3) * x.size(4) * x.size(5))
        return x

    def _compute_A(self):
        a = self.a_input
        batch_size = a.size(0)
        a = self._extract_patches(a)
        spatial_size = a.size(1) * a.size(2)
        a = a.view(-1, a.size(-1))
        if self.module.bias is not None:
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
        a = a/spatial_size
        return a.t() @ (a / batch_size)

    def _compute_G(self):
        # g: batch_size * n_filters * out_h * out_w
        # n_filters is actually the output dimension (analogous to Linear layer)
        g = self.g_output
        spatial_size = g.size(2) * g.size(3)
        batch_size = g.shape[0]
        g = g.transpose(1, 2).transpose(2, 3)
        g = try_contiguous(g)
        g = g.view(-1, g.size(-1))

        if self.batch_averaged:
            g = g * batch_size
        g = g * spatial_size

        return g.t() @ (g / g.size(0))
