import os
import json
import torch
import data_utils
from torchvision.transforms import Resize
import torch.nn.functional as F

class SoftmaxPooling2D(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(SoftmaxPooling2D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x):
        N, C, H, W = x.shape
        h, w = self.kernel_size

        # Unfold the input tensor to extract sliding patches
        patches = F.unfold(x, kernel_size=(h, w), stride=self.stride, padding=self.padding)
        
        # Reshape to (N, C, h*w, num_patches)
        patches = patches.view(N, C, h * w, -1)

        # Apply softmax over the spatial dimensions (h*w)
        softmax_weights = F.softmax(patches, dim=2)
        
        # Compute weighted average
        weighted_patches = patches * softmax_weights
        pooled = weighted_patches.sum(dim=2)

        # Reshape back to (N, C, H', W')
        H_out = (H + 2 * self.padding - h) // self.stride[0] + 1
        W_out = (W + 2 * self.padding - w) // self.stride[1] + 1
        pooled = pooled.view(N, C, H_out, W_out)

        return pooled


class CBM_model(torch.nn.Module):
    def __init__(self, backbone_name, W_c, W_g, b_g, proj_mean, proj_std, device="cuda", map_size=(12,12)):
        super().__init__()
        model, _ = data_utils.get_target_model(backbone_name, device)
        self.backbone = torch.nn.Sequential(*list(model.resnet.children())[:-1])
        self.proj_layer = torch.nn.Conv2d(in_channels=W_c.shape[1], out_channels=W_c.shape[0], kernel_size=1,
                                            bias=False).to(device)
        self.map_size = map_size
        self.proj_layer.load_state_dict({"weight": W_c})
        self.pool_layer = SoftmaxPooling2D(kernel_size=map_size)
        self.resize_layer = Resize(map_size)
        self.proj_mean = proj_mean
        self.proj_std = proj_std
        self.final = torch.nn.Linear(in_features=W_g.shape[1], out_features=W_g.shape[0]).to(device)
        self.final.load_state_dict({"weight": W_g, "bias": b_g})
        self.concepts = None

    def forward(self, x, intervention_masks=None):
        assert x.dim() == 4, f"Expected 4D input, got {x.dim()}D"
        x = self.backbone(x).last_hidden_state
        if not self.map_size[0] == x.shape[2] or not self.map_size[1] == x.shape[3]:
            x = self.resize_layer(x)
        x = self.proj_layer(x)
        proj_map = x
        if intervention_masks is not None:
            assert intervention_masks.shape == proj_map.shape, "intervention masks shape does not match feature maps."
            proj_map += intervention_masks
            x += intervention_masks
        x = torch.squeeze(self.pool_layer(x))
        proj_c = (x - self.proj_mean) / self.proj_std
        x = self.final(proj_c)
        return x, proj_c, proj_map


def load_cbm(load_dir, backbone, device):

    W_c = torch.load(os.path.join(load_dir ,"W_c.pt"), map_location=device)
    W_g = torch.load(os.path.join(load_dir, "W_g.pt"), map_location=device)
    b_g = torch.load(os.path.join(load_dir, "b_g.pt"), map_location=device)

    proj_mean = torch.load(os.path.join(load_dir, "proj_mean.pt"), map_location=device)
    proj_std = torch.load(os.path.join(load_dir, "proj_std.pt"), map_location=device)

    model = CBM_model(backbone, W_c, W_g, b_g, proj_mean, proj_std, device)
    return model
