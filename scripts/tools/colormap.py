import numpy as np
import torch
def create_label_colormap():
    """Creates a label colormap used in CityScapes segmentation benchmark.
    Returns:
        A colormap for visualizing segmentation results.
    """
    colormap = torch.zeros((256, 3), dtype=torch.uint8)
    colormap[0] = torch.Tensor([0, 0, 0])
    colormap[1] = torch.Tensor([244, 35, 232])
    colormap[2] = torch.Tensor([70, 70, 70])
    colormap[3] = torch.Tensor([102, 102, 156])
    colormap[4] = torch.Tensor([190, 153, 153])
    colormap[5] = torch.Tensor([153, 153, 153])
    colormap[6] = torch.Tensor([250, 170, 30])
    colormap[7] = torch.Tensor([220, 220, 0])
    colormap[8] = torch.Tensor([107, 142, 35])
    colormap[9] = torch.Tensor([152, 251, 152])
    colormap[10] = torch.Tensor([70, 130, 180])
    colormap[11] = torch.Tensor([220, 20, 60])
    colormap[12] = torch.Tensor([255, 0, 0])
    colormap[13] = torch.Tensor([0, 0, 142])
    colormap[14] = torch.Tensor([0, 0, 70])
    colormap[15] = torch.Tensor([0, 60, 100])
    colormap[16] = torch.Tensor([0, 80, 100])
    colormap[17] = torch.Tensor([0, 0, 230])
    colormap[18] = torch.Tensor([119, 11, 32])

    return colormap


def colorize(mask, colormap,ignore_index = -1):
    bs,h,w = mask.shape
    color_mask = torch.zeros((bs,h,w,3),dtype=torch.long)
    mask[mask == ignore_index] = 0
    for i in torch.unique(mask):
        color_mask[mask == i] = colormap[i].long()

    return color_mask.permute(0,3,1,2)