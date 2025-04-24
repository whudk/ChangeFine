import torch
import torch.nn.functional as F

from lib.vis.gradcam.utils import find_alexnet_layer, find_vgg_layer, find_resnet_layer, find_densenet_layer, find_squeezenet_layer,find_samclipcd_layer


# class GradCAM(object):
#     """Calculate GradCAM salinecy map.
#
#     A simple example:
#
#         # initialize a model, model_dict and gradcam
#         resnet = torchvision.models.resnet101(pretrained=True)
#         resnet.eval()
#         model_dict = dict(model_type='resnet', arch=resnet, layer_name='layer4', input_size=(224, 224))
#         gradcam = GradCAM(model_dict)
#
#         # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
#         img = load_img()
#         normed_img = normalizer(img)
#
#         # get a GradCAM saliency map on the class index 10.
#         mask, logit = gradcam(normed_img, class_idx=10)
#
#         # make heatmap from mask and synthesize saliency map using heatmap and img
#         heatmap, cam_result = visualize_cam(mask, img)
#
#
#     Args:
#         model_dict (dict): a dictionary that contains 'model_type', 'arch', layer_name', 'input_size'(optional) as keys.
#         verbose (bool): whether to print output size of the saliency map givien 'layer_name' and 'input_size' in model_dict.
#     """
#     def __init__(self, model_dict, verbose=False):
#         model_type = model_dict['type']
#         layer_name = model_dict['layer_name']
#         self.model_arch = model_dict['arch']
#         #self.model_pretrained = model_dict["pretrained"]
#
#
#         self.gradients = dict()
#         self.activations = dict()
#         def backward_hook(module, grad_input, grad_output):
#             self.gradients['value'] = grad_output[0]
#             return None
#         def forward_hook(module, input, output):
#             self.activations['value'] = output
#             return None
#
#         if 'vgg' in model_type.lower():
#             target_layer = find_vgg_layer(self.model_arch, layer_name)
#         elif 'resnet' in model_type.lower():
#             target_layer = find_resnet_layer(self.model_arch, layer_name)
#         elif 'densenet' in model_type.lower():
#             target_layer = find_densenet_layer(self.model_arch, layer_name)
#         elif 'alexnet' in model_type.lower():
#             target_layer = find_alexnet_layer(self.model_arch, layer_name)
#         elif 'squeezenet' in model_type.lower():
#             target_layer = find_squeezenet_layer(self.model_arch, layer_name)
#         elif 'shpcd' in model_type.lower():
#             target_layer = find_shpcdnet_layer(self.model_arch, layer_name)
#         elif 'samclipcd' in model_type.lower():
#             target_layer = find_samclipcd_layer(self.model_arch.module, layer_name)
#
#         target_layer.register_forward_hook(forward_hook)
#         target_layer.register_backward_hook(backward_hook)
#
#     #     if verbose:
#     #         try:
#     #             input_size = model_dict['input_size']
#     #         except KeyError:
#     #             print("please specify size of input image in model_dict. e.g. {'input_size':(224, 224)}")
#     #             pass
#     #         else:
#     #             device = 'cuda' if next(self.model_arch.parameters()).is_cuda else 'cpu'
#     #
#     #
#     #             inputs_left = torch.zeros(1, 3, *(input_size),device = device)
#     #             inputs_right =  torch.zeros(1, 3, *(input_size),device = device)
#     #             feats = self.model_pretrained.backbone.get_feats(inputs_right)
#     #             feats = [feat[:, 1:, :] for feat in feats]
#     #             self.model_arch(inputs_left,feats)
#     # #            self.model_arch(inputs_left)
#     #             print('saliency_map size :', self.activations['value'].shape[2:])
#
#
#     #def forward(self, input_left, input_right,class_idx=None, retain_graph=False):
#     def forward(self, output,class_idx=None, retain_graph=False):
#         """
#         Args:
#             logit: input image with shape of (1, 3, H, W)
#             class_idx (int): class index for calculating GradCAM.
#                     If not specified, the class index that makes the highest model prediction score will be used.
#         Return:
#             mask: saliency map of the same spatial dimension with input
#             logit: model output
#         """
#
#         b, c, h, w = output.size()
#         h, w = 256, 256        #logit = self.model_arch(input_left, input_right)["pred"]
#         # logit = self.model_arch(input)
#         # if class_idx is None:
#         #     score = logit[:, logit.max(1)[-1]].squeeze()
#         # else:
#         #     score = logit[:, class_idx].squeeze()
#
#         self.model_arch.zero_grad()
#         one_hot = torch.zeros(output.size()).to(output.device)
#         one_hot[0][class_idx] = 1
#         output.backward(gradient=one_hot,retain_graph = retain_graph)
#
#
#         gradients = self.gradients['value']
#         activations = self.activations['value']
#         b, k, u, v = gradients.size()
#
#         alpha = gradients.view(b, k, -1).mean(2)
#         #alpha = F.relu(gradients.view(b, k, -1)).mean(2)
#         weights = alpha.view(b, k, 1, 1)
#
#         saliency_map = (weights*activations).sum(1, keepdim=True)
#         saliency_map = F.relu(saliency_map)
#         saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
#         saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
#         saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data
#
#         return saliency_map
#
#     def __call__(self, input, class_idx=None, retain_graph=False):
#         return self.forward(input, class_idx, retain_graph)
#
#
# class GradCAMpp(GradCAM):
#     """Calculate GradCAM++ salinecy map.
#
#     A simple example:
#
#         # initialize a model, model_dict and gradcampp
#         resnet = torchvision.models.resnet101(pretrained=True)
#         resnet.eval()
#         model_dict = dict(model_type='resnet', arch=resnet, layer_name='layer4', input_size=(224, 224))
#         gradcampp = GradCAMpp(model_dict)
#
#         # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
#         img = load_img()
#         normed_img = normalizer(img)
#
#         # get a GradCAM saliency map on the class index 10.
#         mask, logit = gradcampp(normed_img, class_idx=10)
#
#         # make heatmap from mask and synthesize saliency map using heatmap and img
#         heatmap, cam_result = visualize_cam(mask, img)
#
#
#     Args:
#         model_dict (dict): a dictionary that contains 'model_type', 'arch', layer_name', 'input_size'(optional) as keys.
#         verbose (bool): whether to print output size of the saliency map givien 'layer_name' and 'input_size' in model_dict.
#     """
#     def __init__(self, model_dict, verbose=False):
#         super(GradCAMpp, self).__init__(model_dict, verbose)
#
#     def forward(self, output, class_idx=None, retain_graph=False):
#     #def forward(self, input, class_idx=None, retain_graph=False):
#         """
#         Args:
#             logit: output with shape of (b, c, H, W)
#             class_idx (int): class index for calculating GradCAM.
#                     If not specified, the class index that makes the highest model prediction score will be used.
#         Return:
#             mask: saliency map of the same spatial dimension with input
#             logit: model output
#         """
#         b, c, h, w = output.size()
#
#
#         # if class_idx is None:
#         #     score = logit[:, logit.max(1)[-1]].squeeze()
#         # else:
#         #     score = logit[:, class_idx].squeeze()
#
#         self.model_arch.zero_grad()
#
#         one_hot = torch.zeros(output.size()).to(output.device)
#         one_hot[0][class_idx] = 1
#         output.backward(gradient=one_hot,retain_graph = True)
#
#         gradients = self.gradients['value']  # [B, C, H, W]
#         activations = self.activations['value']
#
#             # 1. 计算 alpha 分子和分母
#         alpha_num = gradients.pow(2)
#
#         alpha_denom = gradients.pow(2) * 2.0 + \
#                       activations * gradients.pow(3)  # shape: [B, C, H, W]
#         alpha_denom = alpha_denom.sum(dim=(2, 3), keepdim=True)  # sum over spatial dims
#
#         # 避免除零
#         alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
#
#         # 2. 计算 alpha 权重
#         alpha = alpha_num / (alpha_denom + 1e-7)  # shape: [B, C, H, W]
#
#         # 3. 计算正梯度
#         positive_gradients = F.relu(gradients)
#
#         # 4. 通道加权和
#         weights = (alpha * positive_gradients).sum(dim=(2, 3), keepdim=True)  # shape: [B, C, 1, 1]
#
#         # 5. 生成 Grad-CAM++
#         saliency_map = (weights * activations).sum(dim=1, keepdim=True)  # shape: [B, 1, H, W]
#
#         return saliency_map, output



import torch
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)

    def generate(self):


        return self.compute_cam(self.gradients, self.activations)

    def compute_cam(self, gradients, activations):
        raise NotImplementedError("This should be implemented in the subclass.")


class GradCAMpp:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(self.save_activation)
                module.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self):
        grads = self.gradients  # [B, C, H, W]
        acts = self.activations  # [B, C, H, W]
        B, C, H, W = grads.size()

        grads_pow_2 = grads ** 2
        grads_pow_3 = grads ** 3
        sum_acts = torch.sum(acts.view(B, C, -1), dim=2).view(B, C, 1, 1)

        eps = 1e-6
        alpha = grads_pow_2 / (2 * grads_pow_2 + sum_acts * grads_pow_3 + eps)
        weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]

        cam = (weights * acts).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        cam = F.relu(cam)

        # 归一化
        cam = (cam - cam.min()) / (cam.max() - cam.min() + eps)

        return cam
