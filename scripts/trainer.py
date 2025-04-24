from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils.tools.logger import Logger as Log
import random

import matplotlib.pyplot as plt
import torch.nn as nn
from utils.distributed import get_world_size, get_rank, is_distributed
import torch
import time
import os
import torch.nn.functional as F
import torch.distributed as dist

from utils.tools.average_meter import AverageMeter

import torchvision.models as mod
from models import clip
from models.sam.build_sam import build_sam_vit_b,build_sam_vit_l,build_sam_vit_b
from lib.loss.loss_manager import LossManager
from lib.vis.seg_visualizer import SegVisualizer
from dataset.data_loader import DataLoader
from scripts.tools.optim_scheduler import OptimScheduler


from scripts.tools.module_runner import ModuleRunner
from scripts.tools.evaluator import get_evaluator

from torch.utils.tensorboard import SummaryWriter
#classnames = ["background","Buildings"]
import numpy as np
import cv2
import multiprocessing
num_cores = multiprocessing.cpu_count()  # 获取 CPU 核心数
torch.set_num_threads(1)



class Trainer(object):

    def __init__(self, configer):
        self.configer = configer
        self.batch_time = AverageMeter()
        self.foward_time = AverageMeter()
        self.backward_time = AverageMeter()
        self.loss_time = AverageMeter()
        self.data_time = AverageMeter()
        self.train_losses = AverageMeter()
        self.val_losses = AverageMeter()
        self.loss_manager = LossManager(configer)
        self.module_runner = ModuleRunner(configer)
        self.data_loader = DataLoader(configer)
        self.optim_scheduler = OptimScheduler(configer)
        self.evaluator = get_evaluator(configer, self)
        self.seg_visualizer = SegVisualizer(configer)
        self.train_loader = None
        self.optimizer = None
        self.scheduler = None
        self.running_score = None
        self.visualizer = self.configer.get("eval","visualizer")
        self.__init_trainer()
        self.tb_logger = SummaryWriter(log_dir=self.configer.get("logging", "tb_name"))  # 指定日志存储路径

    def build_clip(self):
        clip_model, _ = clip.load("./pretrained/clip_checkpoints/ViT-B-16.pt", "cuda")


        return  clip_model.eval()

    def build_sam(self):
        # load sam
        sam = build_sam_vit_b(checkpoint=r"./pretrained/sam_checkpoints/sam_vit_b.pth").eval()
        return sam



    def conduct_classnames(self):
        xml_data = self.configer.get("data", "classnames")
        from xml.dom.minidom import parse
        dom_tree = parse(xml_data)


        type_elements = dom_tree.getElementsByTagName('type')
        types = []

        for type_element in type_elements:

            types.append(type_element.firstChild.data.strip())
        return types



    def __init_trainer(self):

        from models.ChangeFine import ChangeFine
        #from models.SamClip_v0 import SamClipCD

        self.use_fp16 = self.configer.get("train", "fp16")
        Log.info("use fp16={} for trainning".format(self.configer.get("train", "fp16")))
        Log.info("Trainning Classes = {}".format(self.configer.get("data","num_classes")))

        clip_model = self.build_clip()
        sam_model = self.build_sam()
        class_names = self.conduct_classnames()

        model_kwargs = self.configer.get("network","params")


        self.model = ChangeFine(
            self.configer,
            clip_model=clip_model,
            sam_model= sam_model,
            class_names= class_names,
            **model_kwargs
        )


        self.model = self.module_runner.load_net(self.model)

        for name, params in self.model.named_parameters():
            if params.type() != 'torch.cuda.FloatTensor':
                print(name, params.type())

        json_file = self.configer.get("data","traintxt")
        self.train_loader = self.data_loader.build_loader(json_file)



        self.scheduler = self.optim_scheduler.cosine_scheduler(
            base_value=self.configer.get('lr', 'base_lr'),
            final_value=1e-8,
            epochs=self.configer.get("train", "epoch"),
            niter_per_ep=len(self.train_loader) // self.configer.get("train", "batch_size"),
            warmup_epochs=self.configer.get("train", "warm_up_epoch")
        )

        self.neg_loader = None
        if self.configer.exists("data", "neg") :
            neg_json = self.configer.get("data", "neg")["json"]
            if os.path.isfile(neg_json):
                self.neg_loader = self.data_loader.build_loader(neg_json, split="neg")

        Log.info("Training batchs: {}".format(len(self.train_loader)))
        self.val_loader = self.data_loader.build_loader(self.configer.get("data","valtxt"), split="val")

        #Log.info('Params Group Method: {}'.format(self.configer.get('optim', 'group_method')))

        # if self.configer.get('optim', 'group_method') == 'decay':
        #     params_group = self.group_weight(self.model)
        # else:
        #     assert self.configer.get('optim', 'group_method') is None
        params_group = self._get_parameters(self.model)

        self.optimizer, _ = self.optim_scheduler.init_optimizer(params_group)

        self.scheduler = self.optim_scheduler.cosine_scheduler(
            base_value=self.configer.get('lr', 'base_lr'),
            final_value=1e-8,
            epochs=self.configer.get("train", "epoch"),
            niter_per_ep=len(self.train_loader),
            warmup_epochs=self.configer.get("train", "warm_up_epoch")
        )

        self.with_contrast = True if self.configer.exists("contrast") else False
        if self.configer.exists("contrast", "warmup_iters"):
            self.contrast_warmup_iters = self.configer.get("contrast", "warmup_iters")
        else:
            self.contrast_warmup_iters = 0

        self.loss = self.loss_manager.build_loss()

        self.loss = self.module_runner.to_device(self.loss)
        for param in clip_model.parameters():
            param.requires_grad = False


        self.excluded_param_names = []
        if dist.is_initialized():
            for name, param in self.model.module.named_parameters():
                #print(name, param.requires_grad)
                if param.requires_grad is False:
                    self.excluded_param_names.append(name)
        else:
            for name, param in self.model.named_parameters():
                #print(name, param.requires_grad)
                if param.requires_grad is False:
                    self.excluded_param_names.append(name)
        # for name, param in self.model.named_parameters():
        #     # print(name, param.requires_grad)
        #     if param.requires_grad is True:
        #         print(name, param.requires_grad)

        #排除decoder的
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        Log.info("Model Parameters:{}".format(trainable_params))
        # input_tensor = torch.randn(1, 768)
        # flops, params = profile(model, inputs=(input_tensor,), verbose=False)

    @staticmethod
    def group_weight(module):
        group_decay = []
        group_no_decay = []
        for m in module.modules():
            if isinstance(m, nn.Linear):
                group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            elif isinstance(m, nn.modules.conv._ConvNd):
                group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            else:
                if hasattr(m, 'weight'):
                    group_no_decay.append(m.weight)
                if hasattr(m, 'bias'):
                    group_no_decay.append(m.bias)

        assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
        groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
        return groups

    def _get_parameters(self, model):
        regularized = []
        not_regularized = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # we do not regularize biases nor Norm parameters
            if name.endswith(".bias") or len(param.shape) == 1:
                not_regularized.append(param)
            else:
                regularized.append(param)
        return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]

    def fill_holes(self, mask):
        """
        Fill holes in a binary mask using the flood fill algorithm.

        Parameters:
        - mask: A tensor of shape [B, H, W] with values 0 (background) and 1 (foreground).

        Returns:
        - mask_filled: The mask with holes filled, with values strictly 0 or 1.
        """
        # Clone the mask to ensure we don't modify the original tensor
        mask_filled = mask.clone()
        B, H, W = mask.shape
        mask_np = mask_filled.cpu().numpy()

        # Apply flood fill on each mask in the batch
        for b in range(B):
            # Convert the mask to 8-bit (for OpenCV, 0 = background, 255 = foreground)
            mask_b = (mask_np[b] * 255).astype(np.uint8)

            # If the mask is completely black (no foreground), no need to fill
            if np.sum(mask_b) == 0:
                mask_np[b] = mask_b
                continue

            # Add a 1-pixel border to the mask for flood fill compatibility
            mask_in = np.zeros((H + 2, W + 2), np.uint8)
            mask_in[1:-1, 1:-1] = mask_b

            # Perform flood fill starting from the top-left corner (0, 0) in the padded mask
            cv2.floodFill(mask_in, None, (0, 0), 255)

            # Invert the flood-filled mask to get the background that was filled
            im_floodfill_inv = cv2.bitwise_not(mask_in[1:-1, 1:-1])  # Remove the padding

            # Combine the filled areas with the original foreground
            mask_np[b] = im_floodfill_inv | mask_b

        # Ensure the result is strictly binary (0 or 1)
        mask_np[mask_np > 0] = 1

        # Convert the mask back to a tensor and return
        return torch.from_numpy(mask_np).long().to(mask.device)

    # def FillHole(imgPath, SavePath):
    #     im_in = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE);
    #     cv2.imwrite("im_in.png", im_in)
    #     # 复制 im_in 图像
    #     im_floodfill = im_in.copy()
    #
    #     # Mask 用于 floodFill，官方要求长宽+2
    #     h, w = im_in.shape[:2]
    #     mask = np.zeros((h + 2, w + 2), np.uint8)
    #
    #     # floodFill函数中的seedPoint对应像素必须是背景
    #     isbreak = False
    #     for i in range(im_floodfill.shape[0]):
    #         for j in range(im_floodfill.shape[1]):
    #             if (im_floodfill[i][j] == 0):
    #                 seedPoint = (i, j)
    #                 isbreak = True
    #                 break
    #         if (isbreak):
    #             break
    #
    #     # 得到im_floodfill 255填充非孔洞值
    #     cv2.floodFill(im_floodfill, mask, seedPoint, 255)
    #
    #     # 得到im_floodfill的逆im_floodfill_inv
    #     im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    #
    #     # 把im_in、im_floodfill_inv这两幅图像结合起来得到前景
    #     im_out = im_in | im_floodfill_inv
    #
    def __val(self,data_loader=None, vis_dir = "", save = True):
        """
          Validation function during the train phase.
        """
        """
                 Validation function during the train phase.
               """

        import torch.distributed as dist
        self.model.eval()
        self.loss.eval()



        data_loader = self.val_loader if data_loader is None else data_loader


        if data_loader is None:
            Log.info("val_loader is None,skip validation")
            return
        # visdir = os.path.join("./val", self.configer.get("network",  "image_encoder")['name'])
        # if not os.path.exists(visdir):
        #     os.makedirs(visdir)




        for id,(oldimage, newimage, targets, img_records) in enumerate(self.val_loader):

            if id % 10 == 0 and get_rank() == 0:
                Log.info('{} images processed\n'.format(id))
            start_time = time.time()

            oldimage = oldimage.cuda()
            newimage = newimage.cuda()
            targets = targets.cuda()


            # Calculate features
            with torch.no_grad():
                res = self.model(oldimage.float().cuda(), newimage.float().cuda(), batched_input=img_records)
                #preds, pred_aux = self.model(oldimage.float().cuda(), newimage.float().cuda(), img_records = img_records)
                preds = res["pred"]
                #show preds with softmax
                # Apply softmax to the predictions (along the class dimension, dim=1)
                preds_softmax = torch.softmax(preds, dim=1)[:,1,:,:].unsqueeze(1) # Shape: B x 2 x H x W

                loss = self.loss(self.model, res, targets, False)["loss"]

                pred_aux = F.interpolate(res["pred_aux"],(targets.shape[1],targets.shape[2]),mode='bilinear',align_corners=False)
                preds = F.interpolate(preds, (targets.shape[1], targets.shape[2]), mode='bilinear',
                                     align_corners=False)

                #output = preds * 1.0   + pred_aux * 0.4
                if preds.shape[1] == 1:
                    pos_prob = torch.sigmoid(preds)
                    neg_prob = 1 - pos_prob
                    preds = torch.cat([neg_prob, pos_prob], dim=1)

                #output = torch.argmax(preds,dim=1)
                output = torch.argmax(preds, dim=1)

                # preds: [B, 2, H, W]
                #output = preds_softmax[:, 0, :, :]  # 第1通道表示变化概率

                # 二值化（阈值 0.5 或其他）
                #output = ((output >= 0.05).float() * 1).long()  # [B, H, W]，值为 0 或 255

                bs, h, w = output.shape
                batch_size = oldimage.shape[0]
                #Log.info(get_rank())
                if  get_rank() == 0:

                    if self.visualizer is True:
                        self.seg_visualizer.vis_cd(preds_chg=output, targets=targets, ori_left_image=oldimage,
                                                   ori_right_image=newimage, names=img_records, sub_dir=os.path.join(vis_dir,"fine"),
                                                   draw_contours=True)

                        corse = torch.argmax(pred_aux, dim=1)
                        self.seg_visualizer.vis_cd(preds_chg=corse, targets=targets, ori_left_image=oldimage,
                                                   ori_right_image=newimage, names=img_records, sub_dir=os.path.join(vis_dir,"coarse"),
                                                   draw_contours=True, pallete_target=True)
                        #
                        # pred_prob = torch.softmax(preds, dim=1)[:,1,:,:]
                        # self.seg_visualizer.vis_cd(preds_chg=pred_prob, targets=targets, ori_left_image=oldimage,
                        #                            ori_right_image=newimage, names=img_records, sub_dir=os.path.join(vis_dir,"chgprob"),
                        #                            draw_contours=False, pallete_target = False)
                        #show gradcam



                #print(loss)
                if not is_distributed() or get_rank() == 0:
                    self.val_losses.update(loss, batch_size)
                    self.evaluator.update_score_seg(output, targets)


            self.batch_time.update(time.time() - start_time)
            start_time = time.time()


        if not is_distributed() or get_rank() == 0:
            self.evaluator.update_performance()
            self.evaluator.update_acc()
            self.evaluator.print_scores()
        self.configer.update(['val_loss'], self.val_losses.avg)

        if save:
            if is_distributed():
                self.module_runner.save_net(self.model.module, save_mode='performance',
                                        excluded_param_names=self.excluded_param_names)
            else:
                self.module_runner.save_net(self.model, save_mode='performance', excluded_param_names = self.excluded_param_names)
            #self.module_runner.save_net(self.model, save_mode='val_loss')

        # Print the log info & reset the states.
        if not is_distributed() or get_rank() == 0:
            Log.info(
                'Test Time {batch_time.sum:.3f}s, ({batch_time.avg:.3f})\t'
                'Loss {loss.avg:.8f}\n'.format(
                    batch_time=self.batch_time, loss=self.val_losses))
            #
            # self.tb_logger.add_scalar("mIOU", self.configer.get("performance"), self.configer.get("iters"))
            # self.tb_logger.add_scalar("acc", self.configer.get("accuracy"), self.configer.get("iters"))
            self.tb_logger.add_scalars("eval", {"mIOU": self.configer.get("performance")}, self.configer.get("iters"))
            self.tb_logger.add_scalars("acc", {"acc": self.configer.get("accuracy")}, self.configer.get("iters"))

            self.evaluator.print_scores()
        #self.evaluator.print_scores()
        self.batch_time.reset()
        self.val_losses.reset()
        self.evaluator.reset()
        self.model.train()

        # for name , params in self.model.named_parameters():
        #     if params.requires_grad is False:
        #         print(params)
        self.loss.train()


    def _vis_gradcam(self, vis_dir):
        from lib.vis.gradcam.gradcam import GradCAMpp, GradCAM

        self.model.eval()

        cam_generator = GradCAMpp(self.model.module, target_layer="embed_feats")  # ResNet最后一层卷积层


        for id, (oldimage, newimage, targets, img_records) in enumerate(self.val_loader):

            if id % 10 == 0 and get_rank() == 0:
                Log.info('{} images processed\n'.format(id))
            start_time = time.time()

            oldimage = oldimage.cuda()
            newimage = newimage.cuda()
            targets = targets.cuda()
            # forward pass
            outputs = self.model(oldimage, newimage, batched_input=img_records)
            preds = outputs["pred"]  # [B, C, H, W]

            # 可视化的类别索引（例如变化类为1，不变为0）
            class_idx = 1  # 你可以换成其他类别

            # 使用 class_idx 的平均响应作为 loss，用于生成 gradcam
            class_response = preds[:, class_idx, :, :]  # [B, H, W]
            loss = class_response.mean()

            self.model.zero_grad()
            loss.backward(retain_graph=True)

            # 生成 Grad-CAM++
            cam_map = cam_generator.generate()  # shape: [B, 1, H, W]

            # 可视化
            if get_rank() == 0:
                self.seg_visualizer.vis_gradcampp(
                    cam_map, names=img_records, sub_dir=os.path.join(vis_dir,"gradcam")
                )



        Log.info("Done.")
        return


    def _vis_attention_map(self, data_loader=None):

        import torch.distributed as dist


        start_time = time.time()
        replicas = self.evaluator.prepare_validaton()

        data_loader = self.val_loader if data_loader is None else data_loader
        visdir = os.path.join("./val", self.configer.get("network", "image_encoder")['name'])
        if not os.path.exists(visdir):
            os.makedirs(visdir)

        def denorm(image):
            mean = torch.Tensor(self.configer.get("normalize", 'mean')).cuda()
            std = torch.Tensor(self.configer.get("normalize", 'std')).cuda()
            image = image * std[None, :, None, None] + mean[None, :, None, None]
            return image
        for j, data_dict in enumerate(data_loader):
            if j % 10 == 0:
                Log.info('{} images processed\n'.format(j))

            inputs = data_dict["img"].cuda()
            targets = data_dict["labelmap"].cuda()
            img_metas = data_dict["img_metas"]
            b, c, ori_h, ori_w = inputs.shape
            w_featmap = inputs.shape[-2] // 16
            h_featmap = inputs.shape[-1] // 16


            nh = h_featmap * w_featmap
            import matplotlib.pyplot as plt
            num_heads = 3
            with torch.no_grad():

                attention_map = self.model.get_last_selfattention(inputs)

                # 创建子图来显示每个头的注意力图
                fig, axs = plt.subplots(1, num_heads, figsize=(16, 4))
                fig.suptitle(f'Attention Maps', fontsize=16)
                # 选择要可视化的注意力头数和样本索引

                for sample_idx in range(inputs.shape[0]):
                    # 遍历每个头并可视化注意力图
                    for head in range(num_heads):
                        head_attention = attention_map[sample_idx, head,:,:].view(nh, nh).cpu().numpy()
                        axs[head].imshow(head_attention, cmap='hot', interpolation='nearest')
                        axs[head].set_title(f'Head {head + 1}')
                        axs[head].axis('off')
                plt.show()

                # sample_index = 0
                # # 获取指定样本的注意力图
                # sample_attention = attention_map[sample_index, :num_heads, :, :]
                #
                # # 创建网格图像
                # grid_image = torchvision.utils.make_grid(sample_attention, nrow=num_heads, normalize=True)
                #
                # # 转换为NumPy数组并调整形状
                # grid_image = grid_image.permute(1, 2, 0).cpu().numpy()
                #
                # # 可视化热力图
                # plt.imshow(grid_image, cmap='hot', interpolation='nearest')
                # plt.colorbar()
                # plt.title('Attention Map - Sample {}'.format(sample_index))
                # plt.show()



    def __train(self):

        self.model.train()
        self.loss.train()
        if self.use_fp16:
            from torch.cuda.amp import autocast, GradScaler
            scaler = torch.cuda.amp.GradScaler()
        if self.neg_loader is not None:
            iter_neg = iter(self.neg_loader)
            neg_id = 0
            neg_epoch = 0


        with_contrast = False


        for id,(oldimg, newimg, target, img_records) in enumerate(self.train_loader):


            self.module_runner.get_lr_from_scheduler(self.optimizer, self.scheduler)
            start_time = time.time()
            self.data_time.update(time.time() - start_time)
            foward_start_time = time.time()
            if self.configer.exists("train","exchange"):
                    mask = torch.rand(oldimg.shape[0]) < self.configer.get("train","exchange")  # 生成 shape=[B] 的随机值，大于 0.5 为 True
                    oldimg[mask], newimg[mask] = newimg[mask], oldimg[mask]
            oldimg = oldimg.cuda()
            newimg = newimg.cuda()


            targets = target.cuda()
            loss_start_time = time.time()
            #if self.use_fp16:
            with autocast(enabled= self.use_fp16):
                res = self.model(oldimg, newimg, targets, batched_input = img_records)
                self.foward_time.update(time.time() - foward_start_time)
                if self.configer.get('epoch') >= self.configer.get("train", "epoch") *0.8 and self.configer.get('contrast', 'use'):
                    with_contrast = True
                loss = self.loss(self.model, res, targets, with_contrast)["loss"]


            batch_size = oldimg.shape[0]
            #ground_truth = torch.arange(batch_size, dtype=torch.long, device=oldimage.device)



            backward_loss = display_loss = loss
            self.train_losses.update(display_loss.item())
            self.loss_time.update(time.time() - loss_start_time)
            self.optimizer.zero_grad()
            # use amp
            backward_start_time = time.time()
            if self.use_fp16 :
                #scaler.scale(loss).backward()
                scaler.scale(backward_loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:

                backward_loss.backward()
                self.optimizer.step()
            # continue
            self.backward_time.update(time.time() - backward_start_time)
            self.batch_time.update(time.time() - start_time)
            start_time = time.time()
            self.configer.plus_one('iters')

            # Print the log info & reset the states.
            if self.configer.get('iters') % self.configer.get('solver', 'display_iter') == 0:
                #torch.cuda.synchronize()
                avg_loss = torch.tensor(self.train_losses.avg, device=oldimg.device)
                if dist.is_initialized():

                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    avg_loss = avg_loss.item() / dist.get_world_size()

                if get_rank() == 0:
                    Log.info('Train Epoch: {0}\tTrain Iteration: {1}\t'
                             'Time {batch_time.sum:.3f}s / {2}iters, ({batch_time.avg:.3f})\t'
                             'Forward Time {foward_time.sum:.3f}s / {2}iters, ({foward_time.avg:.3f})\t'
                             'Backward Time {backward_time.sum:.3f}s / {2}iters, ({backward_time.avg:.3f})\t'
                             'Loss Time {loss_time.sum:.3f}s / {2}iters, ({loss_time.avg:.3f})\t'
                             'Data load {data_time.sum:.3f}s / {2}iters, ({data_time.avg:3f})\n'
                             'Learning rate = {3}\tLoss = {loss:.8f}\n'.format(
                        self.configer.get('epoch'), self.configer.get('iters'),
                        self.configer.get('solver', 'display_iter'),
                        self.module_runner.get_lr_from_scheduler(self.optimizer, self.scheduler), batch_time=self.batch_time,
                        foward_time=self.foward_time, backward_time=self.backward_time, loss_time=self.loss_time,
                        data_time=self.data_time, loss=avg_loss))
            if self.tb_logger is not None:
                i_iter = self.configer.get("iters")
                #self.tb_logger.add_scalar("lr", self.module_runner.get_lr(self.optimizer, self.scheduler)[0], i_iter)
                self.tb_logger.add_scalars("loss",{"total_loss": display_loss},i_iter)

                self.loss_time.reset()
                self.batch_time.reset()
                self.foward_time.reset()
                self.backward_time.reset()
                self.data_time.reset()
                self.train_losses.reset()

            #torch.cuda.empty_cache()
            # save checkpoints for swa
            #
            if self.configer.get('iters') % self.configer.get('solver', 'test_interval') == 0:
                vis_dir = self.configer.get("checkpoints", "checkpoints_name")
                self.__val(vis_dir=vis_dir)
                #self._vis_gradcam(vis_dir=vis_dir)
            # if self.configer.get('iters') == self.configer.get('solver', 'max_iters'):
            #     break
        # torch.save({
        #     'epoch': self.configer.get('epoch'),
        #     'state_dict': self.model.state_dict(),
        #     'optimizer_state_dict': self.optimizer.state_dict(),
        #     'loss': self.train_losses,
        # }, f"checkpoints/latentclipcd_{self.configer.get('epoch')}.pt")
        self.configer.plus_one('epoch')

    def val(self):

        vis_dir = self.configer.get("checkpoints","checkpoints_name")
        if self.configer.get('network', 'resume') is not None:
            self.__val(vis_dir=vis_dir, save=False)


    def train(self):




        # if self.configer.get('network', 'resume') is not None:
        #    if self.configer.get('network', 'resume_val'):
        #        self.__val()
        #        #self._vis_gradcam()
        #        return
        #    elif self.configer.get('network', 'resume_train'):
        #        self.__val(data_loader=self.data_loader.get_valloader(dataset='train'))
        #        return
        #    # return
        #
        # if self.configer.get('network', 'resume') is not None and self.configer.get('network', 'resume_val'):
        #    self.__val(data_loader=self.data_loader.get_valloader(dataset='val'))
        #    return

        while self.configer.get('epoch') < self.configer.get('train', 'epoch'):
            if is_distributed():
                self.train_loader.sampler.set_epoch(self.configer.get('epoch'))
            self.__train()

        # use swa to average the model
        if 'swa' in self.configer.get('lr', 'lr_policy'):
            self.optimizer.swap_swa_sgd()
            self.optimizer.bn_update(self.train_loader, self.seg_net)

        self.__val(data_loader=self.data_loader.get_valloader(dataset='val'))




