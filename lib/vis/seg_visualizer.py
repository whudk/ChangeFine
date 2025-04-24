#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Visualizer for segmentation.
import glob
import os

import cv2
import numpy as np
import torchvision.utils

from utils.tools.logger import Logger as Log
import torch
import matplotlib.pyplot as plt

SEG_DIR = 'vis/results/seg'

class DeNormalize(object):
    """DeNormalize a ``torch.tensor``

    Args:
        inputs (torch.tensor): tensor to be normalized.
        mean: (list): the mean of RGB
        std: (list): the std of RGB

    Returns:
        Tensor: Normalized tensor.
    """
    def __init__(self, div_value, mean, std):
        self.div_value = div_value
        self.mean =  torch.Tensor(mean)
        self.std =torch.Tensor(std)

    def __call__(self, inputs):
        result = inputs.cpu().clone()

        result = result * self.std[None, :, None, None] + self.mean[None, :, None, None]
        # for i in range(result.size(0)):
        #     result[i,:, :, :] = result[i,:, :, :] * self.std[i] + self.mean[i]

        return result.mul_(self.div_value).to(inputs.device)
class SegVisualizer(object):

    def __init__(self, configer=None):
        self.configer = configer

    def vis_fn(self, preds, targets, ori_img_in=None, name='default', sub_dir='fn'):
        base_dir = os.path.join(self.configer.get('project_dir'), SEG_DIR, sub_dir)
        if not os.path.exists(base_dir):
            Log.error('Dir:{} not exists!'.format(base_dir))
            os.makedirs(base_dir)

        if not isinstance(preds, np.ndarray):
            if len(preds.size()) > 3:
                Log.error('Preds size is not valid.')
                exit(1)

            if len(preds.size()) == 3:
                preds = preds.clone().data.cpu().numpy()

            if len(preds.size()) == 2:
                preds = preds.unsqueeze(0).data.cpu().numpy()

        else:
            if len(preds.shape) > 3:
                Log.error('Preds size is not valid.')
                exit(1)

            if len(preds.shape) == 2:
                preds = preds.unsqueeze(0)

        if not isinstance(targets, np.ndarray):

            if len(targets.size()) == 3:
                targets = targets.clone().data.cpu().numpy()

            if len(targets.size()) == 2:
                targets = targets.unsqueeze(0).data.cpu().numpy()

        else:
            if len(targets.shape) == 2:
                targets = targets.unsqueeze(0)

        if ori_img_in is not None:
            if not isinstance(ori_img_in, np.ndarray):
                if len(ori_img_in.size()) < 3:
                    Log.error('Image size is not valid.')
                    exit(1)

                if len(ori_img_in.size()) == 4:
                    ori_img_in = ori_img_in.data.cpu()

                if len(ori_img_in.size()) == 3:
                    ori_img_in = ori_img_in.unsqueeze(0).data.cpu()

                ori_img = ori_img_in.clone()
                for i in range(ori_img_in.size(0)):
                    ori_img[i] = DeNormalize(div_value=self.configer.get('normalize', 'div_value'),
                                             mean=self.configer.get('normalize', 'mean'),
                                             std=self.configer.get('normalize', 'std'))(ori_img_in.clone())

                ori_img = ori_img.numpy().transpose(2, 3, 1).astype(np.uint8)

            else:
                if len(ori_img_in.shape) == 3:
                    ori_img_in = ori_img_in.unsqueeze(0)

                ori_img = ori_img_in.copy()

        for img_id in range(preds.shape[0]):
            label = targets[img_id]
            pred = preds[img_id]
            result = np.zeros(shape=(pred.shape[0], pred.shape[1], 3), dtype=np.uint8)

            for i in range(self.configer.get('data', 'num_classes')):
                mask0 = np.zeros_like(label, dtype=np.uint8)
                mask1 = np.zeros_like(label, dtype=np.uint8)
                mask0[label[:] == i] += 1
                mask0[pred[:] == i] += 1
                mask1[pred[:] == i] += 1
                result[mask0[:] == 1] = self.configer.get('details', 'color_list')[i]
                result[mask1[:] == 1] = (0, 0, 0)

            image_result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            if ori_img_in is not None:
                image_result  = cv2.addWeighted(ori_img[i], 0.6, image_result, 0.4, 0)

            cv2.imwrite(os.path.join(base_dir, '{}_{}.jpg'.format(name, img_id)), image_result)

    def vis_gans(self,real,fake,rec, names = None,sub_dir = 'gans'):
        base_dir = os.path.join(self.configer.get('project_dir'), SEG_DIR, sub_dir)
        if not os.path.exists(base_dir):
            Log.error('Dir:{} not exists!'.format(base_dir))
            os.makedirs(base_dir)

        real_imgs = DeNormalize(div_value=self.configer.get('normalize', 'div_value'),
                              mean=self.configer.get('normalize', 'mean'),
                              std=self.configer.get('normalize', 'std'))(real.clone()) / 255.0

        fake_imgs =  DeNormalize(div_value=self.configer.get('normalize', 'div_value'),
                              mean=self.configer.get('normalize', 'mean'),
                              std=self.configer.get('normalize', 'std'))(fake.clone()) / 255.0

        rec_imgs = DeNormalize(div_value=self.configer.get('normalize', 'div_value'),
                                mean=self.configer.get('normalize', 'mean'),
                                std=self.configer.get('normalize', 'std'))(rec.clone()) /255.0
        # real_imgs = real_imgs.numpy().transpose(0, 2, 3, 1).astype(np.uint8)
        # fake_imgs = fake_imgs.numpy().transpose(0, 2, 3, 1).astype(np.uint8)
        # rec_imgs = rec_imgs.numpy().transpose(0, 2, 3, 1).astype(np.uint8)


        save_images  = torchvision.utils.make_grid(torch.cat((real_imgs,fake_imgs,rec_imgs),dim =0),nrows = real_imgs.size()[0] * 3)

        torchvision.utils.save_image(save_images, os.path.join(base_dir,  str(names) + "_.jpg"))

    def color_preds(self, label, preds):
        color_map = np.zeros((label.shape[1], label.shape[2], 3), dtype=np.uint8)
        label = label.cpu().numpy()
        preds = preds.cpu().numpy()

        color_map[(label[0] == 1), :] = [255, 255, 255]  # 将标签为1映射为白色
        color_map[(label[0] == 0), :] = [0, 0, 0]  # 将标签为0映射为黑色
        color_map[(label[0] == 1) & (preds[0] == 0), :] = [0, 255, 0]  # 将FN映射为绿色
        color_map[(label[0] == 0) & (preds[0] == 1), :] = [255, 0, 0]  # 将FP映射为红色
        color_map = torch.from_numpy(color_map).permute(2, 0, 1) / 255.0
        return color_map
    def vis_gradcampp(self, cam_map, names = None, sub_dir='fp', cls = 1):
        from lib.vis.gradcam.utils import  visualize_cam
        base_dir = os.path.join(self.configer.get('project_dir'), SEG_DIR, sub_dir)




        #heatmap, result = visualize_cam(saliency_map, output)


        if not os.path.exists(base_dir):
            Log.error('Dir:{} not exists!'.format(base_dir))
            os.makedirs(base_dir)

        for b in range(len(names)):
            basename = os.path.basename(names[b]["old_path"]).split('.')[0]
            save_name = os.path.join(base_dir, "{}_{}".format(basename, "cam.png"))
            plt.clf()  # 清除上一个绘图
            plt.imshow(cam_map.squeeze().cpu(), cmap='jet', vmin=0, vmax=1)
            # plt.axis("off")
            # plt.title("Grad-CAM++")
            # plt.show()
            plt.savefig(save_name)



    def vis_cd(self, preds_chg, targets, ori_left_image, ori_right_image, names = None, sub_dir='fp', draw_contours = False, palette_left = False, palette_right = False, pallete_target = True  ):
        #参数说明：
        #preds_chg:预测结果 256 * 256 （0，1）
        #targets: 真值：256*256 （0， 1）
        #ori_left_image：T1 Image( 0 - 1)
        #ori_right_image : T2 Image( 0-1)
        #draw_contours, pallete_left, palette_right, pallite_target False
        # return 保存到  sub_dir_names, names 传入 影像编号{1，2，3，4， ...}
        base_dir = os.path.join(self.configer.get('project_dir'), SEG_DIR, sub_dir)
        if not os.path.exists(base_dir):
            Log.error('Dir:{} not exists!'.format(base_dir))
            os.makedirs(base_dir)

        #DeNormalize 从 0-1 根据均值方差  回溯到【0-255】，T1
        ori_imgs_left = DeNormalize(
            div_value=self.configer.get('normalize','left')['div_value'],
            mean=self.configer.get('normalize', 'left')['mean'],
            std=self.configer.get('normalize', 'left')['std']
        )(ori_left_image.clone())
        # DeNormalize 从 0-1 根据均值方差  回溯到【0-255】 T2
        ori_imgs_right = DeNormalize(
            div_value=self.configer.get('normalize', 'right')['div_value'],
            mean=self.configer.get('normalize', 'right')['mean'],
            std=self.configer.get('normalize', 'right')['std']
        )(ori_right_image.clone())

        if palette_left:
            ori_imgs_left = self.map_image(ori_imgs_left,targets)

        if palette_right:
            ori_imgs_right = self.map_image(ori_imgs_right, targets)

        if draw_contours:
            ori_imgs_left,ori_imgs_right = self.draw_contours(preds_chg,targets,ori_imgs_left,ori_imgs_right)


        # ori_imgs_right = ori_imgs_right[:, [2, 1, 0]] / 255.0
        # ori_imgs_left = ori_imgs_left[:, [2, 1, 0]] / 255.0
        ori_imgs_right = ori_imgs_right / 255.0
        ori_imgs_left = ori_imgs_left/ 255.0
        b,_,_ = targets.shape
        mask_ignore = torch.zeros_like(targets, dtype=torch.long).to(targets.device)
        mask_ignore[targets == -1] = 1
        mask_ignore = mask_ignore.bool()
        targets[mask_ignore] = 0
        preds_chg[mask_ignore] = 0
        targets = targets.unsqueeze(1).repeat(1, 3, 1, 1)
        # if pallete_target:
        #     targets = self.map_image(targets, targets)

        preds_chg = preds_chg.unsqueeze(1).repeat(1, 3, 1, 1)
        show_tensors  = torch.cat((ori_imgs_left,ori_imgs_right, targets,preds_chg),dim=0)


        for b in range(len(ori_imgs_left)):
            basename = os.path.basename(names[b]["old_path"]).split('.')[0]
            save_left = os.path.join(base_dir, "{}_{}".format(basename,"t1.png") )
            save_right = os.path.join(base_dir, "{}_{}".format(basename,"t2.png") )
            save_preds = os.path.join(base_dir, "{}_{}".format(basename, "t_pred.png") )
            save_label = os.path.join(base_dir, "{}_{}".format(basename, "t_label.png"))
            torchvision.utils.save_image(ori_imgs_left[b], save_left)
            torchvision.utils.save_image(ori_imgs_right[b], save_right)
            if pallete_target:
                color_pred = self.color_preds(targets[b], preds_chg[b])
            else:
                color_pred = preds_chg[b]
            torchvision.utils.save_image(color_pred, save_preds)
            torchvision.utils.save_image(targets[b].float(), save_label)

        # grid_image =torchvision.utils.make_grid(show_tensors,nrow=b)
        #
        #
        # save_name = os.path.join(base_dir,"{}_{}.png".format(self.configer.get("iters"),names))
        #
        # torchvision.utils.save_image(grid_image,save_name)


    def draw_contours(self,pred, gt, left_image, right_image):

        device = pred.device
        if isinstance(pred, torch.Tensor):
            pred1 = pred.cpu().numpy().astype(np.uint8)
        if isinstance(gt, torch.Tensor):
            gt1 = gt.cpu().numpy().astype(np.uint8)
        if isinstance(left_image, torch.Tensor):
            left_image1 = left_image.permute(0,2,3,1).cpu().contiguous().numpy().astype(np.uint8)
        if isinstance(right_image, torch.Tensor):
            right_image1 = right_image.permute(0,2,3,1).cpu().contiguous().numpy().astype(np.uint8)

        bs = left_image.shape[0]

        for b in range(bs):
            pred_contours, h = cv2.findContours(pred1[b],mode=cv2.RETR_TREE,method= cv2.CHAIN_APPROX_NONE)
            gt_contours,_ = cv2.findContours(gt1[b],mode=cv2.RETR_TREE,method= cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(left_image1[b],pred_contours,-1,(0,255,0),thickness=2)
            cv2.drawContours(left_image1[b],gt_contours,-1,(255,0,0),thickness=2)
            cv2.drawContours(right_image1[b], pred_contours, -1, (0, 255, 0), thickness=2)
            cv2.drawContours(right_image1[b], gt_contours, -1, (255, 0, 0), thickness=2)

        left_image = torch.from_numpy(left_image1).to(device)
        right_image = torch.from_numpy(right_image1).to(device)
        return left_image.permute(0,3,1,2),right_image.permute(0,3,1,2)
    def map_image(self,image,targets):
        color_palette = self.configer.get('details', 'color_list').copy()
        color_palette *= 100
        image = image.permute(0, 2, 3, 1).long()
        for i in range(100):
           # mask0 = torch.zeros_like(targets, dtype=torch.uint8)
            mask1 = torch.zeros_like(targets, dtype=torch.uint8)
            #mask0[preds_seg[:,:,:,0] == i] += 1
            mask1[image[:,:,:,0] == i] += 1
            #preds_seg[mask0[:] == 1,:] = torch.tensor(self.configer.get('details', 'color_list')[i]).to(preds_seg.device)
            image[mask1[:] == 1,:] = torch.tensor(color_palette[i]).to(image.device)

        return  image.permute(0, 3, 1, 2)

    def vis_shpcd(self, preds_chg, targets, ori_left_image, ori_right_image,  heatmap = None,result_heatmap = None, names = None, sub_dir='fp', draw_contours = False ,denorm = None ):



        base_dir = os.path.join(self.configer.get('project_dir'), SEG_DIR, sub_dir)
        if not os.path.exists(base_dir):
            Log.error('Dir:{} not exists!'.format(base_dir))
            os.makedirs(base_dir)
        if denorm is not None:
            ori_imgs_left = DeNormalize(
                div_value=self.configer.get('normalize','left')['div_value'],
                mean=self.configer.get('normalize', 'left')['mean'],
                std=self.configer.get('normalize', 'left')['std']
            )(ori_left_image.clone())
            ori_imgs_right = DeNormalize(
                div_value=self.configer.get('normalize', 'right')['div_value'],
                mean=self.configer.get('normalize', 'right')['mean'],
                std=self.configer.get('normalize', 'right')['std']
            )(ori_right_image.clone())





        #preds_seg  = preds_seg.unsqueeze(1).repeat(1, 3, 1, 1).permute(0,2,3,1)
        if len(ori_left_image.shape) == 3:
            ori_imgs_left  = torch.unsqueeze(ori_left_image,dim=0)
        if len(ori_right_image.shape) == 3:
            ori_imgs_right  = torch.unsqueeze(ori_right_image,dim=0)




        ori_imgs_left = self.map_image(ori_imgs_left,targets)


        #preds_seg = preds_seg.permute(0,3,1,2)
        #preds_seg = preds_seg / 255.0

        b,_,_ = targets.shape

        mask_ignore  = torch.zeros_like(targets,dtype=torch.long).to(targets.device)
        mask_ignore[targets == -1] = 1
        mask_ignore = mask_ignore.bool()
        targets[mask_ignore] = 0
        preds_chg[mask_ignore] = 0




        if draw_contours:
            ori_imgs_left,ori_imgs_right = self.draw_contours(preds_chg,targets,ori_imgs_left,ori_imgs_right)

        targets = targets.unsqueeze(1).repeat(1, 3, 1, 1)
        preds_chg = preds_chg.unsqueeze(1).repeat(1, 3, 1, 1)
        ori_imgs_right = ori_imgs_right[:,[2,1,0]] /255.0
        ori_imgs_left = ori_imgs_left / 255.0


        show_tensors  = torch.cat((ori_imgs_left,ori_imgs_right, targets,preds_chg),dim=0)
        if heatmap is not None:
            result_heatmap = DeNormalize(
                div_value=self.configer.get('normalize', 'right')['div_value'],
                mean=self.configer.get('normalize', 'right')['mean'],
                std=self.configer.get('normalize', 'right')['std']
            )(result_heatmap.clone()) / 255.0
            show_tensors = torch.cat((show_tensors,heatmap,result_heatmap),dim = 0)



        for b in range(len(ori_imgs_left)):
            save_left = os.path.join(base_dir, "{}_{}_{}_hlubm.png".format(self.configer.get("iters"), b, names))
            save_right = os.path.join(base_dir, "{}_{}_{}_crism.png".format(self.configer.get("iters"), b, names))
            save_preds = os.path.join(base_dir, "{}_{}_{}_preds.png".format(self.configer.get("iters"), b, names))
            save_label = os.path.join(base_dir, "{}_{}_{}_label.png".format(self.configer.get("iters"), b, names))
            torchvision.utils.save_image(ori_imgs_left[b], save_left)
            torchvision.utils.save_image(ori_imgs_right[b], save_right)
            color_pred = self.color_preds(targets[b], preds_chg[b])
            torchvision.utils.save_image(color_pred, save_preds)
            torchvision.utils.save_image(targets[b].float(), save_label)
        # grid_image =torchvision.utils.make_grid(show_tensors,nrow=b)
        #
        # save_name = os.path.join(base_dir, "{}_{}.png".format(self.configer.get("iters"), names))
        #
        # torchvision.utils.save_image(grid_image,save_name)


    def vis_fp(self, preds, targets, ori_img_in=None, names=None, sub_dir='fp'):
        base_dir = os.path.join(self.configer.get('project_dir'), SEG_DIR, sub_dir)
        if not os.path.exists(base_dir):
            Log.error('Dir:{} not exists!'.format(base_dir))
            os.makedirs(base_dir)

        mask_ignore = torch.zeros_like(targets, dtype=torch.long).to(targets.device)
        mask_ignore[targets == -1] = 1
        mask_ignore = mask_ignore.bool()
        targets[mask_ignore] = 0
        preds[mask_ignore] = 0


        if not isinstance(preds, np.ndarray):
            if len(preds.size()) > 3:
                Log.error('Preds size is not valid.')
                exit(1)

            elif len(preds.size()) == 3:
                preds = preds.clone().data.cpu().numpy()

            elif len(preds.size()) == 2:
                preds = preds.unsqueeze(0).data.cpu().numpy()
            else:
                Log.error('Preds size is not valid.')
                exit(1)

        else:
            if len(preds.shape) > 3:
                Log.error('Preds size is not valid.')
                exit(1)

            if len(preds.shape) == 2:
                preds = preds.unsqueeze(0)

        if not isinstance(targets, np.ndarray):

            if len(targets.size()) == 3:
                targets = targets.clone().data.cpu().numpy()

            elif len(targets.size()) == 2:
                targets = targets.unsqueeze(0).data.cpu().numpy()
            else:
                Log.error('targets size is not valid.')
                exit(1)

        else:
            if len(targets.shape) == 2:
                targets = targets.unsqueeze(0)

        if ori_img_in is not None:
            if not isinstance(ori_img_in, np.ndarray):
                if len(ori_img_in.size()) < 3:
                    Log.error('Image size is not valid.')
                    exit(1)

                elif len(ori_img_in.size()) == 4:
                    ori_img_in = ori_img_in.data.cpu()

                elif len(ori_img_in.size()) == 3:
                    ori_img_in = ori_img_in.unsqueeze(0).data.cpu()
                else:
                    Log.error('Image size is not valid.')
                    exit(1)

                ori_img = ori_img_in.clone()

                ori_img= DeNormalize(div_value=self.configer.get('normalize', 'div_value'),
                                             mean=self.configer.get('normalize', 'mean'),
                                             std=self.configer.get('normalize', 'std'))(ori_img.clone())

                # for i in range(ori_img_in.size(0)):
                #     ori_img[i] = DeNormalize(div_value=self.configer.get('normalize', 'div_value'),
                #                              mean=self.configer.get('normalize', 'mean'),
                #                              std=self.configer.get('normalize', 'std'))(ori_img_in.clone())

                ori_img = ori_img.numpy().transpose(0, 2, 3, 1).astype(np.uint8)

            else:
                if len(ori_img_in.shape) == 3:
                    ori_img_in = ori_img_in.unsqueeze(0)

                ori_img = ori_img_in.copy()

        for img_id in range(preds.shape[0]):
            label = targets[img_id]
            pred = preds[img_id]
            result = np.zeros(shape=(pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
            gt_image =  np.zeros(shape=(pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
            pred_image = np.zeros(shape=(pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
            # for i in range(self.configer.get('data', 'num_classes')):
            #     mask0 = np.zeros_like(label, dtype=np.uint8)
            #     mask1 = np.zeros_like(label, dtype=np.uint8)
            #     mask0[label[:] == i] += 1
            #     mask0[pred[:] == i] += 1
            #     mask1[label[:] == i] += 1
            #     result[mask0[:] == 1] = self.configer.get('details', 'color_list')[i]
            #     result[mask1[:] == 1] = (0, 0, 0)
            # image_result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            # if ori_img_in is not None:
            #     image_result = cv2.addWeighted(ori_img[img_id], 0.6, image_result, 0.4, 0)
            for i in range(self.configer.get('data', 'num_classes')):
                mask0 = np.zeros_like(label, dtype=np.uint8)
                mask1 = np.zeros_like(label, dtype=np.uint8)
                mask0[label[:] == i] += 1
                mask1[pred[:] == i] += 1
                gt_image[mask0[:] == 1] = self.configer.get('details', 'color_list')[i]
                pred_image[mask1[:] == 1] = self.configer.get('details', 'color_list')[i]

            #name = os.path.splitext(os.path.basename(names[img_id]))[0]




            tensor_ori_img = torch.from_numpy(ori_img[img_id]).unsqueeze(0).permute(0,3,1,2)
            tensor_ori_img = tensor_ori_img / 255.0
            tensor_gt_image = torch.from_numpy(gt_image).unsqueeze(0).permute(0,3,1,2)
            tensor_gt_image = tensor_gt_image / 255.0
            tensor_pred_image = torch.from_numpy(pred_image).unsqueeze(0).permute(0, 3, 1, 2)
            tensor_pred_image = tensor_pred_image / 255.0

            grid_image = torch.cat((tensor_ori_img,tensor_gt_image,tensor_pred_image))
            grid_image = torchvision.utils.make_grid(grid_image,nrow=1)
            save_name = os.path.join(base_dir, "{}_{}".format(self.configer.get("iters"), names))
            torchvision.utils.save_image(grid_image,os.path.join(base_dir, save_name))
            #
            # cv2.imwrite(os.path.join(base_dir, 'image' + names[img_id]), ori_img[img_id])
            # cv2.imwrite(os.path.join(base_dir, 'gt'+ names[img_id]), gt_image)
            # cv2.imwrite(os.path.join(base_dir, 'pred' + names[img_id]), pred_image)
    def vis_gt(self,target_path,name):
        colorlist = self.configer.get("details",'color_list')
        basename = os.path.basename(target_path).split('.')[0]
        #dirname  = os.path.dirname(target_path)

        out_name = os.path.join(name,basename + "_color.png")


        label = cv2.imread(target_path)
        if len(label.shape) > 2:
            label = label[:,:,0]
        #h,w,c = label.shape
        h,w   = label.shape


        colorlabel = np.zeros(shape=(h,w,3),dtype=np.int)

        n_classes = self.configer.get('data',"num_classes")

        for c in range(n_classes):
            mask = np.zeros_like(label)
            mask[label == c] = 1
            colorlabel[mask == 1] = colorlist[c]
        cv2.imwrite(out_name,colorlabel)

        return


    def error_map(self, im, pred, gt):
        canvas = im.copy()
        canvas[np.where((gt - pred != [0, 0, 0]).all(axis=2))] = [0, 0, 0]
        pred[np.where((gt - pred == [0, 0, 0]).all(axis=2))] = [0, 0, 0]
        canvas = cv2.addWeighted(canvas, 1.0, pred, 1.0, 0)
        # canvas = cv2.addWeighted(im, 0.3, canvas, 0.7, 0)
        canvas[np.where((gt == [0, 0, 0]).all(axis=2))] = [0, 0, 0]
        return canvas






if __name__ == '__main__':
    import argparse
    from lib.utils.tools.configer import Configer
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', default='D:/dengkai/dengkai_DL/configs/RSdata/CYCLE_GANS.json', type=str,
                        dest='configs', help='The file of the hyper parameters.')

    parser.add_argument('--data_dir', default='D:/dengkai/data/resultdiffusion/labels', type=str,
                         help='The dir of data.')
    parser.add_argument('REMAIN', nargs='*')
    args_parser = parser.parse_args()

    configer = Configer(args_parser=args_parser)


    data_dir  = args_parser.data_dir

    out_dir = data_dir + "/label_color"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    vis = SegVisualizer(configer = configer)


    exts = ['*.png','*.tif','*.jpg','*.bmp']
    all_files = []
    for ext in exts:
        all_files.extend(glob.glob(os.path.join(data_dir,ext)))
    for file in all_files:
        vis.vis_gt(file,out_dir)
    print ('done')