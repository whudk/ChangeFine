from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os

import cv2
import torch

import numpy as np
import random
from torch.utils import data
import math
from tqdm import tqdm
from xml.dom.minidom import parse



from utils.tools.logger import Logger as Log
import torch.nn.functional as F
from skimage import measure
from  models.sam.utils.amg import rle_to_mask
from torch.utils.data import DataLoader
from pycocotools import mask as maskUtils
from models.sam.utils.transforms import ResizeLongestSide
import json


import numpy as np
import json

class SamClipCD_dataset(torch.utils.data.Dataset):

    def __init__(self, json_file,configer, aug_transform=None,transxml = None,
                 img_transform=None, label_transform=None, device="cuda", split="train",


                 **kwargs):


        self.configer = configer
        self.aug_transform = aug_transform
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.device = device
        self.num_classes = self.configer.get('data', 'num_classes')

        # refine label value
        self.transxml = self.configer.get('config', 'transxml')
        self.dict = None
        if os.path.exists(self.transxml):
            _, self.dict = self.readtransxml(self.transxml)

        self.neg = False
        if split == "neg":
            neg_dir = self.configer.get('data', 'neg')["dir"]
            with open(json_file, 'r', encoding="utf-8") as file:
                samples = json.load(file)["samples"]
            self.annotations = {
                "images":[]
            }
            single_smp = {}
            for i, smp in enumerate(samples):
                single_smp["oldpath"] =  neg_dir + smp["pre"]#os.path.join(neg_dir, smp["pre"])
                single_smp["newpath"] = neg_dir + smp["cur"]#os.path.join(neg_dir, smp["cur"])
                single_smp["labelpath"] = neg_dir + smp["label"]  # os.path.join(neg_dir, smp["label"])

                self.annotations["images"].append(single_smp)
            self.neg = True
            #self.annotations = [ os.path.join(neg_dir, ann["pre"])  + " " + os.path.join(neg_dir, ann["cur"])  + " " + os.path.join(neg_dir, ann["label"]) for ann in self.annotations]
        else:
            with open(json_file, 'r', encoding="utf-8") as file:
                self.annotations = json.load(file)
            #   读 文件目录 或者是一个txt
            #   txt 的组织  old.png  new.png  change_mask.png


        self.images = self.annotations['images']
        # if 'categories' in self.annotations.keys():
        #     self.categories = self.annotations["categories"]
        # if 'annotations' in self.annotations.keys():
        #     self.anns = {ann['image_id']: [] for ann in self.annotations['annotations']}
        #     for ann in self.annotations['annotations']:
        #         bbox = ann['bbox']
        #         segmentation = ann["segmentation"]
        #         x_min, y_min, width, height = bbox
        #         x_max = x_min + width
        #         y_max = y_min + height
        #         center_coor = [(x_min + x_max) / 2, (y_min + y_max) / 2]  # Calculate center
        #         self.anns[ann['image_id']].append(
        #             {'bbox': [x_min, y_min, x_max, y_max], 'center_coor': center_coor, 'segmentation': segmentation,
        #              'label': self.find_id_by_name(ann['category_id'], self.categories),
        #              'category_id': ann['category_id']})








        # if split == "val":
        #     indices = np.random.choice(range(len(self.images)), 500, replace=True).tolist()
        #     self.images = [self.images[i] for  i in indices]
            #self.anns = [self.anns[i] for i in indices]

        #self.configer = configer
        self.aug_transform = aug_transform
        self.img_transform = img_transform
        self.label_transform = label_transform

        self.transform = ResizeLongestSide(512)

        #self.num_classes = self.configer.get('data', 'num_classes')


    def find_id_by_name(self, name, categories):
        if name == 'unknown':
            return 1

        for category in categories:
            if category['name'] == name:
                return category['id']
        return None  # Return None if no match is found

    def find_boxes_and_centers(self, target):

        point_coords = []
        point_labels = []
        bboxes = []

        label, N  = measure.label(target, 0, True,2)
        props = measure.regionprops(label)

        for n in range(1, N+1):
            point_label = int(np.unique(target[label == n]))
            point_coords.append(props[n-1].centroid)
            point_labels.append(point_label)
            bboxes.append(props[n-1].bbox)
        return point_coords,point_labels,bboxes
    def checkpaths(self, lines, shuffle=True, clampN=False):
        self.ids = []
        for id, line in tqdm(enumerate(lines)):
            paths = line.rstrip().split()
            exist_flag = True
            for path in paths:
                if not os.path.exists(path):
                    print("{} is not existed".format(path))
                    # print("check line:{} in {}".format(id+1,f.__dir__))
                    exist_flag = False
                    break
            if exist_flag:
                self.ids.append(line)
        if shuffle:
            random.shuffle(self.ids)
        if clampN:
            self.ids = self.ids[:1000]
        return self.ids
    def get_classnames(self):
        return  self.class_names
    def readtransxml(self,transxml):
        path_transxml = transxml
        if not os.path.exists(path_transxml):
            Log('Error:{} is not existed.'.format(path_transxml))
        transBM = parse(path_transxml)
        root = transBM.documentElement
        all_codes = root.getElementsByTagName('BM')
        all_dict = {}
        num_class = 0
        for node in all_codes:
            class_geoid_name = node.attributes['key'].value
            class_id = node.attributes['val'].value
            all_dict[int(class_geoid_name)] = int(class_id)
            if int(class_id) > num_class:
                num_class = num_class + 1
        return num_class,all_dict

    def random_select_points(self, mask, N = 20):
        y_coords, x_coords = np.where(mask == 1)

        # 检查是否有足够的点
        if len(y_coords) < N:
            indices = np.random.choice(range(len(y_coords)), N, replace=True)
            selected_coords = [(y_coords[i], x_coords[i]) for i in indices]
        else:
            # 从这些坐标中随机选择 N 个
            indices = np.random.choice(range(len(y_coords)), N, replace=False)
            selected_coords = [(y_coords[i], x_coords[i]) for i in indices]

            # 打印所选点的坐标
            #print("选中的点的坐标:", selected_coords)
        return  selected_coords

    def _stastic_ids(self):
        from PIL import Image
        # Initialize counters for class distributions
        class_0_count = 0
        class_1_count = 0

        # Initialize accumulators for mean and std
        mean_accum = np.zeros(3)  # Running sum of means for oldimages (R, G, B)
        std_accum = np.zeros(3)  # Running sum of stds for oldimages (R, G, B)

        # Initialize counts for averaging
        num_images = len(self.images)

        for idx in tqdm(range(num_images)):
            # Replace backslashes with forward slashes for cross-platform compatibility
            oldimage_path = self.images[idx]["oldpath"].replace("\\", "/")
            newimage_path = self.images[idx]["newpath"].replace("\\", "/")
            label_path = self.images[idx]["labelpath"].replace("\\", "/")

            # Load label image (assuming it's a grayscale image with 0 and 1 as labels)
            label_image = Image.open(label_path).convert('L')  # Convert to grayscale if not already
            label_array = np.array(label_image)  # Convert to numpy array

            # Count the number of 0s and 1s in the label
            class_0_count += np.sum(label_array == 0)
            class_1_count += np.sum(label_array == 1)

            # For mean and std calculation: load the old and new images
            oldimage = Image.open(oldimage_path).convert('RGB')  # Assuming color images
            oldimage_array = np.array(oldimage) /255.0 # Convert to numpy array
            old_mean = np.mean(oldimage_array, axis=(0, 1))  # Mean for (R, G, B)
            old_std = np.std(oldimage_array, axis=(0, 1))  # Std for (R, G, B)

            newimage = Image.open(newimage_path).convert('RGB')  # Assuming color images
            newimage_array = np.array(newimage) /255.0  # Convert to numpy array
            new_mean = np.mean(newimage_array, axis=(0, 1))  # Mean for (R, G, B)
            new_std = np.std(newimage_array, axis=(0, 1))  # Std for (R, G, B)

            # Accumulate means and stds
            mean_accum += (old_mean + new_mean)
            std_accum += (old_std + new_std)


        # Calculate averages for all images
        avg_mean = mean_accum /  (2  * num_images)
        avg_std = std_accum / (2 * num_images)


        # Calculate class distribution (0 and 1 proportions)
        total_pixels = class_0_count + class_1_count
        class_0_proportion = class_0_count / total_pixels
        class_1_proportion = class_1_count / total_pixels

        Log.info(f"Old image dataset average mean (R, G, B): {avg_mean}")
        Log.info(f"Old image dataset average std (R, G, B): {avg_std}")


        return avg_mean, avg_std,  class_0_proportion, class_1_proportion


    def __getitem__(self, idx):

        oldimage_path = self.images[idx]["oldpath"].replace("\\", "/")
        newimage_path = self.images[idx]["newpath"].replace("\\", "/")
        label_path = self.images[idx]["labelpath"].replace("\\", "/")
        # oldimage_path = oldimage_path.replace("/home/dk/","F:").replace("\\", "/")
        # newimage_path = newimage_path.replace("/home/dk/","F:").replace("\\", "/")
        # label_path = label_path.replace("/home/dk/", "F:").replace("\\", "/")

        # oldimage_path = oldimage_path.replace("/home/dk/", "F:/")
        # newimage_path = newimage_path.replace("/home/dk/", "F:/")
        # label_path = label_path.replace("/home/dk/", "F:/")

        oldimg  = cv2.imdecode(np.fromfile(oldimage_path,dtype=np.uint8),-1)
        newimg = cv2.imdecode(np.fromfile(newimage_path,dtype=np.uint8),-1)



        oldimg = cv2.cvtColor(oldimg, cv2.COLOR_BGR2RGB)

        newimg = cv2.cvtColor(newimg, cv2.COLOR_BGR2RGB)


        # if self.neg  is False:
        #     oldimg = cv2.cvtColor(oldimg, cv2.COLOR_BGR2RGB)[:,:,::-1]
        #     newimg = cv2.cvtColor(newimg, cv2.COLOR_BGR2RGB)[:,:,::-1]

        H,W,C = oldimg.shape

        if not os.path.exists(label_path):
            label = None
            Log.error('cannot find {}'.format(label_path))
        else:
            label =  cv2.imdecode(np.fromfile(label_path,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
            if self.dict is not None:
                label_map = np.vectorize(self.dict.get)(label)
                if None in label_map:
                    pix_val = np.unique(label)
                    mix_index = []
                    for pix in pix_val:
                        if str(pix_val) not in self.dict.keys():
                            mix_index.append(pix)
                    assert len(mix_index) == 0,'{}\n cannot find {} in transxml.'.format(label_path,mix_index)
                else:
                    label = label_map



        oldimg = oldimg.copy()
        newimg = newimg.copy()
        label = label.copy()
        # oldimg_zero = (oldimg[:, :, 0] == 0) & (oldimg[:, :, 1] == 0) & (oldimg[:, :, 2] == 0)
        # newimg_zero = (newimg[:, :, 0] == 0) & (newimg[:, :, 1] == 0) & (newimg[:, :, 2] == 0)
        # maskzero = oldimg_zero | newimg_zero
        # oldimg[:, :, 0][maskzero] = 0
        # oldimg[:, :, 1][maskzero] = 0
        # oldimg[:, :, 2][maskzero] = 0
        # newimg[:, :, 0][maskzero] = 0
        # newimg[:, :, 1][maskzero] = 0
        # newimg[:, :, 2][maskzero] = 0


        if self.num_classes  == 2:
            label[label < 127] = 0
            label[label >= 127] = 1


        # label[label <= 127] = 0
        # label[label  > 127 ] = 1
        # label[maskzero] = 255

        img_records = {}


        #boxes = self.anns[idx]["bb"]

        # annotations = self.anns.get(self.images[idx]["id"], [])
        # bboxes = np.array([ann['bbox'] for ann in annotations])
        #
        # point_coords = []
        # for ann in annotations:
        #     mask = maskUtils.decode(ann['segmentation'])
        #     point_coords.append(self.random_select_points(mask,20))
        # point_coords = np.array(point_coords)
        #
        #
        #
        # point_labels = np.array([ann['label'] for ann in annotations])
        # mask_inputs= np.array([ maskUtils.decode(ann['segmentation']) for ann in annotations])
        #
        #
        #
        # coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None
        # if point_coords is not None:
        #     assert (
        #             point_labels is not None
        #     ), "point_labels must be supplied if point_coords is supplied."
        #     point_coords = self.transform.apply_coords(point_coords, (H,W))
        #     coords_torch = torch.as_tensor(point_coords, dtype=torch.float)
        #     labels_torch = torch.as_tensor(point_labels, dtype=torch.int)
        #     #coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        # if bboxes is not None:
        #     box = self.transform.apply_boxes(bboxes, (H,W))
        #     box_torch = torch.as_tensor(box, dtype=torch.float)
        #     #box_torch = box_torch[None, :]
        # if mask_inputs is not None:
        #     mask_input_torch = torch.as_tensor(mask_inputs, dtype=torch.float).unsqueeze(1)


        # img_records["point_coords"] = coords_torch
        # img_records["point_labels"] = labels_torch
        # img_records["boxes"] = box_torch
        # img_records["mask_inputs"] = mask_input_torch

        # img_records["point_coords"] = None
        # img_records["point_labels"] = None
        # img_records["boxes"] = None
        # img_records["mask_inputs"] = None
        img_records = {}

        if self.aug_transform is not None:
            oldimg, newimg, label = self.aug_transform(oldimg, newimg, labelmap=label)

        label = torch.from_numpy(label)
        if self.img_transform is not None:
            if isinstance(self.img_transform, list):
                oldimg = self.img_transform[0](oldimg)
                newimg = self.img_transform[1](newimg)
            else:
                oldimg = self.img_transform(oldimg)
                newimg = self.img_transform(newimg)
        if self.label_transform is not None:
            label = self.label_transform(label)

        if isinstance(oldimg, np.ndarray):
            oldimg = torch.from_numpy(oldimg).permute(2, 0, 1)

        if isinstance(newimg, np.ndarray):
            newimg = torch.from_numpy(newimg).permute(2, 0, 1)
            # mask_patch = torch.from_numpy(mask_patch)
            # print(torch.min(labelimg),torch.max(labelimg))

        #img = torch.cat([oldimg, newimg], dim=0)


        img_records["old_path"] = oldimage_path
        img_records["new_path"] = newimage_path
        img_records["label_path"] = label_path
        #img_records["mask_inputs"] = label



        return oldimg, newimg, label, img_records
    def __len__(self):
        return len(self.images)




def readtransxml( transxml):
    path_transxml = transxml
    if not os.path.exists(path_transxml):
        Log('Error:{} is not existed.'.format(path_transxml))
    transBM = parse(path_transxml)
    root = transBM.documentElement
    all_codes = root.getElementsByTagName('BM')
    all_dict = {}
    num_class = 0
    for node in all_codes:
        class_geoid_name = node.attributes['key'].value
        class_id = node.attributes['val'].value
        all_dict[int(class_geoid_name)] = class_id
        # if int(class_id) > num_class:
        #     num_class = num_class + 1
    return num_class, all_dict

import pandas as pd
def addcdtxt2clipcaption(input_txt,transxml,output_dir):
    with open(input_txt, "r") as f:
        ids = f.readlines()
    prompt = 'changes from {} to {}'
    _, class_trans = readtransxml(transxml)

    out_dict = {
        "old_path": [],
        "new_path": [],
        "target_path": [],
        "caption": []
    }

    for line in tqdm(ids):
        old_path, new_path, label_path = line.rstrip().split()
        label_img = cv2.imdecode(np.fromfile(label_path,dtype=np.uint8),-1)

        classes = np.unique(label_img)
        caption_classes  = []
        for cls in classes:
            if cls == 0:#background
                continue
            caption_classes.append(class_trans[cls])


        prompt = 'changes from {} to {}'.format("unknown",', '.join(caption_classes))
        out_dict["old_path"].append(old_path)
        out_dict["new_path"].append(new_path)
        out_dict["target_path"].append(label_path)
        out_dict["caption"].append(prompt)
    df = pd.DataFrame(out_dict)
    df.to_csv(output_dir,index= None, encoding="utf-8")

    return






def labels_to_coco(all_paths, category_ids, output_json_path, background = 0, change_value = None):
    # Assuming 'maskUtils.encode()' returns a binary format, you might need something like this
    def convert_rle(rle):
        if type(rle) == list:
            # Already in a serializable format
            return rle
        elif 'counts' in rle and type(rle['counts']) == bytes:
            # Convert bytes to string
            rle['counts'] = rle['counts'].decode('utf-8')
        return rle
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": id, "name": name} for id, name in category_ids.items()]
    }

    annotation_id = 1

    for image_id, path in tqdm(enumerate(all_paths)):
        oldimage_path, newimage_path, label_path = path.rstrip().split()
        label_image = cv2.imdecode(np.fromfile(label_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        if label_image is None:
            continue  # Skip if the image couldn't be read

        # Add image information to COCO
        coco_data["images"].append({
            "id": image_id,
            "width": label_image.shape[1],
            "height": label_image.shape[0],
            "oldpath": oldimage_path,
            "newpath": newimage_path,
            "labelpath": label_path

        })
        pixel_values = np.unique(label_image)
        for pixel_value  in pixel_values:
            if pixel_value == background:
                continue
            category_id = category_ids[pixel_value]

            #print("val = {}, category_id = {}".format(pixel_value, category_id))
            binary_mask = np.uint8(label_image == pixel_value)

            rle = maskUtils.encode(np.asfortranarray(binary_mask))
            rle = convert_rle(rle)


            area = maskUtils.area(rle)
            bbox = maskUtils.toBbox(rle).tolist()



            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": rle,
                "bbox": bbox,
                "area": area.item(),  # Convert from numpy to Python scalar
                "iscrowd": 0
            })
            annotation_id += 1

    # Save the COCO data to a JSON file
    with open(output_json_path, 'w', encoding="utf-8") as json_file:
        json.dump(coco_data, json_file, ensure_ascii=False, indent=4)
# Assuming label_image is a 2D numpy array you have loaded or generated
# label_image = ...

# Convert to COCO format
# coco_data = label_image_to_coco(label_image)

# Save to JSON file
# with open('label_image_coco_format.json', 'w') as f:
#     json.dump(coco_data, f, indent=4)
def split_train_val(json_file, val_ratio=0.1, val_size=500):
    with open(json_file, 'r', encoding="utf-8") as file:
        annotations = json.load(file)
    images = annotations["images"]
    anns = annotations["annotations"]
    # Compute validation size
    val_size = min(val_size, int(len(images) * val_ratio))

    # Randomly select indices for validation images
    val_indices = np.random.choice(range(len(images)), val_size, replace=False)
    val_image_ids = {images[i]['id'] for i in val_indices}

    # Separate images into validation and training sets
    val_images = [img for img in images if img['id'] in val_image_ids]
    train_images = [img for img in images if img['id'] not in val_image_ids]

    # Separate annotations into validation and training sets
    val_anns = [ann for ann in anns if ann['image_id'] in val_image_ids]
    train_anns = [ann for ann in anns if ann['image_id'] not in val_image_ids]

    # Create new dictionaries for training and validation
    train_dataset = {"images": train_images, "annotations": train_anns, "categories": annotations["categories"]}
    val_dataset = {"images": val_images, "annotations": val_anns, "categories": annotations["categories"]}

    # Save the new datasets
    train_json = json_file.replace('.json', '_train.json')
    val_json = json_file.replace('.json', '_val.json')

    with open(train_json, 'w', encoding="utf-8") as f:
        json.dump(train_dataset, f, ensure_ascii=False, indent=4)

    with open(val_json, 'w', encoding="utf-8") as f:
        json.dump(val_dataset, f, ensure_ascii=False, indent=4)

    print(f"Training and validation sets have been saved to {train_json} and {val_json}, respectively.")

#import xml.etree.ElementTree as ET
if __name__ == '__main__':
    # config = r"D:\dengkai\code\clip_sam_cd\config\clip_change.json"
    # txt = r"D:\data\LEVIR-CD\test.txt"
    #
    # category_xml = r"D:\dengkai\code\dengkai_DL\configs\clipchange\trans_type.xml"
    #
    # category_ids = readtransxml(category_xml)[1]
    #
    #
    # with open(txt,"r",encoding="utf-8" ) as f:
    #     lines = f.readlines()
    #
    # labels_to_coco(lines,category_ids,r"D:\data\LEVIR-CD\liver_cd_test.json")
    json_file = r"F:\data\02_1M2M_BHYB\1M2M_BHYB_linux.json"


    split_train_val(json_file)
    #
    #
    # pass


    # from utils.tools.configer import Configer
    # from dataset.tools.cv2_aug_transform_chg import CV2AugCompose_CHG
    # config = Configer(configs=config)
    #
    # transforms = CV2AugCompose_CHG(config, split="train")
    # json_file = r"F:\data\02_1M2M_BHYB\1M2M_BHYB.json"
    #
    #
    #
    # dataset = SamClipCD_dataset(json_file, config, aug_transform= transforms)
    #
    # dataloader = DataLoader(dataset,
    #                         batch_size=1,
    #                         shuffle= False)
    #
    # iter = iter(dataloader)
    #
    # data = iter.next()

    pass
