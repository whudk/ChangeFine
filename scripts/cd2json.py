import sys

import cv2
from tqdm import  tqdm
import numpy as np
from pycocotools import mask as maskUtils
import json
import argparse
import os
from xml.dom.minidom import parse
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
        "categories": 'unknown' if category_ids is None else [{"id": id, "name": name} for id, name in category_ids.items()]
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
            if category_ids is not None:
                if pixel_value not in category_ids.keys():
                    category_id = 'unknown'
                else:
                    category_id = category_ids[pixel_value]
            else:
                category_id = 'unknown'
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
import json
import numpy as np




def split_dataset(json_file, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    # 确保比例之和为1
    #assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum up to 1."

    with open(json_file, 'r', encoding="utf-8") as file:
        annotations = json.load(file)

    images = annotations["images"]
    anns = annotations["annotations"]

    # 计算验证集和测试集的大小
    total_images = len(images)
    val_size = int(total_images * val_ratio)
    test_size = int(total_images * test_ratio)
    train_size = total_images - val_size - test_size

    # 随机选择索引用于分割数据集
    all_indices = np.random.permutation(range(len(images)))
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:train_size + val_size]
    test_indices = all_indices[train_size + val_size:]

    def extract_subset(indices):
        subset_image_ids = {images[i]['id'] for i in indices}
        subset_images = [img for img in images if img['id'] in subset_image_ids]
        subset_anns = [ann for ann in anns if ann['image_id'] in subset_image_ids]
        return {"images": subset_images, "annotations": subset_anns, "categories": annotations["categories"]}

    # 创建新的数据集字典
    train_dataset = extract_subset(train_indices)
    val_dataset = extract_subset(val_indices)
    test_dataset = extract_subset(test_indices)

    # 保存新的数据集
    def save_dataset(dataset, filename):
        with open(filename, 'w', encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)

    save_dataset(train_dataset, json_file.replace('.json', '_train.json'))
    if val_ratio > 0.0:
        save_dataset(val_dataset, json_file.replace('.json', '_val.json'))
    if test_ratio > 0.0:
        save_dataset(test_dataset, json_file.replace('.json', '_test.json'))

    print("Datasets have been successfully split and saved.")


# 示例调用



def split_train_val(json_file, val_ratio=0.1, val_size=1584):
    with open(json_file, 'r', encoding="utf-8") as file:
        annotations = json.load(file)
    images = annotations["images"]
    anns = annotations["annotations"]
    # Compute validation size
    val_size = max(val_size, int(len(images) * val_ratio))

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
def readtransxml( transxml):
    path_transxml = transxml
    if not os.path.exists(path_transxml):
        print('Error:{} is not existed.'.format(path_transxml))
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
#import xml.etree.ElementTree as ET
import glob
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt', default="", type=str, required=False , help='The file of the hyper parameters.')
    parser.add_argument('--dir', default="", type=str, required=False, help='The file of the hyper parameters.')
    parser.add_argument('--category_xml', default='', type=str, required=False, help='The category of changetype.')
    parser.add_argument('--output_json', default="", required= False, type = str,help = "the output path of convert json path")
    parser.add_argument('--train_ratio', default=1.0, type=float, help='The ratio of train dataset.')
    parser.add_argument('--val_ratio', default=0.0, type=float, help='The ratio of val dataset.')
    parser.add_argument('--test_ratio', default=0.0, type=float, help='The ratio of test dataset.')
    args = parser.parse_args()

    #split_dataset(args.output_json, train_ratio=args.train_ratio, val_ratio=args.val_ratio, test_ratio=args.test_ratio)

    pretifs = glob.glob(os.path.join(dir, "A\*.png"))
    curtifs = glob.glob(os.path.join(dir, "B\*.png"))
    lbltifs = glob.glob(os.path.join(dir, "label\*.png"))



    lines = []

    for (oldpath, newpath, curcls) in zip(pretifs,curtifs, lbltifs):
        lines.append(oldpath + "\t" + newpath + "\t" + curcls)
    # with open(args.txt, 'w', encoding="utf-8") as f:
    #     for line in lines:
    #         f.writelines(line)
    #         f.writelines("\n")
    # sys.exit(-1)

    category_ids = None
    if os.path.isfile(args.category_xml):
        category_ids = readtransxml(args.category_xml)[1]
    labels_to_coco(lines, category_ids, args.output_json)

    split_dataset(args.output_json, train_ratio=args.train_ratio, val_ratio=args.val_ratio, test_ratio=args.test_ratio)
    # split_train_val(args.output_json, val_ratio=0.1, val_size=500)

    print("convert scueessful.")
