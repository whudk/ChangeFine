import os
from PIL import Image

old_input_folder = '/home/xiaoshan/wsy/Dataset/DSIFN-CD/val/t1'
new_input_folder = '/home/xiaoshan/wsy/Dataset/DSIFN-CD/val/t2'
label_input_folder = '/home/xiaoshan/wsy/Dataset/DSIFN-CD/val/mask_512'

old_output_folder = '/home/xiaoshan/wsy/Dataset/DSIFN-CD/val/A'
new_output_folder = '/home/xiaoshan/wsy/Dataset/DSIFN-CD/val/B'
label_output_folder = '/home/xiaoshan/wsy/Dataset/DSIFN-CD/val/label'

crop_size = (256,256)
overlap = 0

for file_name in sorted(os.listdir(old_input_folder)):
    old_img_path = os.path.join(old_input_folder, file_name)
    new_img_path = os.path.join(new_input_folder, file_name)
    label_img_path = os.path.join(label_input_folder, file_name)

    olg_img = Image.open(old_img_path)
    new_img = Image.open(new_img_path)
    label_img = Image.open(label_img_path)
    img_width, img_height = olg_img.size


    for x in range(0, img_width - crop_size[0] + 1, crop_size[0] - overlap):
        for y in range(0, img_height - crop_size[1] + 1, crop_size[1] - overlap):
            box = (x, y, x + crop_size[0], y + crop_size[1])

            old_cropped_img = olg_img.crop(box)
            new_cropped_img = new_img.crop(box)
            label_cropped_img = label_img.crop(box)

            old_cropped_filename = f'{os.path.splitext(file_name)[0]}_crop_{x}_{y}.png'
            new_cropped_filename = f'{os.path.splitext(file_name)[0]}_crop_{x}_{y}.png'
            label_cropped_filename = f'{os.path.splitext(file_name)[0]}_crop_{x}_{y}.png'

            old_cropped_img.save(os.path.join(old_output_folder, old_cropped_filename))
            new_cropped_img.save(os.path.join(new_output_folder, new_cropped_filename))
            label_cropped_img.save(os.path.join(label_output_folder, label_cropped_filename))