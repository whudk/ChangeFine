import os
import json

def generate_json(old_dir,new_dir,label_dir,json_output_path):
    # 获取文件名字，并排序
    old_files_list = sorted(os.listdir(old_dir))
    new_files_list = sorted(os.listdir(new_dir))
    label_files_list = sorted(os.listdir(label_dir))

    #每个文件夹里的文件数量相同且一一对应
    images = []
    count = 0

    #遍历文件夹，生成字典
    for old_file, new_file, label_file in zip(old_files_list,new_files_list,label_files_list):
        oldpath = os.path.join(old_dir,old_file)
        newpath = os.path.join(new_dir,new_file)
        labelpath = os.path.join(label_dir,label_file)

        images.append({
            "id":count,
            "oldpath":oldpath,
            "newpath":newpath,
            "labelpath":labelpath
        })
        output_data = {"images":images}
        with open(json_output_path,'w') as json_file:
            json.dump(output_data,json_file,indent=4)
        count += 1

old_dir = '/home/xiaoshan/wsy/Dataset/DSIFN-CD/train/A'
new_dir = '/home/xiaoshan/wsy/Dataset/DSIFN-CD/train/B'
label_dir = '/home/xiaoshan/wsy/Dataset/DSIFN-CD/train/label'
json_output_path = '/home/xiaoshan/wsy/clip_sam_cd/config/DSIFN_train.json'
generate_json(old_dir,new_dir,label_dir,json_output_path)