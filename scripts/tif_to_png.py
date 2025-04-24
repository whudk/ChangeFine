import os
from PIL import Image

input_folder = "/home/xiaoshan/wsy/Dataset/WHU-CD/test/label"
output_folder = "/home/xiaoshan/wsy/Dataset/WHU-CD/test/label1"



# 遍历输入文件夹中的所有文件
for filename in os.listdir(input_folder):
    if filename.endswith('.tif'):
        # 构建完整的文件路径
        input_path = os.path.join(input_folder, filename)
        # 打开TIF图像
        with Image.open(input_path) as img:
            # 将灰度图像转换为RGB模式
            img_rgb = img.convert('RGB')
            # 构建输出文件的路径
            output_path = os.path.join(output_folder, f'{os.path.splitext(filename)[0]}.jpg')
            # 保存为JPG格式
            img_rgb.save(output_path, 'JPEG')

        print(f'已转换: {filename} -> {os.path.basename(output_path)}')

print("所有文件已转换完成。")

