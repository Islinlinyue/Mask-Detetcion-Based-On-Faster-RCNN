import os
import re

# 指定图片所在的文件夹路径
folder_path = '/root/autodl-tmp/project/VOCdevkit/VOC2007/JPEGImages'

# 从文件名中提取数字的函数
def extract_number(filename):
    # 假设数字前的字符串是 'masksadad'
    prefix_length = len('maksssksksss')
    # 截取字符串开始至文件扩展名前的部分作为数字
    match = re.search(r'\d+', filename[prefix_length:])
    if match:
        return int(match.group(0))
    return None

# 读取目录下所有文件名
files = os.listdir(folder_path)
# 过滤出所有JPEG图片文件，并按数字排序
files = [f for f in files if f.endswith('.jpg') or f.endswith('.jpeg')]
files.sort(key=extract_number)

# 输出排序后的文件名列表
for file in files:
    print(file)
