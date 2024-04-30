# ----------------------------------------------------#
#   获取测试集的detection-result和images-optional
#   具体视频教程可查看
#   https://www.bilibili.com/video/BV1zE411u7Vw
# ----------------------------------------------------#
import os

import torch
from PIL import Image
from tqdm import tqdm

import time
import json
import matplotlib.pyplot as plt
# from train_res50_fpn import create_model
from train_mobilenetv2 import create_model
from torchvision import transforms
from draw_box_utils import draw_objs
import cv2

'''
这里设置的门限值较低是因为计算map需要用到不同门限条件下的Recall和Precision值。
所以只有保留的框足够多，计算的map才会更精确，详情可以了解map的原理。
计算map时输出的Recall和Precision值指的是门限为0.5时的Recall和Precision值。

此处获得的./input/detection-results/里面的txt的框的数量会比直接predict多一些，这是因为这里的门限低，
目的是为了计算不同门限条件下的Recall和Precision值，从而实现map的计算。

这里的self.iou指的是非极大抑制所用到的iou，具体的可以了解非极大抑制的原理，
如果低分框与高分框的iou大于这里设定的self.iou，那么该低分框将会被剔除。

可能有些同学知道有0.5和0.5:0.95的mAP，这里的self.iou=0.5不代表mAP0.5。
如果想要设定mAP0.x，比如设定mAP0.75，可以去get_map.py设定MINOVERLAP。
'''


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


# get devices
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

# create model
model = create_model(num_classes=21)  # 4

# load train weights
train_weights = "./save_weights/mobile-model-24.pth"
assert os.path.exists(train_weights), "{} file dose not exist.".format(train_weights)
model.load_state_dict(torch.load(train_weights, map_location='cpu')["model"])
model.to(device)

# read class_indict
label_json_path = './pascal_voc_classes.json'
assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
with open(label_json_path, 'r') as f:
    class_dict = json.load(f)

category_index = {v: k for k, v in class_dict.items()}

image_ids = open('./VOCdevkit/VOC2007/ImageSets/test.txt').read().strip().split()

if not os.path.exists("./input"):
    os.makedirs("./input")
if not os.path.exists("./input/detection-results"):
    os.makedirs("./input/detection-results")
if not os.path.exists("./input/images-optional"):
    os.makedirs("./input/images-optional")
os.makedirs("./input/show", exist_ok=True)

for image_id in tqdm(image_ids):
    f = open("./input/detection-results/" + image_id + ".txt", "w")
    image_path = "./VOCdevkit/VOC2007/JPEGImages/" + image_id + ".jpg"
    original_img = Image.open(image_path)
    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        predictions = model(img.to(device))[0]
        t_end = time_synchronized()
        print("inference+NMS time: {}".format(t_end - t_start))

        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()

        if len(predict_boxes) == 0:
            print("没有检测到任何目标!")

        plot_img = draw_objs(original_img,
                             predict_boxes,
                             predict_classes,
                             predict_scores,
                             category_index=category_index,
                             box_thresh=0.5,
                             line_thickness=3,
                             font='arial.ttf',
                             font_size=20)

        # 保存预测的图片结果
        for i in range(len(predict_boxes)):
            area = (predict_boxes[i][2] - predict_boxes[i][0]) * (predict_boxes[i][3] - predict_boxes[i][1])
            f.write("%s %s %s %s %s %s %s\n" % (
                category_index[predict_classes[i]], str(predict_scores[i]), str(int(predict_boxes[i][0])),
                str(int(predict_boxes[i][1])), str(int(predict_boxes[i][2])), str(int(predict_boxes[i][3])),
                str(int(area))))
        if plot_img is None:
            continue
        plt.imshow(plot_img)
        plt.show()
        plot_img.save(os.path.join("./input/show", image_id + ".jpg"))

print("Conversion completed!")
