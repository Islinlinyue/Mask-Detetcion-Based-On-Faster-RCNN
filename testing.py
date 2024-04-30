import os
import torchvision.transforms as transforms
from PIL import Image
import json
from train_res50_fpn import create_model
from draw_box_utils import draw_objs
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
from pytorch_grad_cam import AblationCAM, EigenCAM
from pytorch_grad_cam.ablation_layer import AblationLayerFasterRCNN
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.reshape_transforms import fasterrcnn_reshape_transform
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_accross_batch_and_channels, scale_cam_image


# get devices
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

# create model
model = create_model(num_classes=4) # 4

# load train weights
train_weights = "./save_weights/resNetFpn-model-24.pth"
assert os.path.exists(train_weights), "{} file dose not exist.".format(train_weights)
model.load_state_dict(torch.load(train_weights, map_location='cpu')["model"])
model.to(device)

# read class_indict
label_json_path = './pascal_voc_classes.json'
assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
with open(label_json_path, 'r') as f:
    class_dict = json.load(f)

category_index = {v: k for k, v in class_dict.items()}


model.eval()  # 进入验证模式
with torch.no_grad():
    # init
    # image_ids = 9
    # image_id = tqdm(image_ids)
    image_path = "test9.jpg"
    original_img = Image.open(image_path)
    image_float_np = np.float32(original_img) / 255
    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    # expand batch dimension
    img = img.to(device)
    img = torch.unsqueeze(img, dim=0)

    img_height, img_width = img.shape[-2:]
    init_img = torch.zeros((1, 3, img_height, img_width), device=device)
    model(init_img)

    predictions = model(img.to(device))[0]
    print(predictions)

    predict_boxes = predictions["boxes"].to("cpu").numpy()
    predict_classes = predictions["labels"].to("cpu").numpy()
    predict_scores = predictions["scores"].to("cpu").numpy()

    if len(predict_boxes) == 0:
        print("没有检测到任何目标!")

    # plot_img = draw_objs(original_img,
    #                      predict_boxes,
    #                      predict_classes,
    #                      predict_scores,
    #                      category_index=category_index,
    #                      box_thresh=0.5,
    #                      line_thickness=3,
    #                      font='arial.ttf',
    #                      font_size=20)
    #
    # plt.imshow(plot_img)
    # plt.show()

target_layers = [model.backbone]
targets = [FasterRCNNBoxScoreTarget(labels=predict_classes, bounding_boxes=predict_boxes)]
cam = EigenCAM(model,
               target_layers,
               reshape_transform=fasterrcnn_reshape_transform)

grayscale_cam = cam(img, targets=targets)


grayscale_cam = grayscale_cam[0, :]
cam_image = show_cam_on_image(image_float_np, grayscale_cam, use_rgb=True)
cam_image_pil = Image.fromarray(cam_image.astype('uint8'), 'RGB')

# 现在 cam_image_pil 是一个 PIL.Image 对象，可以用在 ImageDraw.Draw 中
plot_img = draw_objs(cam_image_pil,
                     predict_boxes,
                     predict_classes,
                     predict_scores,
                     category_index=category_index,
                     box_thresh=0.5,
                     line_thickness=3,
                     font='arial.ttf',
                     font_size=20)
#
# 由于 matplotlib 不能直接显示 PIL.Image 对象，我们需要先将它转换为 numpy.ndarray
plot_img_np = np.array(plot_img)

# 使用 matplotlib 展示图像
plt.imshow(plot_img_np)
plt.show()

folder_path = "./gradCam"

path_cam_img = os.path.join('./gradCam', "T3.jpg")
cv2.imwrite(path_cam_img, plot_img_np)
