from torch.utils.data import Dataset
import os
import torch
import json
from PIL import Image
from lxml import etree
from xml.etree import ElementTree as ET
import matplotlib.pyplot as plt
import transforms
import torchvision.transforms as ts
import random
from PIL import ImageDraw
from torchvision.transforms.functional import to_pil_image, to_tensor


class VOCDataSet(Dataset):
    """读取解析PASCAL VOC2007/2012数据集"""

    def __init__(self, voc_root, year="2007", transforms=None, txt_name: str = "train.txt"):
        assert year in ["2007", "2012"], "year must be in ['2007', '2012']"
        self.root = os.path.join(voc_root, "VOCdevkit", f"VOC{year}")
        self.img_root = os.path.join(self.root, "JPEGImages")
        self.annotations_root = os.path.join(self.root, "Annotations")

        # read train.txt or val.txt file
        txt_path = os.path.join(self.root, "ImageSets", "Main", txt_name)
        # print("Expected path to train.txt:", txt_path)
        # print("Current working directory:", os.getcwd())
        assert os.path.exists(txt_path), "not found {} file.".format(txt_name)

        with open(txt_path) as read:
            self.xml_list = [os.path.join(self.annotations_root, line.strip() + ".xml")
                             for line in read.readlines() if len(line.strip()) > 0]

        # check file
        assert len(self.xml_list) > 0, "in '{}' file does not find any information.".format(txt_path)
        for xml_path in self.xml_list:
            assert os.path.exists(xml_path), "not found '{}' file.".format(xml_path)

        # read class_indict
        json_file = './pascal_voc_classes.json'
        assert os.path.exists(json_file), "{} file not exist.".format(json_file)
        with open(json_file, 'r') as f:
            self.class_dict = json.load(f)

        self.transforms = transforms

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, idx):
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = ET.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        img_path = os.path.join(self.img_root, xml_path.replace("xml", "jpg").replace("Annotations", "JPEGImages"))
        image = Image.open(img_path)
        if image.format != "JPEG":
            raise ValueError("Image '{}' format not JPEG".format(img_path))

        boxes = []
        labels = []
        iscrowd = []
        assert "object" in data, "{} lack of object information.".format(xml_path)
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])

            # 进一步检查数据，有的标注信息中可能有w或h为0的情况，这样的数据会导致计算回归loss为nan
            if xmax <= xmin or ymax <= ymin:
                print("Warning: in '{}' xml, there are some bbox w/h <=0".format(xml_path))
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])
            if "difficult" in obj:
                iscrowd.append(int(obj["difficult"]))
            else:
                iscrowd.append(0)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def get_height_and_width(self, idx):
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        return data_height, data_width

    def parse_xml_to_dict(self, xml):
        """
        将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
        Args:
            xml: xml tree obtained by parsing XML file contents using lxml.etree

        Returns:
            Python dictionary holding XML contents.
        """

        if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)  # 递归遍历标签信息
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

    def coco_index(self, idx):
        """
        该方法是专门为pycocotools统计标签信息准备，不对图像和标签作任何处理
        由于不用去读取图片，可大幅缩减统计时间

        Args:
            idx: 输入需要获取图像的索引
        """
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        # img_path = os.path.join(self.img_root, data["filename"])
        # image = Image.open(img_path)
        # if image.format != "JPEG":
        #     raise ValueError("Image format not JPEG")
        boxes = []
        labels = []
        iscrowd = []
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])
            iscrowd.append(int(obj["difficult"]))

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return (data_height, data_width), target

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))



# read class_indict

# category_index = {}
# try:
#     json_file = open('./pascal_voc_classes.json', 'r')
#     class_dict = json.load(json_file)
#     category_index = {v: k for k, v in class_dict.items()}
# except Exception as e:
#     print(e)
#     exit(-1)
#
# data_transform = {
#     "train": transforms.Compose([transforms.ToTensor(),
#                                  transforms.RandomHorizontalFlip(0.5),
#                                  transforms.ColorDistortion(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)]),
#     "val": transforms.Compose([transforms.ToTensor()]),
#     "test": transforms.Compose([transforms.ToTensor()])  # 假设测试集的转换和验证集相同
# }

# voc_root = "F:/Faster R-CNN"
# #
# # load train data set
# train_data_set = VOCDataSet(voc_root, "2007", data_transform["train"], "train.txt")
# val_data_set = VOCDataSet(voc_root, "2007", data_transform["val"], "val.txt")
# test_data_set = VOCDataSet(voc_root, "2007", data_transform["test"], "test.txt")
# print(len(train_data_set))
# print(len(val_data_set))
# print(len(test_data_set))
#
#
# def select_random_images(dataset, num_images):
#     """
#     随机选择指定数量的图像索引。
#
#     Args:
#         dataset (Dataset): 数据集对象，应支持 len() 函数获取长度。
#         num_images (int): 需要随机选择的图像数量。
#
#     Returns:
#         list: 包含随机选中的图像索引的列表。
#     """
#     # 检查请求的图像数量是否超出了数据集的范围
#     if num_images > len(dataset):
#         raise ValueError("Requested more images than are available in the dataset")
#
#     # 生成所有可能的索引
#     indices = list(range(len(dataset)))
#
#     # 从所有索引中随机选择指定数量的索引
#     selected_indices = random.sample(indices, num_images)
#
#     return selected_indices
#
#
# def show_images(dataset, indices, transform):
#     fig, axs = plt.subplots(nrows=len(indices), ncols=2, figsize=(15, 5 * len(indices)))
#
#     for i, idx in enumerate(indices):
#         image, target = dataset[idx]  # 从数据集获取原始图像和目标
#
#         # 显示原始图像
#         if isinstance(image, torch.Tensor):
#             image_pil = to_pil_image(image)  # 如果是张量，则转换为PIL图像
#         else:
#             image_pil = image  # 如果已经是PIL图像，直接使用
#
#         # 应用变换，并传递图像和目标
#         transformed_image, transformed_target = transform(image_pil.copy(), target.copy())  # 应用预设的变换
#
#         # 转换为可显示的格式
#         if isinstance(transformed_image, torch.Tensor):
#             transformed_image = to_pil_image(transformed_image)  # 如果变换返回张量，转换为PIL用于显示
#
#         # 绘制原始图像
#         axs[i, 0].imshow(image_pil)
#         axs[i, 0].set_title('Original Image')
#         axs[i, 0].axis('off')
#
#         # 绘制变换后的图像
#         axs[i, 1].imshow(transformed_image)
#         axs[i, 1].set_title('Transformed Image')
#         axs[i, 1].axis('off')
#
#     plt.tight_layout()
#     plt.show()
#
# indices = select_random_images(train_data_set, 3)  # 随机选择5个图像
# show_images(train_data_set, indices, data_transform['train'])  # 使用训练集的变换

# for index in random.sample(range(0, len(train_data_set)), k=5):
#     img, target = train_data_set[index]
#     img = ts.ToPILImage()(img)
#     draw_box(img,
#              target["boxes"].numpy(),
#              target["labels"].numpy(),
#              [1 for i in range(len(target["labels"].numpy()))],
#              category_index,
#              thresh=0.5,
#              line_thickness=5)
#     plt.imshow(img)
#     plt.show()

# 从指定目录随机选择图像文件
