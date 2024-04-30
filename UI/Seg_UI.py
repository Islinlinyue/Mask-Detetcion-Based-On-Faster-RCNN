# -*- coding: utf-8 -*-

import os
import time

import cv2
import sys
import argparse
# import qtawesome
import torch
from os import getcwd
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore, QtGui, QtWidgets

from train_res50_fpn import create_model as fmodel
from train_mobilenetv2 import create_model as mmodel
from train_vgg import create_model as vmodel
from train_resnet import create_model as rmodel
import random
from torchvision import transforms
import json
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 8888888888888888


class XX(QMainWindow):
    def __init__(self):
        super().__init__()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.m_flag = True
            self.m_Position = event.globalPos() - self.pos()  # 获取鼠标相对窗口的位置
            event.accept()
            self.setCursor(QCursor(Qt.OpenHandCursor))  # 更改鼠标图标

    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self.m_flag:
            self.move(QMouseEvent.globalPos() - self.m_Position)  # 更改窗口位置
            QMouseEvent.accept()

    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False
        self.setCursor(QCursor(Qt.ArrowCursor))


class Ui_MainWindow(object):

    def __init__(self, MainWindow):

        self.path = getcwd()
        self.save_dir = "./save"
        self.save_image = None
        self.save_txt = None
        self.CAM_NUM = 0

        self.timer_camera = QtCore.QTimer()  # 定时器
        self.timer_video = QtCore.QTimer()  # 定时器

        self.setupUi(MainWindow)
        self.retranslateUi(MainWindow)
        self.wind = MainWindow
        self.slot_init()  # 槽函数设置

        # Load a model
        self.model = fmodel(num_classes=4)  # 4
        # load train weights
        train_weights = "./save_weights/resNetFpn-model-24.pth"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        assert os.path.exists(train_weights), "{} file dose not exist.".format(train_weights)
        self.model.load_state_dict(torch.load(train_weights, map_location='cpu')["model"])
        self.model.to(self.device)
        # read class_indict
        self.label_json_path = './pascal_voc_classes.json'
        assert os.path.exists(self.label_json_path), "json file {} dose not exist.".format(self.label_json_path)
        with open(self.label_json_path, 'r') as f:
            class_dict = json.load(f)
        self.category_index = {v: k for k, v in class_dict.items()}

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowModality(QtCore.Qt.NonModal)
        MainWindow.resize(1800, 900)
        MainWindow.setMinimumSize(QtCore.QSize(1800, 900))
        MainWindow.setMaximumSize(QtCore.QSize(1800, 900))
        font = QtGui.QFont()
        font.setFamily("楷体")
        MainWindow.setFont(font)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("./UI/images_test/icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setToolTip("")
        MainWindow.setAutoFillBackground(False)
        MainWindow.setStyleSheet("")
        MainWindow.setTabShape(QtWidgets.QTabWidget.Rounded)
        # MainWindow.setWindowOpacity(0.9) # 设置窗口透明度
        MainWindow.setWindowFlag(QtCore.Qt.FramelessWindowHint)  # 隐藏边框
        pe = QPalette()
        MainWindow.setAutoFillBackground(True)
        pe.setColor(QPalette.Window, Qt.lightGray)  # 设置背景色
        # pe.setColor(QPalette.Background,Qt.blue)
        MainWindow.setPalette(pe)
        MainWindow.setWindowTitle("Mask detection system")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("./UI/images_test/icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)

        MainWindow.setWindowIcon(icon)
        # MainWindow.setWindowIcon(QIcon('Amg.jpg'))  # 设置图标

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.plainTextEdit_result_display = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.plainTextEdit_result_display.setGeometry(QtCore.QRect(1550, 150, 200, 700))
        self.plainTextEdit_result_display.setStyleSheet('background-color: slategray;border-radius: 10px; color: white; border: 3.14px slategray;border-style: outset;')
        self.plainTextEdit_result_display.setLineWidth(-1)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.plainTextEdit_result_display.setFont(font)
        self.plainTextEdit_result_display.setObjectName("plainTextEdit_result_display")
        self.input_img = QtWidgets.QLabel(self.centralwidget)
        self.input_img.setGeometry(QtCore.QRect(50, 150, 700, 700))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        font.setBold(False)
        font.setWeight(50)
        self.input_img.setFont(font)
        self.input_img.setStyleSheet('background-color: gray;border-radius: 10px; border: 3.14px gray;border-style: outset;')
        self.input_img.setObjectName("input_img")
        self.input_img.setScaledContents(True)
        self.output_img = QtWidgets.QLabel(self.centralwidget)
        self.output_img.setGeometry(QtCore.QRect(800, 150, 700, 700))
        self.output_img.setScaledContents(True)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        font.setBold(False)
        font.setWeight(50)
        self.output_img.setFont(font)
        self.output_img.setStyleSheet(
            'background-color: gray;border-radius: 10px; border: 3.14px gray;border-style: outset;')
        self.output_img.setObjectName("output_img")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(10, 940, 1701, 20))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.title = QtWidgets.QLabel(self.centralwidget)
        self.title.setGeometry(QtCore.QRect(20, 10, 581, 31))
        font = QtGui.QFont()
        font.setFamily("楷体")
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.title.setFont(font)
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet('background-color: tomato;border-radius: 10px; color: white; border-style: outset;')
        self.title.setObjectName("title")
        self.background = QtWidgets.QLabel(self.centralwidget)
        self.background.setGeometry(QtCore.QRect(10, 120, 1780, 770))
        self.background.setFrameShape(QtWidgets.QFrame.Box)
        self.background.setAlignment(QtCore.Qt.AlignCenter)
        self.background.setStyleSheet('border-radius: 10px; border: 3.14px azure;border-style: outset;')
        self.background.setObjectName("background")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(30, 60, 1131, 47))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.toolButton_modelselect = QtWidgets.QToolButton(self.layoutWidget)
        self.toolButton_modelselect.setMinimumSize(QtCore.QSize(45, 45))
        self.toolButton_modelselect.setMaximumSize(QtCore.QSize(50, 45))
        self.toolButton_modelselect.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.toolButton_modelselect.setAutoFillBackground(False)
        self.toolButton_modelselect.setStyleSheet("background-color: transparent; border-image: url(./UI/images_test/model_select.png); border-style: outset;")
        self.toolButton_modelselect.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/newPrefix/images_test/folder_web.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_modelselect.setIcon(icon1)
        self.toolButton_modelselect.setIconSize(QtCore.QSize(50, 40))
        self.toolButton_modelselect.setPopupMode(QtWidgets.QToolButton.DelayedPopup)
        self.toolButton_modelselect.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.toolButton_modelselect.setAutoRaise(False)
        self.toolButton_modelselect.setArrowType(QtCore.Qt.NoArrow)
        self.toolButton_modelselect.setObjectName("toolButton_modelselect")
        self.horizontalLayout.addWidget(self.toolButton_modelselect)
        self.model_select = QtWidgets.QComboBox(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.model_select.setFont(font)
        self.model_select.setObjectName("model_select")
        self.model_select.addItem("")
        self.model_select.addItem("")
        self.model_select.addItem("")
        self.model_select.addItem("")
        self.model_select.addItem("")
        self.model_select.setStyleSheet('background-color: firebrick;border-radius: 10px; color:white; border: 3.14px firebrick;  border-style: outset;')
        self.horizontalLayout.addWidget(self.model_select)
        self.toolButton_filelabel = QtWidgets.QToolButton(self.layoutWidget)
        self.toolButton_filelabel.setMinimumSize(QtCore.QSize(45, 45))
        self.toolButton_filelabel.setMaximumSize(QtCore.QSize(50, 45))
        self.toolButton_filelabel.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.toolButton_filelabel.setAutoFillBackground(False)
        self.toolButton_filelabel.setStyleSheet("background-color: transparent;\n"
"border-image: url(./UI/images_test/file.png);")
        self.toolButton_filelabel.setText("")
        self.toolButton_filelabel.setIcon(icon1)
        self.toolButton_filelabel.setIconSize(QtCore.QSize(50, 40))
        self.toolButton_filelabel.setPopupMode(QtWidgets.QToolButton.DelayedPopup)
        self.toolButton_filelabel.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.toolButton_filelabel.setAutoRaise(False)
        self.toolButton_filelabel.setArrowType(QtCore.Qt.NoArrow)
        self.toolButton_filelabel.setObjectName("toolButton_filelabel")
        self.horizontalLayout.addWidget(self.toolButton_filelabel)
        self.button_openfile = QtWidgets.QPushButton(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.button_openfile.setFont(font)
        self.button_openfile.setStyleSheet('background-color: deepskyblue;border-radius: 10px; color:white;  border: 3.14px deepskyblue;border-style: outset;')
        self.button_openfile.setObjectName("button_openfile")
        self.horizontalLayout.addWidget(self.button_openfile)
        self.toolButton_videolabel = QtWidgets.QToolButton(self.layoutWidget)
        self.toolButton_videolabel.setMaximumSize(QtCore.QSize(50, 45))
        self.toolButton_videolabel.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.toolButton_videolabel.setAutoFillBackground(False)
        self.toolButton_videolabel.setStyleSheet("background-color: transparent; border-image: url(./UI/images_test/video.png); border-style: outset;")
        self.toolButton_videolabel.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/newPrefix/images_test/author.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_videolabel.setIcon(icon2)
        self.toolButton_videolabel.setIconSize(QtCore.QSize(50, 39))
        self.toolButton_videolabel.setPopupMode(QtWidgets.QToolButton.DelayedPopup)
        self.toolButton_videolabel.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.toolButton_videolabel.setAutoRaise(False)
        self.toolButton_videolabel.setArrowType(QtCore.Qt.NoArrow)
        self.toolButton_videolabel.setObjectName("toolButton_videolabel")
        self.horizontalLayout.addWidget(self.toolButton_videolabel)
        self.button_openvideo = QtWidgets.QPushButton(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.button_openvideo.setFont(font)
        self.button_openvideo.setStyleSheet('background-color: limegreen;border-radius: 10px; color:white;  border: 3.14px limegreen;border-style: outset;')
        self.button_openvideo.setObjectName("button_openvideo")
        self.horizontalLayout.addWidget(self.button_openvideo)

        self.layoutWidget1 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget1.setGeometry(QtCore.QRect(1720, 10, 60, 31))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.layoutWidget1)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.min = QtWidgets.QPushButton(self.layoutWidget1)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.min.setFont(font)
        self.min.setStyleSheet('background-color: #6DDF6D;border-radius: 10px; border: 3.14px green;border-style: outset;')

        self.min.setText("")
        self.min.setObjectName("min")
        self.horizontalLayout_2.addWidget(self.min)

        self.quit = QtWidgets.QPushButton(self.layoutWidget1)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.quit.setFont(font)
        self.quit.setStyleSheet('background-color: #F76677;border-radius: 10px; border: 3.14px red;border-style: outset;')
        self.quit.setText("")
        self.quit.setObjectName("quit")
        self.horizontalLayout_2.addWidget(self.quit)
        MainWindow.setCentralWidget(self.centralwidget)
        self.actionGoogle_Translate = QtWidgets.QAction(MainWindow)
        self.actionGoogle_Translate.setObjectName("actionGoogle_Translate")
        self.actionHTML_type = QtWidgets.QAction(MainWindow)
        self.actionHTML_type.setObjectName("actionHTML_type")
        self.actionsoftware_version = QtWidgets.QAction(MainWindow)
        self.actionsoftware_version.setObjectName("actionsoftware_version")

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Mask detection system"))
        # self.input_img.setText(_translate("MainWindow", "输入图像"))
        # self.output_img.setText(_translate("MainWindow", "输出图像"))
        self.title.setText(_translate("MainWindow", "Mask detection system"))
        # self.background.setText(_translate("MainWindow", "TextLabel"))
        self.model_select.setItemText(0, _translate("MainWindow", "Model Choose"))
        self.model_select.setItemText(1, _translate("MainWindow", "FPN"))
        self.model_select.setItemText(2, _translate("MainWindow", "Resnet"))
        self.model_select.setItemText(3, _translate("MainWindow", "VGG"))
        self.model_select.setItemText(4, _translate("MainWindow", "Mobilenetv2"))
        self.button_openfile.setText(_translate("MainWindow", "Image Recognition"))
        self.button_openvideo.setText(_translate("MainWindow", "Video Recognition"))

        self.actionGoogle_Translate.setText(_translate("MainWindow", "Google Translate"))
        self.actionHTML_type.setText(_translate("MainWindow", "HTML type"))
        self.actionsoftware_version.setText(_translate("MainWindow", "software version"))


    def slot_init(self):  # 定义槽函数
        # 最小化
        self.min.clicked.connect(self.on_pushButton_min_clicked)
        # 退出
        self.quit.clicked.connect(self.on_pushButton_close_clicked)
        # 模型切换
        self.model_select.currentIndexChanged.connect(self.model_select_obj)
        # 打开图片
        self.button_openfile.clicked.connect(self.choose_file)
        # 打开视频
        self.button_openvideo.clicked.connect(self.open_video)
        self.timer_video.timeout.connect(self.show_video)

    def on_pushButton_min_clicked(self):
        self.wind.showMinimized()
    def on_pushButton_close_clicked(self):
        self.wind.close()

    def model_select_obj(self):
        self.plainTextEdit_result_display.clear()
        self.model_choose = self.model_select.currentText()
        print("功能切换为 :", self.model_choose)
        self.plainTextEdit_result_display.insertPlainText("Function switches to :%s\n" % self.model_choose)
        if self.model_choose == "Resnet":
            self.model = rmodel(num_classes=4)  # 4
            # load train weights
            train_weights = "./save_weights/mobile-model-24.pth" # 待修改
            assert os.path.exists(train_weights), "{} file dose not exist.".format(train_weights)
            self.model.load_state_dict(torch.load(train_weights, map_location='cpu')["model"])
            self.model.to(self.device)
        elif self.model_choose == "VGG":
            self.model = vmodel(num_classes=4)  # 4
            # load train weights
            train_weights = "./save_weights/mobile-model-24.pth" # 待修改
            assert os.path.exists(train_weights), "{} file dose not exist.".format(train_weights)
            self.model.load_state_dict(torch.load(train_weights, map_location='cpu')["model"])
            self.model.to(self.device)
        elif self.model_choose == "FPN":
            self.model = mmodel(num_classes=21)  # 4
            # load train weights
            train_weights = "./save_weights/mobile-model-24.pth"
            assert os.path.exists(train_weights), "{} file dose not exist.".format(train_weights)
            self.model.load_state_dict(torch.load(train_weights, map_location='cpu')["model"])
            self.model.to(self.device)
        else:
            self.model = fmodel(num_classes=4)  # 4
            # load train weights
            train_weights = "./save_weights/mobile-model-24.pth.pth"
            assert os.path.exists(train_weights), "{} file dose not exist.".format(train_weights)
            self.model.load_state_dict(torch.load(train_weights, map_location='cpu')["model"])
            self.model.to(self.device)

    def choose_file(self):
        self.input_img.clear()
        self.output_img.clear()
        self.plainTextEdit_result_display.clear()
        self.timer_camera.stop()
        self.timer_video.stop()
        # 使用图片文件选择对话框选择图片
        self.fileName_choose, filetype = QFileDialog.getOpenFileName(
            self.centralwidget, "Select image file",
            self.path,  # 起始路径
            "image(*.jpg;*.jpeg;*.png)")  # 文件类型
        self.path = self.fileName_choose  # 保存路径
        # self.plainTextEdit_result_display.insertPlainText("打开文件%s\n" % self.fileName_choose)
        if self.fileName_choose != '':
            QtWidgets.QApplication.processEvents()
            # 读取图片
            image = self.cv_imread(self.fileName_choose)  # 读取选择的图片
            self.current_image = image.copy()
            # 在Qt界面中显示图片
            image = cv2.resize(self.current_image, (700, 700))  # 设定图像尺寸为显示界面大小
            show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], show.shape[1] * 3,
                                     QtGui.QImage.Format_RGB888)
            self.input_img.setPixmap(QtGui.QPixmap.fromImage(showImage))

            data_transform = transforms.Compose([transforms.ToTensor()])
            img = data_transform(self.current_image)
            # expand batch dimension
            img = torch.unsqueeze(img, dim=0)

            self.model.eval()  # 进入验证模式
            with torch.no_grad():
                # init
                img_height, img_width = img.shape[-2:]
                init_img = torch.zeros((1, 3, img_height, img_width), device=self.device)
                self.model(init_img)
                predictions = self.model(img.to(self.device))[0]

            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()

            for ddd in range(len(predict_boxes)):
                x1 = int(predict_boxes[ddd][0])
                y1 = int(predict_boxes[ddd][1])
                x2 = int(predict_boxes[ddd][2])
                y2 = int(predict_boxes[ddd][3])

                cv2.rectangle(self.current_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(self.current_image, self.category_index[predict_classes[ddd]] + "%.2f" % predict_scores[ddd], (x1 - 20, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

            self.detect_image = self.current_image

            # 在Qt界面中显示图片
            # self.detect_image = cv2.cvtColor(np.asarray(self.detect_image), cv2.COLOR_RGB2BGR)
            show = cv2.resize(self.detect_image, (700, 700))  # 设定图像尺寸为显示界面大小
            show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], show.shape[1] * 3,
                                     QtGui.QImage.Format_RGB888)
            self.output_img.setPixmap(QtGui.QPixmap.fromImage(showImage))
        else:
            self.input_img.clear()  # 清除画面
            self.output_img.clear()
            self.plainTextEdit_result_display.clear()

    def open_video(self):
        self.input_img.clear()
        self.output_img.clear()
        self.plainTextEdit_result_display.clear()
        self.fileName_choose, filetype = QFileDialog.getOpenFileName(self.centralwidget, "Select video file",
                                                                self.path,  # 起始路径
                                                                "Video(*.mp4;*.avi)")  # 文件类型
        self.path = self.fileName_choose  # 保存路径
        if self.timer_camera.isActive():
            self.timer_camera.stop()

        if not self.timer_video.isActive() and self.fileName_choose != '':
            self.cap = cv2.VideoCapture(self.fileName_choose)
            flag = self.cap.open(self.fileName_choose)  # 检查是否打开视频
            if not flag:  # 视频打开失败提示
                msg = QtWidgets.QMessageBox()
                msg.setWindowTitle("Warning")
                msg.setText("Video can not open!")
                msg.exec_()
            else:
                self.plainTextEdit_result_display.insertPlainText("Open successfully\n")
                # 打开定时器
                self.timer_video.start(30)


    def show_video(self):
        self.input_img.clear()
        self.output_img.clear()
        self.plainTextEdit_result_display.clear()
        self.fileName_choose = "./%s.jpg" % str(time.time())

        # 定时器槽函数，每隔一段时间执行
        flag, image = self.cap.read()  # 获取画面
        start_t = time.time()

        if flag:
            self.current_image = image.copy()
            # 在Qt界面中显示图片
            image = cv2.resize(self.current_image, (700, 700))  # 设定图像尺寸为显示界面大小
            show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], show.shape[1] * 3,
                                     QtGui.QImage.Format_RGB888)
            self.input_img.setPixmap(QtGui.QPixmap.fromImage(showImage))
            data_transform = transforms.Compose([transforms.ToTensor()])
            img = data_transform(self.current_image)
            # expand batch dimension
            img = torch.unsqueeze(img, dim=0)

            self.model.eval()  # 进入验证模式
            with torch.no_grad():
                # init
                img_height, img_width = img.shape[-2:]
                init_img = torch.zeros((1, 3, img_height, img_width), device=self.device)
                self.model(init_img)
                predictions = self.model(img.to(self.device))[0]

            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()

            for ddd in range(len(predict_boxes)):
                x1 = int(predict_boxes[ddd][0])
                y1 = int(predict_boxes[ddd][1])
                x2 = int(predict_boxes[ddd][2])
                y2 = int(predict_boxes[ddd][3])

                cv2.rectangle(self.current_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(self.current_image,
                            self.category_index[predict_classes[ddd]] + "%.2f" % predict_scores[ddd], (x1 - 20, y1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

            cv2.putText(self.current_image,
                        "FPS:%.2f" % float(1 / (time.time() - start_t)), (img_width - 150, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            self.detect_image = self.current_image

            # 在Qt界面中显示图片
            # self.detect_image = cv2.cvtColor(np.asarray(self.current_image), cv2.COLOR_RGB2BGR)
            show = cv2.resize(self.detect_image, (400, 400))  # 设定图像尺寸为显示界面大小
            show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], show.shape[1] * 3,
                                     QtGui.QImage.Format_RGB888)
            self.output_img.setPixmap(QtGui.QPixmap.fromImage(showImage))


    def cv_imread(self, filePath):
        # 读取图片
        # cv_img = cv2.imread(filePath)
        cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
        ## imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
        ## cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
        if len(cv_img.shape) == 3:
            if cv_img.shape[2] > 3:
                cv_img = cv_img[:, :, :3]
        else:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
        return cv_img