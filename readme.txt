1、数据处理
你的数据放在VOCdevkit/VOC2007路径下；
运行voc2frcnn.py生成VOCdevkit/VOC2007/ImageSets下边的train.txt,val.txt,trainval.txt,test.txt，进行训练集测试集验证集划分，划分方式为随机，比例为8：1：1
运行voc_annotation.py生成主路径下的2007_test.txt，2007_train.txt和2007_val.txt
2、模型训练
运行train.py
训练的模型、日志和损失曲线保存在save_weights文件夹
3、模型指标测试
修改get_dr_txt.py第15行左右import create_model，测试哪个就导入哪个
运行get_dr_txt.py获取测试集的检测结果和可视化结果，
检测结果在input/detection-results下边
可视化结果在input/show下边
运行get_gt_txt.py获取测试集的标签文件，保存在input/ground-truth下边
运行get_map.py获取指标测试结果，保存在results，有mAP和F1的测试指标



