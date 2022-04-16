
import xml.etree.cElementTree as ET
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import paddle
import paddle.fluid as fluid
import myyolo3net as mynet
import time
import random

# 基本参数
xmlDir = r"VOC_MASK\Annotations\s\new1"
imgDir = r"VOC_MASK\JPEGImages\new"
xmlList = [os.path.join(xmlDir, f) for f in os.listdir(xmlDir)]
labelDic = {'m': 0, 's': 1}
BATCH_SIZE = 4
imgSize = (448,448)  # w,h
iouLim = 0.1
# anchorBox = [(40, 160), (80, 210), (80, 400)]  # 前宽后高
anchorBox = [(40, 40), (120, 120), (220, 220)]  # 前宽后高
colNum = imgSize[1] // 32
rowNum = imgSize[0] // 32
anchorNum = len(anchorBox)
classNum = len(labelDic)
outDim = (5 + classNum) * anchorNum
learningRate = 1e-4
testSize=96
random.shuffle(xmlList)
xmlListTest=xmlList[-testSize:]
xmlList=xmlList[0:-testSize]

#!!! 没有轮数吗?

# 从xml读图片，标签
def loadXmlandImg(xmlDir, loadImg=False, imgDir=imgDir, resize=False):
    # 输出的size是一个元组（w，h），theObjs为列表，每一位是一个字典，键name->标签，键box->[xxyy]
    xml = ET.parse(xmlDir)
    size = (
        int(xml.find("size").find("width").text),
        int(xml.find("size").find("height").text)
    )

    objs = xml.findall("object")
    theObjs = []
    # 遍历每个物品框生成一堆字典
    for obj in objs:
        bndbox = obj.find("bndbox")
        boxXXYY = [int(bndbox.find("xmin").text), int(bndbox.find("xmax").text),
                   int(bndbox.find("ymin").text), int(bndbox.find("ymax").text)]
        if resize:
            boxXXYY[0] *= resize[0] / size[0]
            boxXXYY[1] *= resize[0] / size[0]
            boxXXYY[2] *= resize[1] / size[1]
            boxXXYY[3] *= resize[1] / size[1]
        theObjs.append({
            "label": labelDic[obj.find("name").text],
            "box": boxXXYY
        })
    # 选择是不是加载图片
    if loadImg:
        fileName = xml.find("filename").text
        fileName = os.path.join(imgDir, fileName)
        img = cv2.imread(fileName)
        if resize:
            img = cv2.resize(img, resize)
        img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32") / 255.0).transpose(2, 0, 1)
        return size, theObjs, img
    else:
        return size, theObjs




# 构造yolo3真实框，输入真实框和框数上限，输出一个（框数，4）的xywh数组
def yolo3GTBoxLabel(gtBoxes, resize,boxNum=50):
    # gtBoxes，一个字典，包含label:01234和box:[x,x,y,y]两个键值对
    # shape = d['shapes']

    thisLabel = [0 for i in range(boxNum)]
    gtBoxArr = [[0, 0, 0, 0] for i in range(boxNum)]

    # 遍历这个图的每一个真实框
    for i in range(min(len(gtBoxes),boxNum)):
        box = gtBoxes[i]
        points = box['box']
        x1 = points[0]
        x2 = points[1]
        y1 = points[2]
        y2 = points[3]
        label = box['label']
        gtx = (x1 + x2) / 2
        gty = (y1 + y2) / 2
        gtw = x2 - x1
        gth = y2 - y1
        gtBoxArr[i] = [gtx/resize[0], gty/resize[1], gtw/resize[0], gth/resize[1]]
        thisLabel[i] = label
    # print(gtBoxArr,thisLabel)
    return np.array(gtBoxArr, dtype="float32"), np.array(thisLabel, dtype="int32")


# s, o, i = loadXmlandImg(os.path.join(xmlDir, xmlList[0]), True, imgDir)
# print(i.shape)


# 定义画矩形框的程序
def draw_rectangle(bbox, currentAxis=None, edgecolor='k', facecolor='y', fill=False, linestyle='-'):
    # currentAxis，坐标轴，通过plt.gca()获取
    # bbox，边界框，包含四个数值的list， [x1, x2, y1, y2]
    # edgecolor，边框线条颜色
    # facecolor，填充颜色
    # fill, 是否填充
    # linestype，边框线型
    # patches.Rectangle需要传入左上角坐标、矩形区域的宽度、高度等参数
    if not currentAxis:
        currentAxis = plt.gca()
    rect = patches.Rectangle((bbox[0], bbox[2]), bbox[1] - bbox[0] + 1, bbox[3] - bbox[2] + 1, linewidth=1,
                             edgecolor=edgecolor, facecolor=facecolor, fill=fill, linestyle=linestyle)
    currentAxis.add_patch(rect)

# 就是从一个 文件名到各种信息的转变 -> reader
def dataReader2(xmllist, img_size=imgSize):#? 需要给imgsize吗？？？？
    labels = []
    gtboxes = []
    imgs = []
    for i in xmllist:   #             路径。 1.
        # 得到图片的size, obj(详细信息width,height) , img本身
        size, obj, img = loadXmlandImg(i, True, resize=img_size)
        imgs.append(img)
        # print(obj)
        box, label = yolo3GTBoxLabel(obj, resize=imgSize)
        labels.append(label)
        gtboxes.append(box)
    return np.array(gtboxes, dtype="float32"), np.array(labels, dtype="int32"), np.array(imgs, dtype="float32")


# 构建网络

maxGTbox = 50
paddle.enable_static()
inputImg = fluid.layers.data("image", shape=[None, 3, imgSize[1], imgSize[0]], dtype="float32")
# inputLabel = fluid.layers.data("label", shape=[None, anchorNum, 5 + classNum, rowNum, colNum], dtype="float32")
inputLabel = fluid.layers.data("label", shape=[None, maxGTbox], dtype="int32")
inputGtbox = fluid.layers.data("gtbox", shape=[None, maxGTbox, 4], dtype="float32")
(C0, C1, C2) = mynet.Darknet53(inputImg)  # 经过骨干网络提取特征
(r0, t0, P0) = mynet.yoloMain(C0, outDim)  # 得到yolo3的0级输出
r0_resize = fluid.layers.resize_nearest(input=r0, scale=2)
C1_New = fluid.layers.concat(input=[r0_resize, C1], axis=1)  # 和C1拼起来
(r1, t1, P1) = mynet.yoloMain(C1_New, outDim)  # yolo3的1级输出
r1_resize = fluid.layers.resize_nearest(input=r1, scale=2)
C1_New = fluid.layers.concat(input=[r1_resize, C2], axis=1)    # 和二级拼起来
(r2, t2, P2) = mynet.yoloMain(C1_New, outDim)
print(P0, P1, P2)

# lossAvg = mynet.yolo3LossNet(P0, inputLabel)
lossAll0 = fluid.layers.yolov3_loss( # 使用paddle框架自己的loss定义损失函数 !!!
    x=P0,
    gt_box=inputGtbox,
    gt_label=inputLabel,
    anchors=[7, 10, 12, 22, 24, 17,22, 45, 46, 33, 43, 88,85, 66, 115, 146, 275, 240],
    anchor_mask=[6,7,8],
    class_num=classNum,
    ignore_thresh=iouLim,
    downsample_ratio=32
)
lossAll1 = fluid.layers.yolov3_loss(
    x=P1,
    gt_box=inputGtbox,
    gt_label=inputLabel,
    anchors=[7, 10, 12, 22, 24, 17,22, 45, 46, 33, 43, 88,85, 66, 115, 146, 275, 240],
    anchor_mask=[3,4,5],
    class_num=classNum,
    ignore_thresh=iouLim,
    downsample_ratio=16
)
lossAll2 = fluid.layers.yolov3_loss(
    x=P2,
    gt_box=inputGtbox,
    gt_label=inputLabel,
    anchors=[7, 10, 12, 22, 24, 17,22, 45, 46, 33, 43, 88,85, 66, 115, 146, 275, 240],
    anchor_mask=[0, 1, 2],
    class_num=classNum,
    ignore_thresh=iouLim,
    downsample_ratio=8
)
lossAll = lossAll0 + lossAll1 + lossAll2
lossAvg = fluid.layers.mean(lossAll)
print(lossAvg)
testProgram = fluid.default_main_program().clone(for_test=True)
opt = fluid.optimizer.Momentum(
    learning_rate=fluid.layers.piecewise_decay(boundaries=[800,1500,4000], values=[1e-3,3e-4,1e-4,5e-5]),
    # learning_rate=1e-4,
    momentum=0.9,
    # regularization=fluid.regularizer.L2Decay(0.0005),
)
a=opt.minimize(lossAvg)
# 开始训练

print("开始编译网络")
initPro = fluid.default_startup_program()
mainProg = fluid.default_main_program()

# place = fluid.CPUPlace() #use cpu???
place = fluid.CUDAPlace(0)  # use gpu
exe = fluid.Executor(place)
exe.run(program=initPro)


def testModel():
    miniBatch = 4
    begin = 0
    end = miniBatch
    lossAll = 0.0
    while end <= testSize:
        b, l, ii = dataReader2(xmlListTest[begin:end])
        lo = exe.run(testProgram,
                     feed={
                         'image': ii,
                         "label": l,
                         'gtbox': b
                     },
                     fetch_list=[lossAvg]
                     )
        lossAll += lo[0]
        # print(lo[0])
        begin += miniBatch
        end += miniBatch
    print("损失：", lossAll / (testSize // miniBatch))


# TODO ...
model_path='./model2'
if(os.path.exists(model_path)):
    fluid.io.load_persistables(exe,model_path, mainProg) # 直接注释掉?
print("开始训练")
dataNum = len(xmlList)
print(dataNum)
# TODO 看看总长度是多少???
# 总次数 = 5 * dataNum/BATCH_SIZE * 1 = ?

acc=[]
loss=[]
epochs = range(0,1)
for i in range(1):  # 训练5次?    0226:改为2次
    print("This is {} times in big FOR".format(i))
    testModel()  # 之前定义的函数
    random.shuffle(xmlList)  # 将序列中的元素随机打乱 -》 打乱xmlList
    tic = time.time()  # 记录时间
    begin = 0
    end = BATCH_SIZE
    while end <= dataNum:  # 小于总数据num
        b, l, ii = dataReader2(xmlList[begin:end])  # 读取list内的内容
        # ['VOC_MASK/Annotations\\00535_Mask_Mouth_Chin.xml'] => np.array类型的容易得到的数据
        # print(b.shape, l.shape, i.shape)
        for j in range(1):
            la = exe.run(
                program=mainProg,
                feed={
                    'image': ii,
                    "label": l,
                    'gtbox': b
                },
                fetch_list=[lossAvg]
            )  # exe.run() 返回什么??
        # print(la)
        if begin % 128 == 0:  # change here?
            print(i, begin, la, time.time() - tic)  # 打印训练的各种信息，时间等等
            loss=loss+la
            # 写一个详细的 tips !!!
        begin += BATCH_SIZE  # 280s 一次
        end += BATCH_SIZE

fluid.io.save_persistables(exe, './model2', fluid.default_main_program())
    # 训练了2h+,还是不知道结果怎么样????
    # 显示中间过程!!
    # 暂时保存模型
    # fluid.io.save_persistables(exe, './model2', fluid.default_main_program())
    # 第二次 20-30s一个batch !!!

    # b, l, i = dataReader2(xmlList[0:1])
    # print(b.shape, l.shape, i.shape)

    # 保存预测模型
fluid.io.save_inference_model("./refMaskModel2/",
                                  feeded_var_names=['image'],
                                  target_vars=[P0, P1, P2],
                                  executor=exe)

import matplotlib.pyplot as plt

plt.plot(epochs,acc,color='r',label='acc')        # r表示红色
# plt.plot(epochs,loss,color=(0,0,0),label='loss')  #也可以用RGB值表示颜色

plt.show()
print('finishing training')