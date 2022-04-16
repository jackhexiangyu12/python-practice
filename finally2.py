# 预测模型
import cv2
import numpy as np
import paddle.fluid as fluid
import time
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
import paddle

paddle.enable_static()

print("开始运行")

fluid.Scope()

## 加载所有文件的路径，每个文件在数组中有一位
modelPath = r'refMaskModel2'
dataMap = {'m': 0, 's': 1}  #这里应该有三种预测，
classNum = 2          #这也应该为3
limObj = 0.8
iouLim=0.1

## 加载模型
place = fluid.CUDAPlace(0)
mainProg = fluid.default_main_program()
exe = fluid.Executor(place)


#fluid.io.load_inference_model载入我们训练的模型

[inference_program,  # 预测用的program
 feed_target_names,  # 一个str列表，它包含需要在预测 Program 中提供数据的变量的名称。
 fetch_targets] = fluid.io.load_inference_model(
    modelPath,
    exe
)


# 测试模型

def loadImg(path):
    img = path
    # print(img)
    img = cv2.resize(img, (448, 448), interpolation=cv2.INTER_AREA)
    img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32") / 255.0).transpose(2, 0, 1)
    return np.array([img], dtype="float32")


def sigma(x):
    return 1 / (1 + math.exp(-x))


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


    #求交并比，计算两个矩形的交并比，通常在检测任务里面可以作为一个检测指标。你的预测bbox和groundtruth之间的差异，就可以通过IOU来体现。
def getIOU(xxyy1, xxyy2):
    x2 = min(xxyy1[1], xxyy2[1])
    x1 = max(xxyy1[0], xxyy2[0])
    y1 = max(xxyy1[2], xxyy2[2])
    y2 = min(xxyy1[3], xxyy2[3])

    if (x2 < x1) or (y1 > y2):
        return 0.0
    else:
        i = (x2 - x1) * (y2 - y1)
        s1 = (xxyy1[1] - xxyy1[0]) * (xxyy1[3] - xxyy1[2])
        s2 = (xxyy2[1] - xxyy2[0]) * (xxyy2[3] - xxyy2[2])
        return i / (s1 + s2 - i)


def refImg(imgPath):
    # maybeBox = [[] for i in range(classNum)]
    maybeBox = []
    img = loadImg(imgPath)

    def logBox(pred, stride, archors):
        pred0_ = pred[0]
        archorNum = len(archors) // 2
        for i in range(pred0_.shape[1]):
            for j in range(pred0_.shape[2]):
                for p in range(archorNum):
                    if sigma(pred0_[4 + p * (5 + classNum), i, j]) > limObj:
                        res = pred0_[p * (5 + classNum):(p + 1) * (5 + classNum), i, j]
                        x = stride * (j + sigma(res[0]))
                        y = stride * (i + sigma(res[1]))
                        w = archors[p * 2] * math.exp(res[2])
                        h = archors[p * 2 + 1] * math.exp(res[3])
                        max_ = (-1000)
                        maxClass = -1
                        # 比较是脸还是口罩
#                         if res[5] > res[6]:
#                             maxClass = 0
#                         else:
#                             maxClass = 1

                        for c in range(5, 5 + classNum):
                            if res[c] > max_:
                                max_ = res[c]
                                maxClass = c - 5

                        maybeBox.append({
                            "bbox": [x - w / 2, x + w / 2, y - h / 2, y + h / 2],
                            "obj": sigma(res[5 + maxClass]),
                            "state": 0,
                            "class": maxClass,
                            "stride": stride
                        })

    tic = time.time()
    pred0, pred1, pred2 = exe.run(
        program=inference_program,
        feed={feed_target_names[0]: img},
        fetch_list=fetch_targets
    )
    print(time.time() - tic)
    logBox(pred0, stride=32, archors=[85, 66, 115, 146, 275, 240])
    logBox(pred1, stride=16, archors=[22, 45, 46, 33, 43, 88])
    logBox(pred2, stride=8, archors=[7, 10, 12, 22, 24, 17])
    # print(maybeBox)
    plt.imshow(img[0].transpose(1, 2, 0))
    # goodBox0 = [[] for i in range(classNum)]
    # 去除多余的框
    # for i in range(classNum):
    while True:
        finish = True
        max_ = -1
        maxBox = 0
        for box in maybeBox:
            if box["state"] == 0 and box["obj"] > max_:
                finish = False
                max_ = box["obj"]
                maxBox = box
        if not finish:
            maxBox["state"] = 2
        for box in maybeBox:
            if box["state"] == 0 and getIOU(box["bbox"], maxBox["bbox"]) > 0.1:
                box["state"] = 1
        if finish:
            break

#     print("face")
    img = img[0].transpose(1, 2, 0).copy()
    for box in maybeBox:
        if box["state"] == 2:
            bbox = box["bbox"]
            print(box)
            if box["class"] == 0:
                # h=int(bbox[3]-bbox[2])
                # bias=int(h*0.2)
                bias=0
                cv2.rectangle(img, (int(bbox[0]), int(bbox[2]+bias)), (int(bbox[1]), int(bbox[3]+bias)),(255,0,0), 5)
                # draw_rectangle(box["bbox"], edgecolor='r')
            else:
                # h=int(bbox[3]-bbox[2])
                # bias=int(h*0.2)
                bias=0
                cv2.rectangle(img, (int(bbox[0]), int(bbox[2]+bias)), (int(bbox[1]), int(bbox[3]+bias)),(0,0,255), 5)
                # draw_rectangle(box["bbox"])

    # print("facemask")
    # for box in maybeBox[1]:
    #     if box["state"] == 2:
    #         print(box)
    #         draw_rectangle(box["bbox"])
    # plt.show()
    return img

import cv2
# import time

cap=cv2.VideoCapture(1)

cap.open(0)

while cap.isOpened():
    success,frame=cap.read()
    if not success:
        print("read error")
        break
    # start_time=time.time()
    # print(frame)

    frame=refImg(frame)
    b, g, r = cv2.split(frame)
    cv2.imshow("my_window",cv2.merge((r, g, b)))

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()

cv2.destroyAllWindows()