import os
import random
import shutil

xmlDir = r"VOC_MASK\Annotations\new"
imgDir = r"VOC_MASK\JPEGImages\new"

xmlList = [os.path.join(xmlDir, f) for f in os.listdir(xmlDir)]

testSize=0.2*len(xmlList)
valSize=0.1*len(xmlList)
random.shuffle(xmlList)
xmlListTest=xmlList[int(testSize):]
xmlListVal=xmlList[:int(valSize)]
xmlListTrain=xmlList[int(valSize):int(testSize)]

for file in xmlListTest:
    shutil.copy(file, r"VOC_MASK\test\Annotations")
    shutil.copy(file.replace("Annotations", "JPEGImages").replace('.xml','.jpg'), r"VOC_MASK\test\JPEGImages")

for file in xmlListVal:
    shutil.copy(file,r"VOC_MASK\val\Annotations")
    shutil.copy(file.replace("Annotations", "JPEGImages").replace('.xml','.jpg'), r"VOC_MASK\val\JPEGImages")

for file in xmlListTrain:
    shutil.copy(file,r"VOC_MASK\train\Annotations")
    shutil.copy(file.replace("Annotations", "JPEGImages").replace('.xml','.jpg'), r"VOC_MASK\train\JPEGImages")



