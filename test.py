import cv2
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
import os 
sometimes = lambda aug: iaa.Sometimes(0.5,aug)

seq = iaa.Sequential(
    [
        iaa.Fliplr(0.7),  # 对20%的图像做左右翻转


        # 使用下面的0个到5个之间的方法去增强图像。注意SomeOf的用法
        iaa.SomeOf((0, 2),
                   [
                        
                       iaa.OneOf([
                           iaa.GaussianBlur((0, 3.0)),
                           iaa.MedianBlur(k=(3, 11)),
                       ]),
                       # 加入高斯噪声
                       iaa.AdditiveGaussianNoise(
                           loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                       ),
                                                 
                     iaa.CoarseDropout(
                               (0.03, 0.15), size_percent=(0.02, 0.05),
                               per_channel=0.2
                           ),                  
                   ],

                   random_order=True  # 随机的顺序把这些操作用在图像上
                   )
    ],
    random_order=True  # 随机的顺序把这些操作用在图像上
)
path = '/home/wang/Desktop/voids0/'
augpath ='/home/wang/Desktop/voids0aug/'

imglist =[]
filelist = os.listdir(path)
for item in filelist:
    img = cv2.imread(path+item)
    imglist.append(img)
    
for count in range(5):
    images_aug = seq.augment_images(imglist)
    for index in range(len(images_aug)):
        filename ='voids'+ str(count)+str(index)+'.jpg'
        cv2.imwrite(augpath+filename,images_aug[index])


