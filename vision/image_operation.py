import torch
import torch.nn as nn
import  torchvision
import cv2
import torchvision.transforms as transforms

class Operation:
    def __init__(self):
        pass





if __name__ == '__main__':
    img = cv2.imread("my_cat.jpg")
    flip_random = transforms.RandomHorizontalFlip()

    cv2.imshow("my_cat",flip_random(img))
    cv2.waitKey()