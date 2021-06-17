import cv2
import os
import numpy
import torch
from torch.utils.data import Dataset

class picture_read(Dataset):
    def __init__(self, imageFolder):
        self.imageFolder = imageFolder
        self.images = os.listdir(self.imageFolder)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        name = self.images[index]
        name = os.path.join(self.imageFolder, name)
        image = cv2.imdecode(numpy.fromfile(name,dtype=numpy.uint8),-1) #防止图片中文名称乱码
        image = numpy.array(image,dtype='float32')
        image = image / 255.
        image = torch.from_numpy(image)
        image = image.permute(2,0,1).unsqueeze(0)
        return image, name