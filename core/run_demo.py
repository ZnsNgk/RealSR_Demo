import os
import cv2
import torch
from .vid_inference import video_SR
from .pic_inference import pic_SR

class run_demo():
    def __init__(self, args):
        self.mode = args.mode
        self.scale = args.scale
        self.device = args.device
        self.format = args.format
        self.__check_folder()
        self.__check_mode()
        self.__check_device()
        self.__check_format()
        self.show()
    def __check_mode(self):
        if self.mode == "pic" or self.mode == "vid":
            pass
        else:
            raise NameError("mode can only choose 'vid' or 'pic'!")
    def __check_device(self):
        if "cuda" in self.device:
            if not torch.cuda.is_available():
                print("CUDA is not available, now try to use cpu!")
                self.device = "cpu"
    def __check_format(self):
        if self.mode == "pic":
            if self.format == None:
                self.format = "png"
            elif self.format == "jpg" or self.format == "png" or self.format == "bmp":
                pass
            else:
                raise NameError("You can only choose 'jpg', 'png' or 'bmp' in picture mode!")
        elif self.mode == "vid":
            if self.format == None:
                self.format = "mp4"
            elif self.format == "mp4" or self.format == "avi":
                pass
            else:
                raise NameError("You can only choose 'mp4' or 'avi' in video mode!")
    def __check_folder(self):
        if not os.path.exists("./output/"):
            os.mkdir("./output/")
        if not os.path.exists("./temp/"):
            os.mkdir("./temp/")
    def show(self):
        print("----------CONFIG----------")
        print("Mode: " + self.mode)
        print("Scale Factor: " + str(self.scale))
        print("Device: " + self.device)
        print("Format: " + str(self.format))
        print("--------------------------")
    def run(self):
        if self.mode == "pic":
            pic_SR('./input/', self.scale, self.device, self.format)
        elif self.mode == "vid":
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            if self.format == "avi":
                fourcc = cv2.VideoWriter_fourcc(*'xvid')
            video_SR('./input/', self.scale, self.device, fourcc, self.format)