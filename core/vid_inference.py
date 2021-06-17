import os
import torch
import numpy
import cv2
from moviepy.editor import VideoFileClip
from .vid_dataset import video_read
from .model import MSRN

def video_SR(videoFolder, scale, device, fourcc, video_format):
    del_temp()  #先清理一次临时文件夹
    net = MSRN(scale)
    para = torch.load('./core/parafile/x' + str(scale) + '.pth', map_location=torch.device(device))
    net.load_state_dict(para)
    device = torch.device(device)
    net = net.to(device)
    torch.set_grad_enabled(False)
    torch.no_grad()
    net.eval()
    data_num = video_read(videoFolder).__len__()
    print("一共有"+ str(data_num) + "个视频")
    for n in range(data_num):
        print("正在处理第" + str(n+1) +"个视频")
        name = video_read(videoFolder).__getitem__(n)
        cap = cv2.VideoCapture(name)
        frame_num = int(cap.get(7))
        weight = int(cap.get(3))
        height = int(cap.get(4))
        fps = cap.get(5)
        audio = VideoFileClip(name)
        output_filepath = name.replace('input','temp')
        output_filepath = output_filepath[:-3]
        output_filepath = output_filepath + video_format
        out = cv2.VideoWriter(output_filepath, fourcc, fps, (weight*scale,height*scale), True)
        i = 0
        while cap.isOpened():
            ret, frame = cap.read()
            print("该视频一共" + str(frame_num) + "帧，正在处理第" + str(i) + "帧", end="\r")
            frame = numpy.array(frame,dtype='float32')
            if frame.size == 1:
                print("\n此帧为空，即将退出")
                break
            frame = frame / 255.
            frame = torch.from_numpy(frame)
            frame = frame.permute(2,0,1)
            frame = frame.unsqueeze(0).to(device)
            frame_SR = net(frame)
            frame_SR = frame_SR.permute(0,2,3,1)        #转换维度，把颜色维度放在最后
            frame_SR = numpy.squeeze(frame_SR,0).cpu()
            frame_SR = numpy.array(frame_SR)
            frame_SR = frame_SR * 255.
            cv2.imwrite('./temp/tmp.bmp',frame_SR)
            frame_SR = cv2.imread('./temp/tmp.bmp')
            out.write(frame_SR)
            i += 1
            if i == frame_num:
                break
        cap.release()
        out.release()
        print("\n")
        video = VideoFileClip(output_filepath)  # 设置视频的音频
        video = video.set_audio(audio.audio)  # 保存新的视频文件
        filepath = output_filepath.replace('temp','output')
        video.write_videofile(filepath)
        del_temp()

def del_temp():
    del_list = os.listdir('./temp')    #清理临时文件
    for file in del_list:
        file_path = os.path.join('./temp/', file)
        if os.path.isfile(file_path):
            os.remove(file_path)