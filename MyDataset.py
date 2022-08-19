from PIL import Image
from torch.utils.data import Dataset
import os

#集成Dataset类
class MyDataset(Dataset):
    # def __init__(self, txt_path, transform = None, target_transform = None):
    #     """
    #     tex_path : txt文本路径，该文本包含了图像的路径信息，以及标签信息
    #     transform：数据处理，对图像进行随机剪裁，以及转换成tensor
    #     """
    #     fh = open(txt_path, 'r')  #读取文件
    #     imgs = []  #用来存储路径与标签
    #     #一行一行的读取
    #     for line in fh:
    #         line = line.rstrip()  #这一行就是图像的路径，以及标签  
            
    #         words = line.split()
    #         imgs.append((words[0], int(words[1])))  #路径和标签添加到列表中
    #         self.imgs = imgs                        
    #         self.transform = transform
    #         self.target_transform = target_transform

    def __init__(self, imgset, transform = None, target_transform = None):
        """
        imgPath_list : 图像的路径信息
        class_list ： 标签信息
        transform ：数据处理，对图像进行随机剪裁，以及转换成tensor
        """
        # fh = open(txt_path, 'r')  #读取文件
        imgs = []  #用来存储路径与标签
        #一行一行的读取
        frames = os.listdir(imgset[0]) #得到文件夹下的所有文件名称
        frames.sort()
        for fr in frames:
            imgPath = imgset[0] + '/' + fr
            imgs.append((imgPath, float(imgset[1])))  #路径和标签添加到列表中
            self.imgs = imgs                        
            self.transform = transform
            self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]   #通过index索引返回一个图像路径fn 与 标签label
        img = Image.open(fn).convert('RGB')  #把图像转成RGB
        if self.transform is not None:
            img = self.transform(img) 
        return img, label              #这就返回一个样本

    def __len__(self):
        return len(self.imgs)          #返回长度，index就会自动的指导读取多少
