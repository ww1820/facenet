# 制作人脸特征向量的数据库 最后会保存两个文件，分别是数据库中的人脸特征向量和对应的名字。当然也可以保存在一起
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os
import h5py
import scipy.io
from pathlib import Path
from MyDataset import MyDataset
import sys

workers = 0 if os.name == 'nt' else 4
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
mtcnn = MTCNN(image_size=150,
              margin=0,
              min_face_size=20,
              thresholds=[0.6, 0.7, 0.7],
              factor=0.709,
              post_process=True,
              device=device)

resnet = InceptionResnetV1(pretrained='casia-webface').eval().to(device)


def collate_fn(x):
    return x[0]


# 将所有的单人照图片放在各自的文件夹中，文件夹名字就是人的名字,存放格式如下
'''
--orgin
    |--1.jpg
    |--2.jpg
    |--3.jpg
    |--4.jpg
    |--5.jpg
'''

# 读取info
# infoPath = '/data/wwu/DataSet/Release_v1.0_pb/pb_info/'
# imgRoot = '/data/wwu/DataSet/Release_v1.0_pb/pb_faceclips/'

infoPath = '/data/wwu/DataSet/bbt/bbt_info/'
imgRoot = '/data/wwu/DataSet/bbt/bbt_faceclips/'
# imgRoot = '/data/wwu/DataSet/Release_v1.0_pb/pb_faceclips/ImageRoot/'

files = os.listdir(infoPath)  #得到文件夹下的所有文件名称
files.sort()
imgset_list = []
class_list = []

for file in files:  #遍历文件夹
    if not os.path.isdir(file):  #判断是否是文件夹
        matPath = infoPath + file
        info_struct = scipy.io.loadmat(matPath)
        info = info_struct['info']
        episode_info = info[0, 0]['episode'].item()
        character_info = info[0, 0]['character'].item()
        id_info = info[0, 0]['id'].item()
        faceclip_info = info[0, 0]['faceclip'].item()
        imgsetPath = imgRoot + str(
            episode_info) + '/' + character_info + '/' + str(
                faceclip_info) + '/'
        imgset_list.append((imgsetPath, id_info))
        # class_list.append(id_info)
        debug = 1

fea = []
i = 1
for imgset_i in imgset_list:
    # dataset = datasets.ImageFolder(imgRoot + str(episode_i))  #加载图像集
    # dataset = datasets.ImageFolder(imgRoot)  #加载图像集
    # dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
    # loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)
    loader = MyDataset(imgset_i)
    aligned = []  # aligned就是从图像上抠出的人脸，大小是之前定义的image_size
    names = []

    for x, y in loader:
        # path = '/data/wwu/DataSet/Release_v1.0_pb/aligned/{}/'.format(dataset.idx_to_class[y])  # 这个是要保存的人脸路径
        # if not os.path.exists(path):
        #    i = 1
        #    Path(path).mkdir(parents=True, exist_ok=True)
        # 如果要保存识别到的人脸，在save_path参数指明保存路径即可,不保存可以用None
        x_aligned = mtcnn(x, return_prob=False, save_path=None)

        if x_aligned is not None:
            # print('Face detected with probability: {:8f}'.format(prob))
            aligned.append(x_aligned)
            # names.append(dataset.idx_to_class[y]
    try:
        aligned = torch.stack(aligned).to(device)
    except :
        with open(r'Dogs.txt', 'w+') as f:
            f.write(imgset_i[0])
        print(imgset_i[0])
        continue
    embeddings = resnet(aligned).detach().cpu()  # 提取所有人脸的特征向量
    fea.append(embeddings.numpy())
    class_list.append(y)
    i = i + 1
    # if i > 3:
    #     break
    debug = 1

    # db_name = 'database_' + class_list(i)
    # class_name = 'names' + str(i) + ''

    # torch.save(embeddings,'database.pt')  # 当然也可以保存在一个文件
    # torch.save(names,'names.pt')

print('done.')
print('The number of imagesets is', str(i))

fea_file_name = 'BBT.mat'
label_file_name = 'label2.mat'
scipy.io.savemat(fea_file_name, {'fea': fea})
scipy.io.savemat(label_file_name, {'label': class_list})
