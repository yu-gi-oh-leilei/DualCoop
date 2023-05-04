import torch
import sys
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
import numpy as np
import json
import random
from xml.dom.minidom import parse 
import xml.dom.minidom
import os
import os.path as osp

category_info = {'aeroplane':0, 'bicycle':1, 'bird':2, 'boat':3, 'bottle':4,
                 'bus':5, 'car':6, 'cat':7, 'chair':8, 'cow':9,
                 'diningtable':10, 'dog':11, 'horse':12, 'motorbike':13, 'person':14,
                 'pottedplant':15, 'sheep':16, 'sofa':17, 'train':18, 'tvmonitor':19}

classnames = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                 'bus', 'car', 'cat', 'chair', 'cow',
                 'diningtable', 'dog', 'horse', 'motorbike', 'person',
                 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

def read_labels(path_labels):
    file = path_labels
    labels = []
    with open(file, 'r') as f:
        for line in f:
            tmp = list(map(int, line.strip().split(',')))
            labels.append(torch.tensor(tmp, dtype=torch.float32))
    return labels

class PartialVoc07Dataset(data.Dataset):
    def __init__(self, 
        dataset_dir='/media/data2/MLICdataset/VOC2007/',
        img_dir='/media/data2/MLICdataset/VOC2007/VOCdevkit/VOC2007/JPEGImages', 
        anno_path='/media/data2/MLICdataset/VOC2007/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt', 
        transform=None, 
        labels_path='/media/data2/MLICdataset/VOC2007/VOCdevkit/VOC2007/Annotations',
        mode='trainval',
        label_proportion=1.0,
        dup=None,
    ):
        assert mode in ('trainval', 'test')
        
        self.dup = dup
        self.img_names  = []
        with open(anno_path, 'r') as f:
             self.img_names = f.readlines()
        self.img_dir = img_dir
        self.labels = []
        self.num_labels = len(category_info)
        self.classnames = classnames

        # if 'test' in os.path.split(anno_path)[-1]:
        if 0:
            # no ground truth of test data of voc12, just a placeholder
            self.labels = np.ones((len(self.img_names),20))
            print('no ground truth of test data of voc12, just a placeholder')
        else:
            no_anno_cnt = 0
            res_name_list = []
            for name in self.img_names:
                label_file = os.path.join(labels_path,name[:-1]+'.xml')
                if not os.path.exists(label_file):
                    no_anno_cnt += 1
                    if no_anno_cnt < 10:
                        print("cannot find: %s" % label_file)
                    continue
                res_name_list.append(name)
                label_vector = np.zeros(20)
                DOMTree = xml.dom.minidom.parse(label_file)
                root = DOMTree.documentElement
                objects = root.getElementsByTagName('object')
                for obj in objects:
                    if (obj.getElementsByTagName('difficult')[0].firstChild.data) == '1':
                        continue
                    tag = obj.getElementsByTagName('name')[0].firstChild.data.lower()
                    label_vector[int(category_info[tag])] = 1.0
                self.labels.append(label_vector)

            self.labels = np.array(self.labels).astype(np.float32)
            self.img_names = res_name_list
            if no_anno_cnt > 0:
                print("total no anno file count:", no_anno_cnt)
            # self.temp = torch.from_numpy(self.labels)
        
        os.makedirs(osp.join(dataset_dir, 'partial-labels'), exist_ok=True)
        label_path = osp.join(dataset_dir, 'partial-labels', 'train_proportion_{}.txt'.format(label_proportion))
        # print(label_path)
        if os.path.exists(label_path):
            print('Changing label proportion...')
            self.labels = read_labels(label_path)
        
        if label_proportion == 1.0:
            self.labels[self.labels == 0] = -1


        self.transform = transform
        self.return_name = False

    def __getitem__(self, index):
        if self.dup:
            index = index % self.dup
        name = self.img_names[index][:-1]+'.jpg'
        input = Image.open(os.path.join(self.img_dir, name)).convert('RGB')
          
        if self.transform:
            input = self.transform(input)
        if self.return_name:
            return input, self.labels[index], name
        return input, self.labels[index]

    def __len__(self):
        if not self.dup:
            return len(self.img_names)
        else:
            return len(self.img_names) * self.dup


if __name__ == "__main__":
    import os.path as osp   
    dataset_dir = '/media/data2/MLICdataset/'
    dataset_dir = osp.join(dataset_dir, 'VOC2007')
    train_data_transform = None
    test_data_transform = None

    # 5011  20
    # label = np.load('/media/data2/MLICdataset/VOC2007/partial-labels/train_proportion_1.0.npy')
    # # print(label[0])
    # file = '/media/data2/MLICdataset/VOC2007/partial-labels/train_proportion_1.0.txt'
    # with open(file, 'r') as f:
    #     for
        


    # print(np.random.random(10))
    # print('='*30)
    # train_dataset = PartVoc07Dataset(dataset_dir=dataset_dir,
    #                             img_dir=osp.join(dataset_dir, 'VOCdevkit/VOC2007/JPEGImages'), 
    #                             anno_path = osp.join(dataset_dir, 'VOCdevkit/VOC2007/ImageSets/Main/trainval.txt'), 
    #                             transform = train_data_transform, 
    #                             labels_path = osp.join(dataset_dir, 'VOCdevkit/VOC2007/Annotations'), 
    #                             mode='trainval',
    #                             label_proportion=1.0,
    #                             dup=None)

    # val_dataset = PartVoc07Dataset(dataset_dir=dataset_dir,
    #                               img_dir=osp.join(dataset_dir, 'VOCdevkit/VOC2007/JPEGImages'), 
    #                                 anno_path = osp.join(dataset_dir, 'VOCdevkit/VOC2007/ImageSets/Main/test.txt'), 
    #                                 transform = test_data_transform, 
    #                                 labels_path = osp.join(dataset_dir, 'VOCdevkit/VOC2007/Annotations'),
    #                                 mode='test',
    #                                 label_proportion=1.0,
    #                                 dup=None)
    # print(train_dataset.temp.sum() / len(train_dataset))
    # print( (train_dataset.temp.sum() + val_dataset.temp.sum()) / (len(train_dataset)+ len(val_dataset))  )

    # print(train_dataset[110][1])
    # print(train_dataset[111][1])
    # print(train_dataset[112][1])
    # print(train_dataset[113][1])
    # print(train_dataset[114][1])


    # print('[dataset] PASCAL VOC2007 classification set=%s number of classes=%d  number of images=%d' % ('Train', train_dataset.num_labels, len(train_dataset)))
    # print('[dataset] PASCAL VOC2007 classification set=%s number of classes=%d  number of images=%d' % ('Test', val_dataset.num_labels, len(val_dataset)))