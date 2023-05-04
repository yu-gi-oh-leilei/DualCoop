import sys
sys.path.insert(0, './')
sys.path.insert(0, '../')
import numpy as np
import torch.utils.data as data
from PIL import Image
import torch
import os
import torchvision.transforms as transforms
from randaugment import RandAugment
import xml.dom.minidom


class voc2007(data.Dataset):
    def __init__(self, root, data_split, img_size=224, p=1, annFile="", transform=None, label_mask=None, partial=1+1e-6):
        # data_split = train / val
        self.root = root
        self.classnames = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                           'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                           'train', 'tvmonitor']
        print(root)
        if annFile == "":
            self.annFile = os.path.join(self.root, 'Annotations')
        else:
            raise NotImplementedError

        image_list_file = os.path.join(self.root, 'ImageSets', 'Main', '%s.txt' % data_split)

        with open(image_list_file) as f:
            image_list = f.readlines()
        self.image_list = [a.strip() for a in image_list]

        self.data_split = data_split
        if data_split == 'Train':
            num_examples = len(self.image_list)
            pick_example = int(num_examples * p)
            self.image_list = self.image_list[:pick_example]
        else:
            self.image_list = self.image_list
        
        self.transform = transform

        # create the label mask
        self.mask = None
        self.partial = partial
        if data_split == 'trainval' and partial < 1.:
            if label_mask is None:
                rand_tensor = torch.rand(len(self.image_list), len(self.classnames))
                mask = (rand_tensor < partial).long()
                mask = torch.stack([mask], dim=1)
                # print(mask[110])
                torch.save(mask, os.path.join(self.root, 'Partial_Annotations', 'partial_label_%.2f.pt' % partial))
            else:
                mask = torch.load(os.path.join(self.root, 'Partial_Annotations', label_mask))
            self.mask = mask.long()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, 'JPEGImages', self.image_list[index] + '.jpg')
        img = Image.open(img_path).convert('RGB')
        ann_path = os.path.join(self.annFile, self.image_list[index] + '.xml')
        label_vector = torch.zeros(20)
        DOMTree = xml.dom.minidom.parse(ann_path)
        root = DOMTree.documentElement
        objects = root.getElementsByTagName('object')
        for obj in objects:
            if (obj.getElementsByTagName('difficult')[0].firstChild.data) == '1':
                continue
            tag = obj.getElementsByTagName('name')[0].firstChild.data.lower()
            label_vector[self.classnames.index(tag)] = 1.0
        targets = label_vector.long()
        target = targets[None, ]
        # print(target)
        if self.mask is not None:
            masked = - torch.ones((1, len(self.classnames)), dtype=torch.long)
            print(masked)
            print(self.mask[index])
            target = self.mask[index] * target + (1 - self.mask[index]) * masked
            print(target)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def name(self):
        return 'voc2007'


if __name__ == "__main__":
    import os.path as osp   
    dataset_dir = '/media/data2/MLICdataset/'
    dataset_dir = osp.join(dataset_dir, 'VOC2007')
    train_data_transform = None
    test_data_transform = None

    # self, root, data_split, img_size=224, p=1, annFile="", transform=None, label_mask=None, partial=1+1e-6
    train_dataset = voc2007(root=osp.join(dataset_dir, 'VOCdevkit/VOC2007'),
                            data_split='trainval',
                            img_size=448,
                            p=1.0,
                            annFile='', 
                            # annFile=osp.join(dataset_dir, 'VOCdevkit/VOC2007/ImageSets/Main/trainval.txt'), 
                            transform=None, 
                            label_mask=None,
                            partial=0.5
                            )
    
    print(train_dataset[110])

    # /media/data2/MLICdataset/VOC2007/VOCdevkit/VOC2007
    # /media/data2/MLICdataset/VOC2007/VOCdevkit/VOC2007



    # train_dataset = voc2007(img_dir=osp.join(dataset_dir, 'VOCdevkit/VOC2007/JPEGImages'), 
    #                                 anno_path = osp.join(dataset_dir, 'VOCdevkit/VOC2007/ImageSets/Main/trainval.txt'), 
    #                                 transform = train_data_transform, 
    #                                 labels_path = osp.join(dataset_dir, 'VOCdevkit/VOC2007/Annotations'), 
    #                                 dup=None)

    # val_dataset = voc2007(img_dir=osp.join(dataset_dir, 'VOCdevkit/VOC2007/JPEGImages'), 
    #                                 anno_path = osp.join(dataset_dir, 'VOCdevkit/VOC2007/ImageSets/Main/test.txt'), 
    #                                 transform = test_data_transform, 
    #                                 labels_path = osp.join(dataset_dir, 'VOCdevkit/VOC2007/Annotations'), 
    #                                 dup=None)

