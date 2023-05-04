import torch
import torch.distributed as dist
import torchvision.transforms as transforms
from dataset.cocodataset import CoCoDataset
from all_lib.lib_dualcoop.dataset.partial_cocodataset import PartialCOCO2014dataset
from all_lib.lib_dualcoop.dataset.partial_pascal_voc07 import PartialVoc07Dataset
from dataset.transforms import SLCutoutPIL
from dataset.transforms import MultiScaleCrop
from dataset.transforms import build_transform
from randaugment import RandAugment
import os.path as osp

def distributedsampler(cfg, train_dataset, val_dataset):
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    assert cfg.OPTIMIZER.batch_size // dist.get_world_size() == cfg.OPTIMIZER.batch_size / dist.get_world_size(), 'Batch size is not divisible by num of gpus.'
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.OPTIMIZER.batch_size // dist.get_world_size(), shuffle=(train_sampler is None),
        num_workers=cfg.DATA.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.OPTIMIZER.batch_size // dist.get_world_size(), shuffle=False,
        num_workers=cfg.DATA.num_workers, pin_memory=True, sampler=val_sampler)
    return train_loader, val_loader, train_sampler

def without_distributedsampler(cfg, train_dataset, val_dataset):

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.OPTIMIZER.batch_size, shuffle=True,
        num_workers=cfg.DATA.num_workers, pin_memory=True, drop_last=True)
    
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
    #                         num_workers=args.num_workers, pin_memory=True, 
    #                         collate_fn=collate_fn, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.OPTIMIZER.batch_size, shuffle=False,
        num_workers=cfg.DATA.num_workers, pin_memory=True)

    return train_loader, val_loader



def get_datasets(cfg, logger):
    if cfg.DATA.TRANSFORM.crop:
        train_data_transform_list = [transforms.Resize((cfg.DATA.TRANSFORM.img_size+64, cfg.DATA.TRANSFORM.img_size+64)),
                                                MultiScaleCrop(cfg.DATA.TRANSFORM.img_size, scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor()]
    else:
        train_data_transform_list = [transforms.Resize((cfg.DATA.TRANSFORM.img_size, cfg.DATA.TRANSFORM.img_size)),
                                                RandAugment(),
                                                transforms.ToTensor()]

    test_data_transform_list =  [transforms.Resize((cfg.DATA.TRANSFORM.img_size, cfg.DATA.TRANSFORM.img_size)),
                                            transforms.ToTensor()]
    if cfg.DATA.TRANSFORM.cutout and cfg.DATA.TRANSFORM.crop is not True:
        logger.info("Using Cutout!!!")
        train_data_transform_list.insert(1, SLCutoutPIL(n_holes=cfg.DATA.TRANSFORM.n_holes, length=cfg.DATA.TRANSFORM.length))
    

    if cfg.DATA.TRANSFORM.remove_norm is False:
        if cfg.DATA.TRANSFORM.orid_norm:
            normalize = transforms.Normalize(mean=[0, 0, 0],
                                            std=[1, 1, 1])
            logger.info("mean=[0, 0, 0], std=[1, 1, 1]")
        else:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            logger.info("mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]")
        train_data_transform_list.append(normalize)
        test_data_transform_list.append(normalize)
    else:
        logger.info('remove normalize')

    # train_data_transform = transforms.Compose(train_data_transform_list)
    # # train_data_transform = transforms.Compose(test_data_transform_list)
    # test_data_transform = transforms.Compose(test_data_transform_list)

    # TRANSFORMS: ["random_resized_crop", "MLC_Policy", "random_flip", "normalize"]
    train_data_transform = build_transform(cfg=cfg, is_train=True, choices=None)
    test_data_transform = build_transform(cfg=cfg, is_train=False, choices=None)


    logger.info('train_data_transform {}'.format(train_data_transform))
    logger.info('test_data_transform {}'.format(test_data_transform))


    if cfg.DATA.dataname == 'coco14' or cfg.DATA.dataname == 'COCO2014':
        # ! config your data path here.
        dataset_dir = cfg.DATA.dataset_dir
        dataset_dir = osp.join(dataset_dir, 'COCO2014')
        train_dataset = CoCoDataset(
            image_dir=osp.join(dataset_dir, 'train2014'),
            anno_path=osp.join(dataset_dir, 'annotations/instances_train2014.json'),
            input_transform=train_data_transform,
            labels_path= osp.join(dataset_dir, 'label_npy', 'train_label_vectors_coco14.npy')
        )
        val_dataset = CoCoDataset(
            image_dir=osp.join(dataset_dir, 'val2014'),
            anno_path=osp.join(dataset_dir, 'annotations/instances_val2014.json'),
            input_transform=test_data_transform,
            labels_path= osp.join(dataset_dir, 'label_npy', 'val_label_vectors_coco14.npy')
        )

    elif cfg.DATA.dataname == 'coco14partial' or cfg.DATA.dataname == 'COCO14Partial':
        # ! config your data path here.
        dataset_dir = cfg.DATA.dataset_dir
        dataset_dir = osp.join(dataset_dir, 'COCO2014')
        train_dir = osp.join(dataset_dir, 'train2014')

        train_anno_path = osp.join(dataset_dir, 'annotations/instances_train2014.json')
        train_label_path = './partdata/coco/train_label_vectors.npy'
        
        test_dir = osp.join(dataset_dir, 'val2014')
        test_anno_path = osp.join(dataset_dir, 'annotations/instances_val2014.json')
        test_label_path = './partdata/coco/val_label_vectors.npy'
        
        train_dataset = PartialCOCO2014dataset(
            dataset_dir=dataset_dir,
            mode='train',
            image_dir=train_dir,
            anno_path=train_anno_path,
            labels_path=train_label_path,
            input_transform=train_data_transform,
            label_proportion=cfg.DATA.prob
        )
        val_dataset = PartialCOCO2014dataset(
            dataset_dir=dataset_dir,
            mode='val',
            image_dir=test_dir,
            anno_path=test_anno_path,
            labels_path=test_label_path,
            input_transform=test_data_transform
        )
    elif cfg.DATA.dataname == 'voc2007partial' or cfg.DATA.dataname == 'VOC2007partial':
        dataset_dir = osp.join(cfg.DATA.dataset_dir, 'VOC2007')

        dup=None
        train_dataset = PartialVoc07Dataset(dataset_dir=dataset_dir,
                                img_dir=osp.join(dataset_dir, 'VOCdevkit/VOC2007/JPEGImages'), 
                                anno_path = osp.join(dataset_dir, 'VOCdevkit/VOC2007/ImageSets/Main/trainval.txt'), 
                                transform = train_data_transform, 
                                labels_path = osp.join(dataset_dir, 'VOCdevkit/VOC2007/Annotations'), 
                                mode='trainval',
                                label_proportion=cfg.DATA.prob,
                                dup=None)

        val_dataset = PartialVoc07Dataset(dataset_dir=dataset_dir,
                                  img_dir=osp.join(dataset_dir, 'VOCdevkit/VOC2007/JPEGImages'), 
                                    anno_path = osp.join(dataset_dir, 'VOCdevkit/VOC2007/ImageSets/Main/test.txt'), 
                                    transform = test_data_transform, 
                                    labels_path = osp.join(dataset_dir, 'VOCdevkit/VOC2007/Annotations'),
                                    mode='test',
                                    label_proportion=1.0,
                                    dup=None)
    else:
        raise NotImplementedError("Unknown dataname %s" % cfg.DATA.dataname)

    print("len(train_dataset):", len(train_dataset)) 
    print("len(val_dataset):", len(val_dataset))
    return train_dataset, val_dataset
