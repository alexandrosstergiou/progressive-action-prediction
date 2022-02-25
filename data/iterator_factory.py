'''
---  I M P O R T  S T A T E M E N T S  ---
'''
import os
import random
import sys
import coloredlogs, logging
coloredlogs.install()
import math
import torch
import copy
import numpy as np
import imgaug.augmenters as iaa
import torch.multiprocessing as mp
from torch.nn import functional as F
from . import video_transforms as transforms
from .video_iterator import VideoIter
from torch.utils.data.sampler import RandomSampler
from . import video_sampler as sampler


'''
---  S T A R T  O F  F U N C T I O N  G E T _ D A T A  ---

    [About]
        Function for creating iteratiors for both training and validation sets.

    [Args]
        - data_dir: String containing the complete path of the dataset video files.
        - labels_dir: String containing the complete path of the label files.
        the parent path of where the dataset is. Defaults to `/media/user/disk0`.
        - video_per_train: Float for the precentage of video frames used in training. Defaults to .6.
        - video_per_val: Float for the precentage of video frames used in inference. Defaults to .6.
        - clip_length: Integer for the number of frames to sample per video. Defaults to 8.
        - clip_size: Tuple for the width and height of the frames in the video. Defaults to (224,224).
        - val_clip_length: Integer for the number of frames in the validation clips. If None, they
        will be assigned the same as `clip_length`. Defaults to None.
        - val_clip_size: Integer for the width and height of the frames in the validation clips. If None, they
        will be assigned the same as `clip_size`. Defaults to None.
        - train_interval: Integer for the interval for sampling the training clips. Defaults to 2.
        - val_interval: Integer for the interval for sampling the validation clips. Defaults to 2.
        - mean: List or Tuple for the per-channel mean values of frames. Used to normalise the values in range.
        It uses the ImageNet mean by default. Defaults to [0.485, 0.456, 0.406].
        - std: list or Tuple of the per-channel standard deviation for frames. Used to normalise the values in range.
        It uses the ImageNet std by default. Defaults to [0.229, 0.224, 0.225].
        - seed: Integer for randomisation.
        - return_video_path: Bool for returning the filepath string alongside the video. Used for inference. Defaults to False.

    [Returns]
        - Tuple for training VideoIter object and validation VideoIter object.
'''
def get_data(data_dir=os.path.join('data','UCF-101'),
             labels_dir=os.path.join('data','UCF-101'),
             eval_only=False,
             video_per_train=.6,
             video_per_val=.6,
             num_samplers=4,
             clip_length=8,
             clip_size=(224,224),
             val_clip_length=None,
             val_clip_size=None,
             include_timeslices = True,
             train_interval=[1,2],
             val_interval=[1],
             mean=[0.485, 0.456, 0.406],
             std=[0.229, 0.224, 0.225],
             seed=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
             return_video_path=False,
             **kwargs):

    logging.debug("VideoIter:: clip_length = {}, interval = [train: {}, val: {}], seed = {}".format( \
                clip_length, train_interval, val_interval, seed))

    if not eval_only:

        # Use augmentations only for part of the data
        sometimes_aug = lambda aug: iaa.Sometimes(0.25, aug)
        sometimes_seq = lambda aug: iaa.Sometimes(0.75, aug)


        train_sampler = sampler.RandomSampling(num=clip_length,
                                               interval=train_interval,
                                               speed=[1.0, 1.0],
                                               seed=(seed+0))

        # Special case for Something-Something clip_size[0]=100
        if clip_size[0]<224 or clip_size[1]<224:
            train = VideoIter(dataset_location=data_dir,
                              csv_filepath=os.path.join(labels_dir, 'train.csv'),
                              video_per=video_per_train,
                              num_samplers=num_samplers,
                              return_video_path=return_video_path,
                              include_timeslices = include_timeslices,
                              sampler=train_sampler,
                              video_size=(clip_length,clip_size[0],clip_size[1]),
                              video_transform = transforms.Compose(
                                  transforms=iaa.Sequential([
                                      iaa.Resize({"shorter-side": clip_size[0], "longer-side":"keep-aspect-ratio"}),
                                      sometimes_seq(iaa.Sequential([
                                          sometimes_aug(iaa.GaussianBlur(sigma=[0.1,0.2])),
                                          sometimes_aug(iaa.Add((-5, 5), per_channel=True)),
                                          sometimes_aug(iaa.AverageBlur(k=(1,2))),
                                          sometimes_aug(iaa.Multiply((0.9, 1.1))),
                                          sometimes_aug(iaa.GammaContrast((0.95,1.05),per_channel=True)),
                                          sometimes_aug(iaa.AddToHueAndSaturation((-7, 7), per_channel=True)),
                                          sometimes_aug(iaa.LinearContrast((0.95, 1.05))),
                                          sometimes_aug(
                                              iaa.OneOf([
                                                  iaa.PerspectiveTransform(scale=(0.02, 0.05), keep_size=True),
                                                  iaa.Rotate(rotate=(-10,10)),
                                              ])
                                          )
                                      ])),
                                      iaa.Fliplr(0.5)
                                  ]),
                                  normalise=[mean,std]
                              ),
                              name='train',
                              shuffle_list_seed=(seed+2))

        else:
            train = VideoIter(dataset_location=data_dir,
                              csv_filepath=os.path.join(labels_dir, 'train.csv'),
                              video_per=video_per_train,
                              num_samplers=num_samplers,
                              return_video_path=return_video_path,
                              include_timeslices = include_timeslices,
                              sampler=train_sampler,
                              video_size=(clip_length,clip_size[0],clip_size[1]),
                              video_transform = transforms.Compose(
                                  transforms=iaa.Sequential([
                                      iaa.Resize({"shorter-side": 384, "longer-side":"keep-aspect-ratio"}),
                                      iaa.CropToFixedSize(width=384, height=384, position='center'),
                                      iaa.CropToFixedSize(width=clip_size[1], height=clip_size[0], position='uniform'),
                                      sometimes_seq(iaa.Sequential([
                                          sometimes_aug(iaa.GaussianBlur(sigma=[0.1,0.2])),
                                          sometimes_aug(iaa.Add((-5, 10), per_channel=True)),
                                          sometimes_aug(iaa.AverageBlur(k=(1,2))),
                                          sometimes_aug(iaa.Multiply((0.9, 1.1))),
                                          sometimes_aug(iaa.GammaContrast((0.85,1.15),per_channel=True)),
                                          sometimes_aug(iaa.AddToHueAndSaturation((-7, 7), per_channel=True)),
                                          sometimes_aug(iaa.LinearContrast((0.9, 1.1))),
                                          sometimes_aug(
                                              iaa.OneOf([
                                                  iaa.PerspectiveTransform(scale=(0.02, 0.05), keep_size=True),
                                                  iaa.Rotate(rotate=(-10,10)),
                                              ])
                                          )
                                      ])),
                                      iaa.Fliplr(0.5)
                                  ]),
                                  normalise=[mean,std]
                              ),
                              name='train',
                              shuffle_list_seed=(seed+2))

    # Only return train iterator
    if (val_clip_length is None and val_clip_size is None):
        return train

    val_sampler = sampler.SequentialSampling(num=clip_length,
                                             interval=val_interval,
                                             fix_cursor=True,
                                             shuffle=True)

    if clip_size[0]<224 or clip_size[1]<224:
        val = VideoIter(dataset_location=data_dir,
                        csv_filepath=os.path.join(labels_dir, 'val.csv'),
                        include_timeslices = include_timeslices,
                        video_per=video_per_val,
                        num_samplers=num_samplers,
                        return_video_path=return_video_path,
                        sampler=val_sampler,
                        video_size=(clip_length,clip_size[0],clip_size[1]),
                        video_transform=transforms.Compose(
                                        transforms=iaa.Sequential([
                                            iaa.Resize({"shorter-side": clip_size[0], "longer-side":"keep-aspect-ratio"})
                                        ]),
                                        normalise=[mean,std]),
                         name='val')
    else:
        val = VideoIter(dataset_location=data_dir,
                        csv_filepath=os.path.join(labels_dir, 'val.csv'),
                        include_timeslices = include_timeslices,
                        video_per=video_per_val,
                        num_samplers=num_samplers,
                        return_video_path=return_video_path,
                        sampler=val_sampler,
                        video_size=(16,256,256),
                        video_transform=transforms.Compose(
                                        transforms=iaa.Sequential([
                                                   iaa.Resize({"shorter-side": 294, "longer-side":"keep-aspect-ratio"}),
                                                   iaa.CropToFixedSize(width=294, height=294, position='center'),
                                                   iaa.CropToFixedSize(width=256, height=256, position='center')
                                                   ]),
                                        normalise=[mean,std]),
                         name='val')

    if eval_only:
        return val
    return (train, val)
'''
---  E N D  O F  F U N C T I O N  G E T _ D A T A  ---
'''



'''
---  S T A R T  O F  F U N C T I O N  C R E A T E  ---

    [About]
        Function for creating iterable datasets.

    [Args]
        - batch_size: Integer for the size of each batch.
        - return_len: Boolean for returning the length of the dataset. Defaults to False.

    [Returns]
        - Tuple for training VideoIter object and validation utils.data.DataLoader object.
'''
def create(return_len=False, return_train=True, **kwargs):

    if not return_train:
        val = get_data(eval_only=True,**kwargs)
        val_loader = torch.utils.data.DataLoader(val,
            batch_size=kwargs['batch_size'], shuffle=False,
            num_workers=kwargs['num_workers'], pin_memory=False)
        return val_loader

    dataset_iter = get_data(**kwargs)
    train,val = dataset_iter
    val_loader = torch.utils.data.DataLoader(val,
        batch_size=kwargs['batch_size'], shuffle=False,
        num_workers=kwargs['num_workers'], pin_memory=False)

    return train,val_loader,train.__len__()

    if(not isinstance(dataset_iter,tuple)):
            train_loader = torch.utils.data.DataLoader(dataset_iter,
                batch_size=batch_size, shuffle=True,
                num_workers=kwargs['num_workers'], pin_memory=False)

            return train_loader

    train,val = dataset_iter

    train_loader = torch.utils.data.DataLoader(train,
        batch_size=batch_size, shuffle=True,
        num_workers=kwargs['num_workers'], pin_memory=False)

    val_loader = torch.utils.data.DataLoader(val,
        batch_size=batch_size, shuffle=False,
        num_workers=kwargs['num_workers'], pin_memory=False)

    if return_len:
        return(train_loader,val_loader,train.__len__())
    else:
        return (train_loader, val_loader)
'''
---  E N D  O F  F U N C T I O N  C R E A T E  ---
'''
