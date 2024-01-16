'''
---  I M P O R T  S T A T E M E N T S  ---
'''
import os
import sys
import coloredlogs, logging
coloredlogs.install()
import torch
from .video_iterator import VideoIter
from . import video_sampler as sampler


import pytorchvideo
import torchvision

'''
---  S T A R T  O F  F U N C T I O N  F L E X I B L E _ P A D  ---

    [About]
        Function for data padding based on the largest dim. It will always ensure that the size of dim -2 and -1 match by padding the shortest of the two.

    [Args]
        - data: Tensor of size [..., D2, D1]. 

    [Returns]
        - Tensor of size [..., max(D1,D2), max(D1,D2)].
'''
def flexible_pad(data):
    if data.size()[-2] < data.size()[-1]:
        data = torch.nn.functional.pad(data,(0,0,data.size()[-1]-data.size()[-2],0,0,0))
    if data.size()[-1] < data.size()[-2]:
        data = torch.nn.functional.pad(data,(data.size()[-2]-data.size()[-1],0,0,0,0,0))
    return data
'''
---  E N D  O F  F U N C T I O N  F L E X I B L E _ P A D  ---
'''


'''
---  S T A R T  O F  F U N C T I O N  G E T _ D A T A  ---

    [About]
        Function for creating iteratiors for both training and validation sets.

    [Args]
        - data_dir: String containing the complete path of the dataset video files.
        - labels_dir: String containing the complete path of the label files.
        the parent path of where the dataset is. Defaults to `/media/user/disk0`.
        - eval_only: Bool for returning only the validation set iterator.
        - video_per_train: Float for the precentage of video frames used in training. Defaults to .6.
        - video_per_val: Float for the precentage of video frames used in inference. Defaults to .6.
        - num_samplers: Integer for the number of samplers. Deafaults to 4.
        - clip_length: Integer for the number of frames to sample per video. Defaults to 8.
        - clip_size: Tuple for the width and height of the frames in the video. Defaults to (224,224).
        - val_clip_length: Integer for the number of frames in the validation clips. If None, they
        will be assigned the same as `clip_length`. Defaults to None.
        - val_clip_size: Integer for the width and height of the frames in the validation clips. If None, they
        will be assigned the same as `clip_size`. Defaults to None.
        - include_timeslices: Bool for the filenames containing time slices from the orginal video. Defaults to True.
        - train_interval: Integer for the interval for sampling the training clips. Defaults to 2.
        - val_interval: Integer for the interval for sampling the validation clips. Defaults to 2.
        - mean: List or Tuple for the per-channel mean values of frames. Used to normalise the values in range.
        It uses the ImageNet mean by default. Defaults to [0.485, 0.456, 0.406].
        - std: list or Tuple of the per-channel standard deviation for frames. Used to normalise the values in range.
        It uses the ImageNet std by default. Defaults to [0.229, 0.224, 0.225].
        - seed: Integer for randomisation.
        - return_video_path: Bool for returning the filepath string alongside the video. Used for inference. Defaults to False.
        - use_frames: Bool for videos stored in folders of images. Deafults to False.

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
             use_frames=False,
             **kwargs):

    logging.debug("VideoIter:: clip_length = {}, interval = [train: {}, val: {}], seed = {}".format( \
                clip_length, train_interval, val_interval, seed))

    if not eval_only:


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
                              video_transform = torchvision.transforms.Compose([
                                  pytorchvideo.transforms.Div255(),
                                  pytorchvideo.transforms.Normalize(mean, std),
                                  pytorchvideo.transforms.ShortSideScale(clip_size[0]),
                                  torchvision.transforms.Resize((clip_size[0],clip_size[1])),
                                  pytorchvideo.transforms.Permute([1,0,2,3]),
                                  torchvision.transforms.GaussianBlur(sigma=[0.1, 2.0], kernel_size=5),
                                  pytorchvideo.transforms.Permute([1,0,2,3]),
                                  torchvision.transforms.RandomHorizontalFlip(p=0.5),
                                  torchvision.transforms.Lambda(flexible_pad)
                                  ]),
                              name='train',
                              shuffle_list_seed=(seed+2),
                              is_jpg=use_frames)

        else:
            train = VideoIter(dataset_location=data_dir,
                              csv_filepath=os.path.join(labels_dir, 'train.csv'),
                              video_per=video_per_train,
                              num_samplers=num_samplers,
                              return_video_path=return_video_path,
                              include_timeslices = include_timeslices,
                              sampler=train_sampler,
                              video_size=(clip_length,clip_size[0],clip_size[1]),
                              video_transform = torchvision.transforms.Compose([
                                  pytorchvideo.transforms.Div255(),
                                  pytorchvideo.transforms.Normalize(mean, std),
                                  pytorchvideo.transforms.RandomShortSideScale(min_size=256, max_size=320),
                                  torchvision.transforms.RandomCrop(244),
                                  pytorchvideo.transforms.Permute([1,0,2,3]),
                                  torchvision.transforms.GaussianBlur(sigma=[0.1, 2.0], kernel_size=5),
                                  pytorchvideo.transforms.Permute([1,0,2,3]),
                                  torchvision.transforms.RandomHorizontalFlip(p=0.5)
                                      ]),
                              name='train',
                              shuffle_list_seed=(seed+2),
                              is_jpg=use_frames)

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
                        video_transform=torchvision.transforms.Compose([
                                  pytorchvideo.transforms.Div255(),
                                  pytorchvideo.transforms.Normalize(mean, std),
                                  pytorchvideo.transforms.ShortSideScale(clip_size[0]),
                                  torchvision.transforms.Resize((clip_size[0],clip_size[1])),
                                  torchvision.transforms.Lambda(flexible_pad)
                                      ]),
                         name='val',
                         is_jpg=use_frames)
    else:
        val = VideoIter(dataset_location=data_dir,
                        csv_filepath=os.path.join(labels_dir, 'val.csv'),
                        include_timeslices = include_timeslices,
                        video_per=video_per_val,
                        num_samplers=num_samplers,
                        return_video_path=return_video_path,
                        sampler=val_sampler,
                        video_size=(16,256,256),
                        video_transform=torchvision.transforms.Compose([
                                  pytorchvideo.transforms.Div255(),
                                  pytorchvideo.transforms.Normalize(mean, std),
                                  pytorchvideo.transforms.ShortSideScale(256),
                                  torchvision.transforms.CenterCrop(256),
                                      ]),
                         name='val',
                         is_jpg=use_frames)

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
        - return_train: Bolean for returning the training dataloader. If set to `False` only the val/val_loader is returned. Defaults to `True`.

    [Returns]
        - Tuple containing training and validation VideoIter objects.
'''
def create(return_train=True, **kwargs):

    if not return_train:
        val = get_data(eval_only=True,**kwargs)
        val_loader = torch.utils.data.DataLoader(val,
            batch_size=kwargs['batch_size'], shuffle=False,
            num_workers=kwargs['num_workers'], pin_memory=False)
        return val,val_loader,val.__len__()

    dataset_iter = get_data(**kwargs)
    train,val = dataset_iter

    return train,val,train.__len__()
'''
---  E N D  O F  F U N C T I O N  C R E A T E  ---
'''
