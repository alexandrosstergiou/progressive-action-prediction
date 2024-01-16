'''
---  I M P O R T  S T A T E M E N T S  ---
'''
import json
import time
from datetime import datetime
import os
import csv
import cv2
import random
import numpy as np
import sqlite3
import torch
import torch.utils.data as data
import torch.nn.functional as F
import coloredlogs, logging
coloredlogs.install()
import linecache
import sys
from einops import rearrange

import glob
import torchvision
import pytorchvideo
from pytorchvideo.data.utils import thwc_to_cthw


'''
===  S T A R T  O F  C L A S S  V I D E O  ===

    [About]
        Wrapper class for reading videos from video files or frames folders and return them as numpy arrays.

    [Init Args]
        - vid_path: String that points to the video filepath
        - video_transform: Any object of the video_transforms file, used for applying video transformations (see video_transforms.py file). Defaults to None.
        - end_size: Tuple for the target video size in TxHxW format. Will interpolate the array if the output size from `__extract_frames_fast` is different than that of specified. Defaults to (16, 224, 224).
        - images: Bool for video being stored as folder of images or not. Defaults to False.

    [Methods]
        - __init__ : Class initialiser, takes as argument the video path string.
        - __enter__ : Function for returning the object.
        - reset : Function for resetting the class variables vid_path(string), frame_count(int) and faulty_frame(array)
        - count_frames: Function for returning the number of elements/rows in the database (i.e. the number of saved frames).
        - extract_frames: High level function for reading frames. The indices of the frames are given as an argument of type `list` which should hold EVERY index of the frames to be extracted.
'''
class Video(object):
    """basic Video class"""

    def __init__(self, vid_path, video_transform=None, end_size=(16,224,224), precentage=1., images=False):
        #self.path = vid_path
        self.video_per = precentage
        self.video_path = vid_path
        #self.frame_path = os.path.join(vid_path, 'n_frames')
        self.video_transform = video_transform
        self.end_size = end_size
        self.images = images

    def __enter__(self):
        return self

    def reset(self):
        self.video_path = None
        #self.frame_path = None
        self.frame_count = -1
        return self


    def count_frames(self):
        if self.images:
            return len(glob.glob(self.video_path,"*.jpg"))
        
        cap = cv2.VideoCapture(self.video_path)
        self.frame_count = (float(cap.get(cv2.CAP_PROP_FRAME_COUNT))* self.video_per)
        
        return self.frame_count

    def extract_frames(self, indices):
        
        frames = {}
        if self.images:
            for idx in sorted(indices):
                path = os.path.join(self.video_path,f"{idx+1:05d}.jpg")
                video_frames = [torchvision.io.read_image(path).permute(1,2,0)]
                 
        else:
            video = pytorchvideo.data.encoded_video.EncodedVideo.from_path(self.video_path)
            fs = video._container.decode(**{"video":0}) # get a stream of frames
            
            for i,f in enumerate(fs):
                if i in indices:
                    frames[i] = f
                elif i > indices[-1]:
                    break
            
            result = [frames[pts] for pts in sorted(frames)]
            video_frames = [torch.from_numpy(f.to_ndarray(format='rgb24')) for f in result] # list of length T with items size [H x W x 3]
             
        video_frames = thwc_to_cthw(torch.stack(video_frames).to(torch.float32))
        
        # Final resizing - w/ check
        _,t,_,_ = list(video_frames.size())
        

        # Interpolate temporal dimension if frames are less than the ones required
        if (t!=self.end_size[0]):
            video_frames = F.interpolate(video_frames.unsqueeze(0), size=self.end_size, mode='trilinear',align_corners=False).squeeze(0)
        
        return self.video_transform(video_frames)



'''
===  E N D  O F  C L A S S  V I D E O  ===
'''







'''
===  S T A R T  O F  C L A S S  V I D E O I T E R A T O R ===

    [About]

        Iterator class for loading the dataset filelist for a .CSV file to a dictionary and iteratively create Video class objects for frame loading.

    [Init Args]

        - dataset_location: String for the (full) directory path of the dataset.
        - csv_filepath: String for the (full) filepath of the csv file containing datset information.
        - video_per: Float for the precentage of the full video that is used for predictions.
        - include_timeslices: Boolean, of cases that datasets video directories also include the time-segments in the name (should be mstly used by either Kinetics or HACS).
        - sampler: Any object of the video_sampler file, used for sampling the frames.
        - video_transform: Any object of the video_transforms file, used for applying video transformations (see video_transforms.py file). Defaults to None.
        - name: String for declaring the set that the iterator is made for (e.g. train/test). Defaults to "<NO_NAME>".
        - force_colour: Boolean that is used to determine if the video will be have a single/three channels. Defaults to True.
        - return_video_path: Boolean for returning or not the full video filepath after a video is loaded. Defaults to False.
        - randomise: Boolean for additional randomisation based on date and time. Defaults to True.
        - shuffle_list_seed: Integer, for random shuffling. Defaults to None.
        - is_jpg: Boolean for using image folders. Defaults to False.

    [Methods]

        - __init__ : Class initialiser
        - size_setter: Sets `Video` object's `clip_size` parameter. Useful if multiggrid training is to be used.
        - shuffle: Sets the random seed for video sampling.
        - getitem_array_from_video: Returns a numpy array containing the frames, an integer for the video label and the full video database path. The input to the function is the index of the video (corresponding to an element in the video_dict)
        - __getitem__ : Wrapper function for the getitem_array_from_video function. Can return either the numpy array of frames with also the curresponding lable in int format as well as the complete filepath of the video (if specified by the user)
        - remove_indices: remove items from the video indices array.
        - __len__ : Returning the length/size of the dataset.
        - indices_list: Return the list of video indices.
        - get_video_dict : Main function for creating the video dictionary. Taks as arguments the location of the dataset (directory) the filepath to the .CSV file containing the dataset info and a boolean variable named include_timeslices which is used in the instance that the video folders also include the time segments in their name (defualts to False).

'''
class VideoIter(data.Dataset):

    def __init__(self,
                 dataset_location,
                 csv_filepath,
                 video_per,
                 include_timeslices,
                 video_size,
                 sampler,
                 num_samplers,
                 video_transform=None,
                 name="<NO_NAME>",
                 return_video_path=False,
                 randomise = True,
                 shuffle_list_seed=None,
                 is_jpg=False):

        super(VideoIter, self).__init__()

        # Class parameter initialisation
        self.clip_size = video_size
        self.sampler = sampler
        self.video_per = video_per
        self.dataset_location = dataset_location
        self.video_transform = video_transform
        self.return_video_path = return_video_path
        self.rng = np.random.RandomState(shuffle_list_seed if shuffle_list_seed else 0)
        self.randomise = randomise
        self.is_jpg = is_jpg

        assert num_samplers >= 1, 'VideoIter: The number of samplers cannot be smaller than 1!'
        self.num_samplers = num_samplers

        # Additional randomisation
        if randomise:
            t = int(time.time())
            self.rng = np.random.RandomState(shuffle_list_seed+t if shuffle_list_seed else t)

        # load video dictionary
        self.video_dict = self.get_video_dict(location=dataset_location,csv_file=csv_filepath,include_timeslices=include_timeslices)

        # Create array to hold the video indices
        self.indices = list(self.video_dict.keys())

        # Shuffling indices array
        if shuffle_list_seed is not None:
            self.rng.shuffle(self.indices)


        logging.info("VideoIter:: iterator initialized (phase: '{:s}', num: {:d})".format(name, len(self.indices)))


    def size_setter(self,new_size):
        self.clip_size = new_size
        self.sampler.set_num(new_size[0])

    def shuffle(self,seed):
        self.rng = np.random.RandomState(seed)
        if self.randomise:
            t = int(time.time())
            self.rng = np.random.RandomState(seed+t)
        self.rng.shuffle(self.indices)



    def getitem_array_from_video(self, index):
        # get current video info
        v_id, label, vid_path, frame_count = self.video_dict.get(index)

        try:

            # Create Video object
            video = Video(vid_path=vid_path,video_transform=self.video_transform,end_size=self.clip_size,precentage=self.video_per,images=self.is_jpg)
            if frame_count < 0:
                frame_count = video.count_frames()

            #print('sampling frames...',type(frame_count),type(v_id))
            # dynamic sampling
            sampled_frames = []
            #logging.info('GETTING ITEM')
            # Get indices for every sampler
            for s in range(1,self.num_samplers+1):
                # generating indices
                range_max = int(frame_count * (s/self.num_samplers))
                sampled_indices = self.sampler.sampling(range_max=range_max-1, v_id=v_id)
                # extracting frames
                sampled_frames.append(video.extract_frames(indices=sampled_indices).unsqueeze(0))

            # create tensor
            sampled_frames = torch.cat(sampled_frames, dim=0)

        except IOError as e:
            logging.warning(">> I/O error({0}): {1}".format(e.errno, e.strerror))

        #print('Processed item w/ index: ',v_id, 'and shape',sampled_frames.shape)
        return sampled_frames, label, vid_path



    def __getitem__(self, index):
        succ = False
        while not succ:
            try:
                index = int(index)
                if (index == 0):
                    index += 1
                frames, label, vid_path = self.getitem_array_from_video(index)
                _, _, t, h, w = frames.size()
                if (t!=self.clip_size[0] and h!=self.clip_size[1] and w!=self.clip_size[0]):
                    raise Exception('Clip size should be ({},{},{}), got clip with: ({},{},{})'.format(*self.clip_size,t,h,w))
                succ = True
            except Exception as e:

                _, exc_obj, tb = sys.exc_info()
                f = tb.tb_frame
                #lineno = tb.tb_lineno
                filename = f.f_code.co_filename
                linecache.checkcache(filename)
                #line = linecache.getline(filename, lineno, f.f_globals)
                #message = 'Exception in ({}, line {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj)

                #prev_index = index
                #index = self.rng.choice(range(0, self.__len__()))
                d_time = int(round(datetime.now().timestamp() * 1000)) # Ensure randomisation
                index = random.randrange(d_time % self.__len__())
                #logging.warning("VideoIter:: Warning: {}".format(message))
                #logging.warning("VideoIter:: Inital index of {} changed to index:{}".format(prev_index,index))

        if self.return_video_path:
            return frames, label, vid_path
        else:
            return frames, label


    def remove_indices(self,d_indices):
        del self.indices[d_indices]

    def __len__(self):
        return len(self.indices)


    def indices_list(self):
        return self.indices


    def get_video_dict(self, location, csv_file, include_timeslices=True):

        # Esure that both given dataset location and csv filepath do exist
        assert os.path.exists(location), "VideoIter:: failed to locate dataset at given location: `{}'".format(location)
        assert os.path.exists(csv_file), "VideoIter:: failed to locate csv file at given location: `{}'".format(csv_file)

        # building dataset
        # - videos_dict : Used to store all videos in a dictionary format with video_index(key) : video_info(value)
        # - logging_interval: The number of iterations performed before writting to the logger
        # - found_videos: Integer for counting the videos from the csv file that are indeed part of the dataset in given location
        # - processed_ids: List of all ids_processed in the current logging_interval
        # - labels_dict: Dictionary for associating string labels with ints
        # - labels_last_index: Integer of keeping track of the integers used


        videos_dict = {}
        logging_interval = 10000
        found_videos = 0
        processed_ids = []


        # Store dictionary of labels keys:'str' , values:'int' to a .JSON file (as a common reference between dataset sets)
        if ('train' in csv_file):
            labels_dict_filepath = csv_file.split('train')[0]+'dictionary.json'
        elif ('val' in csv_file):
            labels_dict_filepath = csv_file.split('val')[0]+'dictionary.json'
        else:
            labels_dict_filepath = csv_file.split('test')[0]+'dictionary.json'

        if (os.path.exists(labels_dict_filepath)):

            with open(labels_dict_filepath) as json_dict:
                labels_dict = json.loads(json_dict.read())
            label_last_index =len(labels_dict)
        else:
            labels_dict = {}
            label_last_index = 0

        for i,line in enumerate(csv.DictReader(open(csv_file))):

            if (i%logging_interval == 0):
                print("VideoIter:: Processed {:d}/{:d} videos".format(found_videos,i, ), end='\r')
                #str(processed_ids).strip('[]')))
                sys.stdout.flush()
                processed_ids = []

            # Account for folder name of different datasets (e.g. `youtube_id` , `id`)
            if ('youtube_id' in line):
                id = line.get('youtube_id').strip()
            else:
                id = line.get('id').strip()

            processed_ids.append(id)

            video_path = os.path.join(location, line.get('label'), id)

            # Check if label has already been found and if not add it to the dictionary:
            if not (line.get('label') in labels_dict):
                labels_dict[line.get('label')] = label_last_index
                label_last_index += 1

            # Case that the filename also includes the timeslices
            if (include_timeslices):

                # Convert path in case that it does not include time slices
                if os.path.exists(video_path):
                    os.rename(video_path, video_path + ('_%06d'%int(float(line.get('time_start')))+('_%06d'%int(float(line.get('time_end'))))))

                video_path = video_path + ('_%06d'%int(float(line.get('time_start')))+('_%06d'%int(float(line.get('time_end')))))

            # Check if video indeed exists (handler for not downloaded videos)
            extensions = ['.mp4','.avi','m4a']
            found = False
            if self.is_jpg:
                found = os.path.isdir(video_path)
            else:
                for e in extensions:
                    f = video_path+e
                    if os.path.isfile(f):
                        found = True
                        video_path+=e
                        break
            
            if not found:
                # Uncomment line for additional user feedback
                #logging.warning("VideoIter:: >> cannot locate `{}'".format(video_path))
                continue

            # Increase videos count and read number of frames
            else:
                found_videos += 1
                if self.is_jpg:
                    frame_count = len(glob.glob(f"{video_path}/*.jpg"))
                else:
                    cap = cv2.VideoCapture(video_path)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Append video info to dictionary
            info = [found_videos, line.get('label'), video_path, frame_count]
            videos_dict[found_videos] = info

        # Convert to indicing in alphabetical order
        for j,key in enumerate(sorted(labels_dict.keys())):
            labels_dict[key] = j

        # Convert `videos_dict` labels to numeric
        for key,value in videos_dict.items():
            videos_dict[key][1] = labels_dict[videos_dict[key][1]]

        for k in sorted(labels_dict.keys()):
            num = 0
            for key,value in videos_dict.items():
                if (videos_dict[key][1] == labels_dict[k]):
                    num+=1

            # Additional per-class information
            #logging.info(k,labels_dict[k],'number of examples:',num)


        logging.info("VideoIter:: - Found: {:d}/{:d} videos from csv file".format(found_videos, i))


        # Save dictionary if it does not already exists
        if not (os.path.exists(labels_dict_filepath)):
            logging.info("VideoIter:: Dictionary saved at {} \n".format(labels_dict_filepath))
            with open(labels_dict_filepath,'w') as json_dict:
                json.dump(labels_dict,json_dict)
        else:
            logging.info("VideoIter:: Found dict at: {} \n".format(labels_dict_filepath))


        return videos_dict



'''
===  E N D  O F  C L A S S  V I D E O I T E R A T O R  ===
'''
