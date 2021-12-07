'''
---  I M P O R T  S T A T E M E N T S  ---
'''
import os
os.environ['NUMEXPR_MAX_THREADS'] = '16'
import sys
import json
import socket
import datetime
import coloredlogs, logging
coloredlogs.install()
import argparse
import yaml
import math
import imgaug
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import dataset
from network.symbol_builder import Combined, get_pooling
from network.config import get_config
from data import iterator_factory
from run import metric
from run.model import model
from run.lr_scheduler import MultiFactorScheduler

from torch.optim import SGD, Adam, AdamW
#from torchlars import LARS
#from pytorch_lamb import Lamb

from decimal import Decimal

datasets = [ 'mini-Kinetics',
             'Kinetics-400',
             'Kinetics-600',
             'Kinetics-700',
             'Moments-in-Time',
             'HACS',
             'ActivityNet-v1',
             'ActivityNet-v2',
             'NTU-RGB'
             'UCF-101',
             'HMDB-51']

# Create main parser
parser = argparse.ArgumentParser(description="PyTorch parser for early action prediction from videos")

# debug parser arguments
parser.add_argument('--random_seed', type=int, default=1,
                    help='set seeding (default: 1).')
parser.add_argument('--print_net', type=bool, default=False,
                    help="print the architecture.")

# visible video precentage
parser.add_argument('--video_per', type=float, default=None,
                    help='precentage of the video to be used for prediction.')
parser.add_argument('--num_samplers', type=int, default=3,
                    help='number of video samplers. The window from which frames are sampled from will progressively increase based on `num_frames`*`s`/`num_samplers` for `s` in range(`num_samplers`).')

# data loading parser arguments
parser.add_argument('--dataset', default='HACS', choices=datasets,
                    help="name of the dataset")
parser.add_argument('--data_dir', default='data/',
                    help="path for the video files \n ---- Note that the allowed formats are: ---- \n -> video (.mp4, .mpeg, .avi) \n -> image (.jpg, .jpeg, .png) \n -> SQL with frames encoded as BLOBs (.sql) \n See advice in the README about the directory structure.")
parser.add_argument('--label_dir', default='labels/',
                    help="path for the label files associated with the dataset.")

# training and validation params parser arguments
parser.add_argument('--precision', default='fp32', choices=['fp32','mixed'],
                    help="switch between single (fp32)/mixed (fp16) precision.")
parser.add_argument('--frame_len', default=16,
                    help="define the (max) frame length of each input sample.")
parser.add_argument('--frame_size', default=224,
                    help="define the (max) frame size of each input sample.")
parser.add_argument('--val_frame_interval', type=int, default=[1,2],
                    help="define the sampling interval between frames.")

# GPU-device related parser arguments
parser.add_argument('--gpus', type=list, default=[0,1],
                    help="define gpu id(s).")

# DL model parser arguments
parser.add_argument('--pretrained_dir', type=str,  default=None,
                    help="load pretrained model from path. This can be used for either the backbone or head alone or both. Leave empty when training from scratch.")

parser.add_argument('--backbone', type=str, default='MTnet_s',
                    help="chose the backbone architecture. See `network` dir for more info.")
parser.add_argument('--head', type=str, default='Temper_h',
                    help="chose the head architecture. See `network` dir for more info.")

parser.add_argument('--num_freq_bands', type=int, default = 10,
                    help="choose the number of freq bands, with original value (2 * K + 1)")
parser.add_argument('--max_freq', type=float, default = 10.,
                    help="choose the maximum frequency number.")
parser.add_argument('--num_latents', type=int, default = 512,
                    help="choose number of latents/induced set points/centroids (following terminology from the Perceiver/Set Transformer papers).")
parser.add_argument('--latent_dim', type=int, default = 512,
                    help="latent dimension size.")
parser.add_argument('--cross_heads', type=int, default = 1,
                    help = "number of cross-head attention layers.")
parser.add_argument('--latent_heads', type=int, default = 8,
                    help= "number of latent head attention moduls.")
parser.add_argument('--cross_dim_head', type=int, default = 64,
                    help="number of dimensions per cross attention head.")
parser.add_argument('--latent_dim_head', type=int, default = 64,
                    help="number of dimensions per latent self attention head.")
parser.add_argument('--attn_dropout', type=float, default = 0.,
                    help='dropout probability for the cross head and latent attention.')
parser.add_argument('--ff_dropout', type=float, default = 0.,
                    help='dropout probability for the feed-forward sub-net.')
parser.add_argument('--weight_tie_layers', type=bool, default = False,
                    help="whether to weight tie layers (optional).")

parser.add_argument('--pool', type=str, default='avg', choices=['max','avg','em','edscw','idw','ada'],
                    help='choice of pooling method to use for selection/fusion of frame features.')

parser.add_argument('--workers', type=int, default=8,
                    help='define the number of workers.')

# YAML loader
parser.add_argument('--config', type=str, default=None,
                    help="YAML configuration file to load parser arguments from.")

'''
---  S T A R T  O F  F U N C T I O N  A U T O F I L L  ---
    [About]
        Function for creating log directories based on the parser arguments
    [Args]
        - args: ArgumentParser object containg both the name of task (if empty a default folder is created) and the log file to be created.
    [Returns]
        - args: ArgumentParser object with additionally including the model directory and the model prefix.
'''
def autofill(args, parser):
    # fix for lr mult empty keys
    defaults = vars(parser.parse_args([]))
    for key in defaults['lr_mult']:
        if key not in args.lr_mult.keys():
            args.lr_mult[key] = defaults['lr_mult'][key]

    # customized
    if args.log_file is None:
        print('LOG FILE NOT INITIALISED')
        if not os.path.exists("logs"):
            os.makedirs("logs")
        now = datetime.datetime.now()
        date = str(now.year) + '-' + str(now.month) + '-' + str(now.day)
        args.log_file = "logs/{}_inference_at-{}_datetime_{}.log".format('video_pred', socket.gethostname(), date)

    if args.head:
        args.model_dir = os.path.join(args.results_dir,args.head+'_'+args.backbone)
    else:
        args.model_dir = os.path.join(args.results_dir,args.backbone)


    return args
'''
---  E N D  O F  F U N C T I O N  A U T O F I L L  ---
'''


'''
---  S T A R T  O F  M A I N  F U N C T I O N  ---
'''
if __name__ == "__main__":

    # set args & overwrite if YAML file is used
    args = parser.parse_args()
    args = autofill(args, parser)

    if args.config is not None:
        # load YAML file options
        print(args.config)
        opt = yaml.load(open(args.config), Loader=yaml.FullLoader)
        # overwrite arguments based on YAML options
        vars(args).update(opt)


    # Use file logger + console output (for servers and real-time feedback)
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(args.log_file)
    formatter = logging.Formatter('%(asctime)s: %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)


    logging.info("Using pytorch version {} ({})".format(torch.__version__, torch.__path__))
    logging.info("Start training with args:\n" + json.dumps(vars(args), indent=4, sort_keys=True))
    # Set device states
    logging.info('CUDA availability: '+str(torch.cuda.is_available()))
    os.environ["CUDA_VISIBLE_DEVICES"] = ''.join(str(id)+',' for id in args.gpus)[:-1] # before using torch
    logging.info('CUDA_VISIBLE_DEVICES set to '+os.environ["CUDA_VISIBLE_DEVICES"])
    assert torch.cuda.is_available(), "CUDA is not available. CUDA devices are required from this repo!"
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    clip_length = int(args.frame_len)
    clip_size = int(args.frame_size)

    # Assign values from kwargs
    model_prefix = args.model_dir


    # Load dataset related configuration
    dataset_cfg = dataset.get_config(name=args.dataset)

    # Load model related configuration
    if args.head:
        input_conf = get_config(name=args.head+' w/ '+args.backbone)
    else:
        input_conf = get_config(name=args.backbone)

    # training parameters intialisation
    kwargs = {}
    kwargs.update(dataset_cfg)
    kwargs.update(vars(args))
    kwargs['input_conf'] = input_conf

    # `Combined` object for grouping backbone and head models.
    net = Combined(**kwargs)

    # Create model
    net = model(net=net,
                criterion=torch.nn.CrossEntropyLoss().cuda(),
                model_prefix=model_prefix,
                step_callback_freq=1,
                save_checkpoint_freq=1000)
    net.net.cuda()

    # data iterator - randomisation based on date and time values
    iter_seed = torch.initial_seed() + 100
    now = datetime.datetime.now()
    iter_seed += now.year + now.month + now.day + now.hour + now.minute + now.second

    # Get parent location
    # - `data` folder should include all the dataset examples.
    # - `labels` folder should inclde all labels in .csv format.
    # We use a global label formating - you can have a look at the link in the `README.md` to download the files.

    train_loaders = {}

    print()

    # Create custom loaders for train and validation
    eval_loader = iterator_factory.create(
        return_train=False,
        data_dir=args.data_dir ,
        labels_dir=args.label_dir ,
        video_per_val=val_per,
        num_samplers=args.num_samplers,
        batch_size=1,
        clip_length=clip_length,
        clip_size=clip_size,
        val_clip_length=clip_length,
        val_clip_size=clip_size,
        include_timeslices = dataset_cfg['include_timeslices'],
        val_interval=args.val_frame_interval,
        mean=input_conf['mean'],
        std=input_conf['std'],
        seed=iter_seed,
        num_workers=args.workers)

    print()


    # mixed or single precision based on argument parser
    if args.precision=='mixed':
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler=None

    # Create DataParallel wrapper
    net.net = torch.nn.DataParallel(net.net, device_ids=[gpu_id for gpu_id in (args.gpus)])

    # checkpoint loading
    _, _ = net.load_checkpoint(path=args.pretrained_dir)


    # define evaluation metric
    metrics = metric.MetricList(metric.Loss(name="loss-ce"),
                                metric.Accuracy(name="top1", topk=1),
                                metric.Accuracy(name="top5", topk=5))
    # enable cudnn tune
    cudnn.benchmark = True

    logging.info('LRScheduler: The learning rate will change at steps: '+str([x*num_steps for x in args.lr_steps]))

    # Main training happens here
    net.inference(eval_iter=eval_loader,
                  workers=args.workers,
                  metrics=metrics,
                  iter_per_epoch=num_steps,
                  precision=args.precision,
                  scaler=scaler)

'''
---  E N D  O F  M A I N  F U N C T I O N  ---
'''
