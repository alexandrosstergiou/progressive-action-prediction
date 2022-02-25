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

import dataset
from network.symbol_builder import Combined, get_pooling
from network.config import get_config
from data import iterator_factory
from run import metric
from run.model import model
from run.lr_scheduler import MultiFactorScheduler

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
             'HMDB-51',
             'smthng-smthng_coarse',
             'smthng-smthng_fine',
             'smthng-smthng_sub21',
             'smthng-smthng_v2']

# Create main parser
parser = argparse.ArgumentParser(description="PyTorch parser for early action prediction from videos")

# debug parser arguments
parser.add_argument('--random_seed', type=int, default=1,
                    help='set seeding (default: 1).')
parser.add_argument('--print_net', type=bool, default=False,
                    help="print the architecture.")

# visible video precentage
parser.add_argument('--video_per', type=float, default=None,
                    help='precentage of the video to be used for prediction. Will overwrite if precentages for train/val videos are set individually.')
parser.add_argument('--video_per_train', type=float, default=.4,
                    help='precentage of the video to be used during training.')
parser.add_argument('--video_per_val', type=float, default=.4,
                    help='precentage of the video to be used for prediction during validation.')
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
parser.add_argument('--frame_size', default=(224,224),
                    help="define the (max) frame size of each input sample.")
parser.add_argument('--train_frame_interval', default=[1,2,3,4], nargs='+',
                    help="define the sampling interval between frames.")
parser.add_argument('--val_frame_interval', default=[1,2], nargs='+',
                    help="define the sampling interval between frames.")
parser.add_argument('--batch_size', type=int, default=16,
                    help="batch size")
parser.add_argument('--long_cycles', type=bool, default=False,
                    help="enable long cycles for batches (Multigrid training).")
parser.add_argument('--short_cycles', type=bool, default=False,
                    help="enable short cycles for batches (Multigrid training).")
parser.add_argument('--end_epoch', type=int, default=60,
                    help="maxmium number of training epoch.")


parser.add_argument('--optimiser', type=str, default='AdamW', choices=['AdamW', 'SGD', 'Adam'],
                    help='name of the optimiser to be used.')

parser.add_argument('--lr_base', type=float, default=1e-2,
                    help="base learning rate.")
# Should be set in YAML config file (not possible through argparse)
parser.add_argument('--lr_mult', type=dict, default={'head':.1,'pool':.1,'classifier':1.0},
                    help="learning rate multipliers for different sets of parameters. Acceptable keys include:\n - `head`: for the lr multiplier of the head (temporal) network. Default value is 1.0. \n - `gates`: for the lr multiplier of the per-frame exiting gates. Default value is 0.0. \n - `pool`: For the pooling method. this is only used in the pooling method is parameterised.Default value is 1e-4. \n - `classifier`: for the `fc` clasifier of the network. Default value is 0.0. \n ")
parser.add_argument('--lr_steps', default=[14, 32, 44], nargs='+',
                    help="epochs in which the (base) learning rate will change.")
parser.add_argument('--lr_factor', type=float, default=0.1,
                    help="reduce the learning based on factor.")
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help="weight decay.")

# storing parser arguments
parser.add_argument('--results_dir', type=str, default="./results",
                    help="folder for logging accuracy and saving models.")
parser.add_argument('--save_frequency', type=float, default=60,
                    help="save once after N epochs.")
parser.add_argument('--log_file', type=str, default=None,
                    help="set logging file.")

# GPU-device related parser arguments
parser.add_argument('--gpus', default=[0,1], nargs='+',
                    help="define gpu id(s).")

# DL model parser arguments
parser.add_argument('--pretrained_dir', type=str,  default=None,
                    help="load pretrained model from path. This can be used for either the backbone or head alone or both. Leave empty when training from scratch.")

parser.add_argument('--backbone', type=str, default='r3d_18',
                    help="chose the backbone architecture. See `network` dir for more info.")
parser.add_argument('--head', type=str, default='Tempr_h',
                    help="chose the head architecture. See `network` dir for more info.")

parser.add_argument('--num_freq_bands', type=int, default = 10,
                    help="choose the number of freq bands, with original value (2 * K + 1)")
parser.add_argument('--max_freq', type=float, default = 10.,
                    help="choose the maximum frequency number.")
parser.add_argument('--num_latents', type=int, default = 256,
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

parser.add_argument('--pool', type=str, default='ada', choices=['max','avg','em','edscw','idw','ada'],
                    help='choice of pooling method to use for selection/fusion of frame features.')

parser.add_argument('--workers', type=int, default=8,
                    help='define the number of workers.')


# optimization parser arguments
parser.add_argument('--resume_epoch', type=int, default=0,
                    help="resuming train from defined epoch.")

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
        if not os.path.exists("logs"):
            os.makedirs("logs")
        now = datetime.datetime.now()
        date = str(now.year) + '-' + str(now.month) + '-' + str(now.day)
        final_str = 'observation_ratio_'+str(args.video_per)+'_'+args.head+'_'+args.backbone+'_'+args.pool
        args.log_file = "logs/{}_at-{}_datetime_{}_with_{}.log".format('video_pred', socket.gethostname(), date, final_str)

    ratio = 'observation_ratio_'+str(args.video_per)
    if args.head:
        args.model_dir = os.path.join(args.results_dir,ratio,args.head+'_'+args.backbone+'_'+args.pool)
    else:
        args.model_dir = os.path.join(args.results_dir,ratio,args.backbone)


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

    if args.config is not None:
        # load YAML file options
        print(args.config)
        opt = yaml.load(open(args.config), Loader=yaml.FullLoader)
        # overwrite arguments based on YAML options
        vars(args).update(opt)
    args = autofill(args, parser)


    # Use file logger + console output (for servers and real-time feedback)
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(args.log_file)
    formatter = logging.Formatter('%(asctime)s: %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # handle strings in argparse lists
    args.gpus = [int(i) for i in args.gpus]
    args.lr_steps = [int(i) for i in args.lr_steps]
    args.train_frame_interval = [int(i) for i in args.train_frame_interval]
    args.val_frame_interval = [int(i) for i in args.val_frame_interval]

    # must set visible devices BEFORE importing torch
    if (len(args.gpus) == 1):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus[0])
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ''.join(str(id)+',' for id in args.gpus)[:-1]
    logging.info('CUDA_VISIBLE_DEVICES set to '+os.environ["CUDA_VISIBLE_DEVICES"])

    # import torch
    import torch
    import torch.nn.parallel
    import torch.backends.cudnn as cudnn
    import torch.distributed as dist
    from torch.optim import SGD, Adam, AdamW
    #from torchlars import LARS
    #from pytorch_lamb import Lamb

    logging.info("Using pytorch version {} ({})".format(torch.__version__, torch.__path__))
    logging.info("Start training with args:\n" + json.dumps(vars(args), indent=4, sort_keys=True))
    # Set device states
    logging.info('CUDA availability: '+str(torch.cuda.is_available()))
    assert torch.cuda.is_available(), "CUDA is not available. CUDA devices are required from this repo!"
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    clip_length = int(args.frame_len)
    clip_size = args.frame_size
    if str(clip_size).isdigit():
        clip_size = (int(clip_size),int(clip_size))
    else:
        clip_size = (int(clip_size[0]),int(clip_size[1]))

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
                save_checkpoint_freq=args.save_frequency)
    net.net.cuda()

    # Make results directory for .csv files if it does not exist
    ratio = 'observation_ratio_'+str(args.video_per)
    samplers = 'samplers_'+str(args.num_samplers)
    latents = 'latents_'+str(args.num_latents)+'_heads_'+str(args.latent_heads)
    if args.head:
        results_path = os.path.join(args.results_dir,args.dataset,latents,samplers,ratio,args.head+'_'+args.backbone+'_'+args.pool)
    else:
        results_path = os.path.join(args.results_dir,args.dataset,latents,samplers,ratio,args.backbone+'_'+args.pool)
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # data iterator - randomisation based on date and time values
    iter_seed = torch.initial_seed() + 100 + max(0, args.resume_epoch) * 100
    now = datetime.datetime.now()
    iter_seed += now.year + now.month + now.day + now.hour + now.minute + now.second

    # Get parent location
    # - `data` folder should include all the dataset examples.
    # - `labels` folder should inclde all labels in .csv format.
    # We use a global label formating - you can have a look at the link in the `README.md` to download the files.

    train_loaders = {}

    print()

    # overwrite precentages if `video_per` is defined
    if args.video_per is not None:
        train_per = args.video_per
        val_per = args.video_per
    else:
        train_per = args.video_per_train
        val_per = args.video_per_val

    # Create custom loaders for train and validation
    train_data, eval_loader, train_length = iterator_factory.create(
        data_dir=args.data_dir ,
        labels_dir=args.label_dir ,
        video_per_train=train_per,
        video_per_val=val_per,
        num_samplers=args.num_samplers,
        batch_size=args.batch_size,
        return_len=True,
        clip_length=clip_length,
        clip_size=clip_size,
        val_clip_length=clip_length,
        val_clip_size=clip_size,
        include_timeslices = dataset_cfg['include_timeslices'],
        train_interval=args.train_frame_interval,
        val_interval=args.val_frame_interval,
        mean=input_conf['mean'],
        std=input_conf['std'],
        seed=iter_seed,
        num_workers=args.workers)

    print()
    # Parameter LR configuration for optimiser
    # Base layers are based on the layers as loaded to the model
    params = {
        'classifier':{'lr':args.lr_mult['classifier'],
                'params':[]},
        'head':{'lr':args.lr_mult['head'],
                'params':[]},
        'pool':{'lr':args.lr_mult['pool'],
                'params':[]},
        'base':{'lr':args.lr_base,
                'params':[]}
    }

    # Iterate over all parameters
    for name, param in net.net.named_parameters():
        if 'fc' in name.lower():
            params['classifier']['params'].append(param)
        elif 'head' in name.lower():
            params['head']['params'].append(param)
        elif 'pred_fusion' in name.lower():
            params['pool']['params'].append(param)
        params['base']['params'].append(param)

    # User feedback
    for key in params.keys():
        if key == 'base':
            logging.info("Optimiser:: - \033[35m{}\033[0m lr is set to \033[35m{:.1e}\033[0m for \033[35m{}\033[0m params".format(key, params[key]['lr'], len(params[key]['params'])))
        else:
            lr_n = params['base']['lr']*params[key]['lr']
            logging.info("Optimiser:: - \033[35m{}\033[0m lr is set to \033[35m{:.1e}\033[0m for \033[35m{}\033[0m params".format(key, Decimal(lr_n),len(params[key]['params'])))

    if args.optimiser=='SGD':
        optimiser = torch.optim.SGD([
            {'params': params['classifier']['params'], 'lr_mult': params['classifier']['lr']},
            {'params': params['head']['params'], 'lr_mult': params['head']['lr']},
            {'params': params['pool']['params'], 'lr_mult': params['pool']['lr']},],
            lr=args.lr_base,
            momentum=0.9,
            weight_decay=args.weight_decay,
            nesterov=True)
    elif args.optimiser=='Adam':
        optimiser = torch.optim.Adam([
            {'params': params['classifier']['params'], 'lr_mult': params['classifier']['lr']},
            {'params': params['head']['params'], 'lr_mult': params['head']['lr']},
            {'params': params['pool']['params'], 'lr_mult': params['pool']['lr']},],
            lr=args.lr_base,
            weight_decay=args.weight_decay)
    elif args.optimiser=='AdamW':
        optimiser = torch.optim.AdamW([
            {'params': params['classifier']['params'], 'lr_mult': params['classifier']['lr']},
            {'params': params['head']['params'], 'lr_mult': params['head']['lr']},
            {'params': params['pool']['params'], 'lr_mult': params['pool']['lr']},],
            lr=args.lr_base,
            weight_decay=args.weight_decay)
    else:
        logging.error('Optimiser:: Initialisation of optimiser failed! No implementation available for optimiser named {}'.format(args.optimiser))
        raise NotImplemented

    # mixed or single precision based on argument parser
    if args.precision=='mixed':
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler=None

    # Create DataParallel wrapper
    net.net = torch.nn.DataParallel(net.net, device_ids=[gpu_id for gpu_id in range(torch.cuda.device_count())])

    num_steps = train_length // args.batch_size
    print()
    logging.info("IterScheduler:: Each epoch will have {:d} iterations based on batch size {:d}".format(num_steps,args.batch_size))

    # Long Cycle steps
    if (args.long_cycles):

        count = 0
        index = 0
        iter_sizes = [8, 4, 2, 1]
        initial_num = num_steps

        # Expected to find the number of batches that fit exactly to the number of iterations.
        # So the sum of the floowing batch sizes should be less or equal to the number of batches left.
        while sum(iter_sizes[index:]) <= num_steps:
            # Case 1: 8 x B
            if iter_sizes[index] == 8:
                count += 1
                index = 1
                num_steps -= 8
            # Case 2: 4 x B
            elif iter_sizes[index] == 4:
                count += 1
                index = 2
                num_steps -= 4
            # Case 3: 2 x B
            elif iter_sizes[index] == 2:
                count += 1
                index = 3
                num_steps -= 2
            # Base case
            elif iter_sizes[index] == 1:
                count += 1
                index = 0
                num_steps -= 1

        logging.info("MultiGridBatchScheduler:: New number of batches per epoch is {:d} being equivalent to {:1.3f} of original number of batches with Long cycles".format(count,float(count)/float(initial_num)))
        num_steps = count

    # Short Cycle steps
    if (args.short_cycles):

        # Iterate for *every* batch
        i = 0

        while i <= num_steps:
            m = i%3
            # Case 1: Base case
            if (m==0):
                num_steps -= 1
            # Case 2: b = 2 x B
            if (m==1):
                num_steps -= 2
            # Case 3: b = 4 x B
            else:
                num_steps -= 4

            i += 1

        # Update new number of batches
        logging.info("MultiGridBatchScheduler:: New number of batches per epoch is {:d} being equivalent to {:1.3f} of original number of batches with Short cycles".format(i,float(i)/float(initial_num)))
        num_steps = i

    # Split the batch number to four for every change in the long cycles
    long_steps = None
    if (args.long_cycles):
        step = num_steps//4
        long_steps = list(range(num_steps))[0::step]
        num_steps = long_steps[-1]

        # Create full list of long steps (for all batches)
        for epoch in range(1,args.end_epoch):
            end = long_steps[-1]
            long_steps = long_steps + [x.__add__(end) for x in long_steps[-4:]]

        # Fool-proofing
        if (long_steps[0]==0):
            long_steps[0]=1

    # Options acceptable on training:
    #   - `resume_epoch` == 0 and `pretrained_dir` is None : Training from scratch.
    #   - `resume_epoch` == 0 and `pretrained_dir` is not None: Fine-tuning (load only checkpoint).
    #   - `resume_epoch` != 0 and `pretrained_dir` is not None: Resume training (load entire "state_dict").
    #   - `resume_epoch` != 0 and `pretrained_dir` is None: N/A catch with assert


    assert not(args.resume_epoch != 0 and args.pretrained_dir is None), 'Initialiser::  Error in training configuration occured! Cannot use argument `resume_epoch` with non-zero integer without specifying the `pretrained_dir` string directory to load weights from!'

    # resume training: model and optimiser - (account of various batch sizes)
    if args.resume_epoch == 0:
        if args.pretrained_dir is None:
            # Train from scratch
            epoch_start = 0
            step_counter = 0
        else:
            # Fine tuning
            _, optimiser = net.load_checkpoint(path=args.pretrained_dir, optimiser=optimiser)
        epoch_start = 0
        step_counter = 0
    else:
        # Resume training
        epoch, _ = net.load_checkpoint(path=args.pretrained_dir, epoch=args.resume_epoch)
        epoch_start = args.resume_epoch # change if you are to use "state_dict"'s epoch
        step_counter = epoch_start * num_steps


        # Try to load previous state dict in case `pretrained_dir` is None
        if not args.pretrained_dir:
            try:
                net.load_checkpoint(epoch=args.resume_epoch, optimizer=optimiser)
            except Exception:
                logging.warning('Initialiser:: No previous checkpoint found! You can specify the file path explicitly with `pretrained_dir` argument.')
        epoch_start = args.resume_epoch
        step_counter = epoch_start * num_steps

    # Step dictionary creation
    iteration_steps = {'long_0':[],'long_1':[],'long_2':[],'long_3':[],'short_0':[],'short_1':[],'short_2':[]}
    #Populate dictionary
    for batch_i in range(0,num_steps):
        if (args.long_cycles):
            # Long cycle cases
            if batch_i>=0 and batch_i<num_steps//4:
                iteration_steps['long_0'].append(batch_i)
            elif batch_i>=num_steps//4 and batch_i<num_steps//2:
                iteration_steps['long_1'].append(batch_i)
            elif batch_i>=num_steps//2 and batch_i<(3*num_steps)//4:
                iteration_steps['long_2'].append(batch_i)
            else:
                iteration_steps['long_3'].append(batch_i)

        if (args.short_cycles):
            # Short cases
            if (batch_i%3==0):
                iteration_steps['short_0'].append(batch_i)
            elif (batch_i%3==1):
                iteration_steps['short_1'].append(batch_i)
            else:
                iteration_steps['short_2'].append(batch_i)

    # set learning rate scheduler
    lr_scheduler = MultiFactorScheduler(base_lr=args.lr_base,
                                        steps=[x*num_steps for x in args.lr_steps],
                                        iterations_per_epoch=num_steps,
                                        iteration_steps=iteration_steps,
                                        factor=args.lr_factor,
                                        step_counter=step_counter)
    # define evaluation metric
    metrics = metric.MetricList(metric.Loss(name="loss-ce"),
                                metric.Accuracy(name="top1", topk=1),
                                metric.Accuracy(name="top5", topk=5),
                                metric.BatchSize(name="batch_size"),
                                metric.LearningRate(name="lr"))

    sampler_metrics = metric.MetricList(metric.Loss(name="loss-ce"),
                                metric.Accuracy(name="top1", topk=1),
                                metric.Accuracy(name="top5", topk=5))
    # enable cudnn tune
    cudnn.benchmark = True

    logging.info('LRScheduler: The learning rate will change at steps: '+str([x*num_steps for x in args.lr_steps]))

    # Main training happens here
    net.fit(train_iter=train_data,
            eval_iter=eval_loader,
            batch_shape=(int(args.batch_size),int(clip_length),int(clip_size[0]),int(clip_size[1])),
            workers=args.workers,
            no_cycles=(not(args.long_cycles) and not(args.short_cycles)),
            optimiser=optimiser,
            long_short_steps_dir=iteration_steps,
            lr_scheduler=lr_scheduler,
            metrics=metrics,
            sampler_metrics_list=[sampler_metrics for _ in range(args.num_samplers)],
            iter_per_epoch=num_steps,
            epoch_start=epoch_start,
            epoch_end=args.end_epoch,
            directory=results_path,
            precision=args.precision,
            scaler=scaler,
            samplers=args.num_samplers)

    # Create custom loaders for train and validation
    eval_loader = iterator_factory.create(
        return_train=False,
        return_video_path=True,
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
'''
---  E N D  O F  M A I N  F U N C T I O N  ---
'''
