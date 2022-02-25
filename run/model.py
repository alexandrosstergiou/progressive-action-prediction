'''
---  I M P O R T  S T A T E M E N T S  ---
'''
import torch
import os, sys
import json
import gc
import csv
import time
import coloredlogs, logging
coloredlogs.install()
import math
from . import metric
from . import callback

import torch.cuda.amp as amp

from einops import rearrange

import collections


'''
===  S T A R T  O F  C L A S S  C S V W R I T E R ===

    [About]

        Class for creating a writer for the training and validations scores and saving them to .csv

    [Init Args]

        - filename: String for the name of the file to save to. Will create the file or erase any
        previous versions of it.

    [Methods]

        - __init__ : Class initialiser
        - close : Function for closing the file.
        - write : Function for writing new elements along in a new row.
        - size : Function for getting the size of the file.
        - fname : Function for returning the filename.

'''
class CSVWriter():

    filename = None
    fp = None
    writer = None

    def __init__(self, filename):
        self.filename = filename
        self.fp = open(self.filename, 'a', encoding='utf8')
        self.writer = csv.writer(self.fp, delimiter=';', quotechar='"', quoting=csv.QUOTE_ALL, lineterminator='\n')

    def close(self):
        self.fp.close()

    def write(self, elems):
        self.writer.writerow(elems)

    def size(self):
        return os.path.getsize(self.filename)

    def fname(self):
        return self.filename
'''
===  E N D  O F  C L A S S  C S V W R I T E R ===
'''


'''
===  S T A R T  O F  C L A S S  S T A T I C M O D E L ===

    [About]

        Class responsible for the main functionality during training. Provides function implementations
        for loading a previous model state, create a checkpoint filepath, loading a checkpoint, saving model
        based on the checkpoint path created as well as preform a full forward pass returning the output
        class probabilities and loss(es).

    [Init Args]

        - net: nn.Module containing the full architecture.
        - criterion : nn.Module that specifies the loss criterion (e.g. CrossEntropyLoss). Could also
        include custom losses.
        - model_prefix : String for the prefix to be used when loading a previous state.

    [Methods]

        - __init__ : Class initialiser
        - load_state : Function for loading model state from `state_dict`. Results in a more flexible loading method where parts of keys such as [`module.`,`model.`,`backbone.`,`head.`] are not considered during loading.
        - load_checkpoint : Function for loading model state, optimiser (if specified) and epoch number (if specified) from file.
        - save_checkpoint : Function for saving model state, optimiser and number of epoch to `.pth` file.
        - forward : Function for calculating categorical cross entropy.

'''
class static_model(object):

    def __init__(self,
                 net,
                 criterion=None,
                 model_prefix='',
                 **kwargs):
        if kwargs:
            logging.warning("Initialiser:: Unknown kwargs: {}".format(kwargs))

        # parameter initialisation
        self.net = net
        self.model_prefix = model_prefix
        self.criterion = criterion

    def load_state(self, state_dict, strict=False):
        # Strict mode that structure should match exactly
        if strict:
            missing_keys, unexpected_keys = self.net.load_state_dict(state_dict=state_dict, strict=False)
            if not missing_keys:
                logging.warning("Initialiser:: The following keys were missing: {}".format(missing_keys))
            if not unexpected_keys:
                logging.warning("Initialiser:: The following keys were not expected: {}".format(unexpected_keys))
        else:
            unused_keys = [key for key in state_dict.keys()]
            # Iteration over both state dicts
            n_state_dict_keys = list(self.net.state_dict().keys())
            for f_name, f_param in state_dict.items():
                for n_name, n_param in self.net.state_dict().items():
                    # Remove module, model, head and backbone from keys on both state dicts
                    f_name_trim = f_name.replace('module.','')
                    n_name_trim = n_name.replace('module.','')
                    f_name_trim = f_name_trim.replace('model.','')
                    n_name_trim = n_name_trim.replace('model.','')
                    f_name_trim = f_name_trim.replace('backbone.','')
                    n_name_trim = n_name_trim.replace('backbone.','')
                    f_name_trim = f_name_trim.replace('head.','')
                    n_name_trim = n_name_trim.replace('head.','')
                    # Check if file param name does match (as a sub-string) to the network param
                    if f_name_trim == n_name_trim:
                        # Check if the shape of the parameters also match
                        if f_param.shape == n_param.shape:
                            # Additional verbose
                            #logging.info('Loaded weights for {} from {}'.format(n_name_trim,f_name_trim))
                            self.net.state_dict()[n_name].copy_(f_param.view(n_param.shape))
                            unused_keys.remove(f_name)
                            try:
                                n_state_dict_keys.remove(n_name)
                            except Exception:
                                logging.warning('Name {} not in param list'.format(n_name))

            # indicating missed keys
            if n_state_dict_keys:
                logging.warning("Initialiser:: The following keys were not loaded: {}".format(n_state_dict_keys))
                return False
            # unused keys
            if unused_keys:
                logging.warning("Initialiser:: The following keys were not used: {}".format(unused_keys))


        return True

    def load_checkpoint(self, path, epoch=None, optimiser=None, strict=False):
        # model prefix needs to be defined for creating the checkpoint path
        assert self.model_prefix, "Undefined `model_prefix`"
        # check that file path exists
        assert os.path.exists(path), logging.warning("Initialiser:: Failed to locate model weights path: `{}'".format(path))
        # checkpoint loading
        checkpoint = torch.load(path)

        for key in checkpoint['state_dict'].keys():
            # For weights from non-DataParallel checkpoints
            if 'module.' not in key:
                new_key = 'module.'+key
            else:
                new_key = key
            checkpoint['state_dict'] = collections.OrderedDict((new_key if k == key else k, v) for k, v in checkpoint['state_dict'].items())

        # Try to load `load_state` for `self.net` first
        all_params_loaded = self.load_state(checkpoint['state_dict'], strict=strict)


        # Optimiser handling
        if optimiser:
            if 'optimizer' in checkpoint.keys() and all_params_loaded:
                optimiser.load_state_dict(checkpoint['optimizer'])
                logging.info("Initialiser::  Model & Optimiser states are resumed from: `{}'".format(path))
            else:
                logging.warning("Initialiser:: Did not load optimiser state from: `{}'".format(path))
        else:
            logging.info("Initialiser:: Only model state resumed from: `{}'".format(path))

        if epoch is not None:
            if 'epoch' in checkpoint.keys():
                logging.info("Initialiser:: Epoch number updated from: {} vs {}".format(epoch, checkpoint['epoch']))
                epoch = checkpoint['epoch']
            else:
                logging.warning("Initialiser:: Unable to find epoch information in `state_dict` of the file provided, defaulting to value {}. If fine-tuning, you can specify this manually instead.".format(epoch))
        return epoch, optimiser


    def save_checkpoint(self, epoch, base_directory, optimiser_state=None, best=False):

        # Create save path
        if not best:
            save_path = os.path.join(base_directory,'{}_ep-{:04d}.pth'.format(base_directory.split('/')[-1],epoch))
        else:
            save_path = os.path.join(base_directory,'{}_best.pth'.format(base_directory.split('/')[-1]))


        # Create directory if path does not exist
        if not os.path.exists(base_directory):
            logging.debug("mkdir {}".format(base_directory))
            os.makedirs(base_directory)

        # Create optimiser state if it does not exist. Use the `epoch` and `state_dict`
        state_dict = self.net.state_dict()

        # Check if `backbone_optimiser_state` is not None
        if optimiser_state is None:
            torch.save({'epoch': epoch,
                        'state_dict': state_dict},
                        save_path)
            logging.debug("Checkpoint (only model) saved to: {}".format(save_path))
        else:
            torch.save({'epoch': epoch,
                        'state_dict': state_dict,
                        'optimizer': optimiser_state},
                        save_path)
            logging.debug("Checkpoint (model & optimiser) saved to: {}".format(save_path))


    def forward(self, data, target, precision):
        # Data conversion
        if precision=='mixed':
            data = data.cuda().half()
        else:
            data =  data.cuda().float()
        target = target.cuda()

        # Forward for training/evaluation
        if self.net.training:
            if precision=='mixed':
                with amp.autocast():
                    outputs = self.net(data)
            else:
                outputs = self.net(data)
        else:
            with torch.no_grad():
                if precision=='mixed':
                    with amp.autocast():
                        outputs = self.net(data)
                else:
                    outputs = self.net(data)
        # Check for tuple
        if isinstance(outputs, tuple):
            output = outputs[0]
            outputs_list = rearrange(outputs[1], 'b s c -> s b c')
            o_list = [output]+[o for o in outputs_list]
        else:
            outputs_list = []

        # Use (loss) criterion if specified
        losses = []
        for out in o_list:
            if hasattr(self, 'criterion') and self.criterion is not None and target is not None:
                if precision=='mixed':
                    with amp.autocast():
                        losses.append(self.criterion(out, target))
                else:
                    losses.append(self.criterion(out, target))
            else:
                loss = None

        return o_list, losses
'''
===  E N D  O F  C L A S S  S T A T I C M O D E L ===
'''


'''
===  S T A R T  O F  C L A S S  M O D E L ===

    [About]

        Class for performing the main dataloading and weight updates. Train functionality happens here.

    [Init Args]

        - net: nn.Module containing the full architecture.
        - criterion : nn.Module that specifies the loss criterion (e.g. CrossEntropyLoss). Could also
        include custom losses.
        - model_prefix : String for the prefix to be used when loading a previous state.
        - step_callback: CallbackList for including all Callbacks created.
        - step_callback_freq: Frequency based on which the Callbacks are updates (and values are logged).
        - save_checkpoint_freq: Integer for the frequency based on which the model is to be saved.
        - opt_batch_size: Integer defines the original batch size to be used.


    [Methods]

        - __init__ : Class initialiser
        - step_end_callback: Function for updating the Callbacks list at the end of each iteration step. In the case of validation,
        this function is called at the end of evaluating.
        - epoch_end_callback: Function for updating the Callbacks at the end of each epoch.
        - adjust_learning_rate: Function for adjusting the learning rate based on the iteration/epoch. Primarily used for circles.
        - fit: Main training loop. Performs both training and evaluation.
        - inference: Function for running inference and obtaining statistics.

'''
class model(static_model):

    def __init__(self,
                 net,
                 criterion,
                 model_prefix='',
                 step_callback=None,
                 step_callback_freq=50,
                 save_checkpoint_freq=1,
                 **kwargs):

        # load parameters
        if kwargs:
            logging.warning("Unknown kwargs: {}".format(kwargs))

        super(model, self).__init__(net, criterion=criterion,
                                         model_prefix=model_prefix)

        # load optional arguments
        # - callbacks
        self.callback_kwargs = {'epoch': None,
                                'batch': None,
                                'read_elapse': None,
                                'forward_elapse': None,
                                'backward_elapse': None,
                                'epoch_elapse': None,
                                'namevals': None,
                                'optimiser_dict': None,}

        if not step_callback:
            step_callback = callback.CallbackList(callback.SpeedMonitor(),
                                                  callback.MetricPrinter())

        self.step_callback = step_callback
        self.step_callback_freq = step_callback_freq
        self.save_checkpoint_freq = save_checkpoint_freq


    def step_end_callback(self):
        # logging.debug("Step {} finished!".format(self.i_step))
        self.step_callback(**(self.callback_kwargs))

    def epoch_end_callback(self,results_directory):
        if self.callback_kwargs['epoch_elapse'] is not None:
            # Main logging definition
            logging.info("Epoch [{:d}]   time cost: {:.2f} sec ({:.2f} h)".format(
                    self.callback_kwargs['epoch'],
                    self.callback_kwargs['epoch_elapse'],
                    self.callback_kwargs['epoch_elapse']/3600.))
        if self.callback_kwargs['epoch'] == 0 \
           or ((self.callback_kwargs['epoch']+1) % self.save_checkpoint_freq) == 0:
            self.save_checkpoint(epoch=self.callback_kwargs['epoch']+1,
                                 optimiser_state=self.callback_kwargs['optimiser_dict'],
                                 base_directory=results_directory)


    def adjust_learning_rate(self, lr, optimiser):
        # learning rate adjustment based on provided lr rate
        for param_group in optimiser.param_groups:
            if 'lr_mult' in param_group:
                lr_mult = param_group['lr_mult']
            else:
                lr_mult = 1.0
            param_group['lr'] = lr * lr_mult


    def fit(self,
            train_iter,
            optimiser,
            lr_scheduler,
            long_short_steps_dir,
            no_cycles,
            eval_iter=None,
            batch_shape=(24,16,224,224),
            workers=4,
            metrics=metric.Accuracy(topk=1),
            sampler_metrics_list=[metric.Accuracy(topk=1) for _ in range(4)],
            iter_per_epoch=1000,
            epoch_start=0,
            epoch_end=10000,
            directory=None,
            precision='mixed',
            scaler=None,
            samplers=4,
            **kwargs):

        # Check kwargs used
        if kwargs:
            logging.warning("Unknown kwargs: {}".format(kwargs))

        assert torch.cuda.is_available(), "only support GPU version"
        #torch.autograd.set_detect_anomaly(True)

        pause_sec = 0.
        train_loaders = {}
        active_batches = {}

        n_workers = workers


        cycles = True

        self.step_callback.set_tot_epochs(epoch_end)
        self.step_callback.set_tot_batches(iter_per_epoch)

        fieldnames = ['Epoch', 'Top1_cl', 'Top5_cl','Loss_cl']
        for s in range(samplers):
            fieldnames.append('Top1_samp_'+str(s))
            fieldnames.append('Top5_samp_'+str(s))
            fieldnames.append('Loss_samp_'+str(s))

        # Create files to write results to
        train_file = open(os.path.join(directory,'train_results_s{0:04d}.csv'.format(epoch_start)), mode='a+', newline='')
        train_writer = csv.DictWriter(train_file, fieldnames=fieldnames)
        # if file is empty write header
        if (os.stat(os.path.join(directory,'train_results_s{0:04d}.csv'.format(epoch_start))).st_size == 0):
            train_writer.writeheader()

        val_file = open(os.path.join(directory,'val_results_s{0:04d}.csv'.format(epoch_start)), mode='a+', newline='')
        val_writer = csv.DictWriter(val_file, fieldnames=fieldnames)
        # if file is empty write header
        if (os.stat(os.path.join(directory,'val_results_s{0:04d}.csv'.format(epoch_start))).st_size == 0):
            val_writer.writeheader()
        #val_writer = csv.writer(f_val, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        train_file.close()
        val_file.close()

        # Set best eval for saving model
        best_val_top1 = 0.0

        for i_epoch in range(epoch_start, epoch_end):

            # Create files to write results to
            train_file = open(os.path.join(directory,'train_results_s{0:04d}.csv'.format(epoch_start)), mode='a+', newline='')
            train_writer = csv.DictWriter(train_file, fieldnames=fieldnames)

            val_file = open(os.path.join(directory,'val_results_s{0:04d}.csv'.format(epoch_start)), mode='a+', newline='')
            val_writer = csv.DictWriter(val_file, fieldnames=fieldnames)


            if (no_cycles):
                logging.info('No cycles selected')
                train_loader = iter(torch.utils.data.DataLoader(train_iter,batch_size=batch_shape[0], shuffle=True,num_workers=n_workers, pin_memory=False))
                cycles = False

            self.callback_kwargs['epoch'] = i_epoch
            epoch_start_time = time.time()


            # values for writing average topk,loss per epoch
            train_top1_sum = {'cl':[]}
            train_top5_sum = {'cl':[]}
            train_loss_sum = {'cl':[]}
            val_top1_sum = {'cl':[]}
            val_top5_sum = {'cl':[]}
            val_loss_sum = {'cl':[]}
            for k in range(samplers):
                train_top1_sum['samp_'+str(k)] = []
                train_top5_sum['samp_'+str(k)] = []
                train_loss_sum['samp_'+str(k)] = []
                val_top1_sum['samp_'+str(k)] = []
                val_top5_sum['samp_'+str(k)] = []
                val_loss_sum['samp_'+str(k)] = []

            # Reset all metrics
            metrics.reset()
            for s_m in sampler_metrics_list:
                s_m.reset()
            # change network `mode` to training to ensure weight updates.
            self.net.train()
            # Time variable definitions
            sum_read_elapse = 0.
            sum_forward_elapse = 0
            sum_backward_elapse = 0
            epoch_speeds = [0,0,0]
            batch_start_time = time.time()
            logging.debug("Start epoch {:d}:".format(i_epoch))
            for i_batch in range(iter_per_epoch):

                b = batch_shape[0]
                t = batch_shape[1]
                h = batch_shape[2]
                w = batch_shape[3]

                selection = ''

                loader_id = 0
                # Increment long cycle steps (8*B).
                if i_batch in long_short_steps_dir['long_0']:
                    b = 8 * b
                    t = t//4
                    h = int(h//math.sqrt(2))
                    w = int(w//math.sqrt(2))

                # Increment long cycle steps (4*B).
                elif i_batch in long_short_steps_dir['long_1']:
                    b = 4 * b
                    t = t//2
                    h = int(h//math.sqrt(2))
                    w = int(w//math.sqrt(2))

                # Increment long cycle steps (2*B).
                elif i_batch in long_short_steps_dir['long_2']:
                    b = 2 * b
                    t = t//2

                # Increment short cycle steps (2*b).
                if i_batch in long_short_steps_dir['short_1']:
                    loader_id = 1
                    b = 2 * b
                    h = int(h//math.sqrt(2))
                    w = int(w//math.sqrt(2))

                # Increment short cycle steps (4*b).
                elif i_batch in long_short_steps_dir['short_2']:
                    loader_id = 2
                    b = 4 * b
                    h = h//2
                    w = w//2

                if (h%2 != 0): h+=1
                if (w%2 != 0): w+=1

                batch_s = (t,h,w)
                if cycles:
                    train_iter.size_setter(new_size=(t,h,w))

                    if (batch_s not in active_batches.values()):
                        logging.info('Creating dataloader for batch of size ({},{},{},{})'.format(b,*batch_s))
                        # Ensure rendomisation
                        train_iter.shuffle(i_epoch+i_batch)
                        # create dataloader corresponding to the created dataset.
                        train_loader = iter(torch.utils.data.DataLoader(train_iter,batch_size=b, shuffle=True,num_workers=n_workers, pin_memory=False))
                        if loader_id in train_loaders:
                            del train_loaders[loader_id]
                        train_loaders[loader_id]=train_loader
                        if loader_id in active_batches:
                            del active_batches[loader_id]
                        active_batches[loader_id]=batch_s
                        gc.collect()

                    try:
                        gc.collect()
                        sum_read_elapse = time.time()
                        data,target = next(train_loaders[loader_id])
                        sum_read_elapse = time.time() - sum_read_elapse
                    except Exception as e:
                        logging.warning(e)
                        # Reinitialise if used up
                        logging.warning('Re-creating dataloader for batch of size ({},{},{},{})'.format(b,*batch_s))
                        # Ensure rendomisation
                        train_iter.shuffle(i_epoch+i_batch)
                        train_loader = iter(torch.utils.data.DataLoader(train_iter,batch_size=b, shuffle=True,num_workers=n_workers, pin_memory=False))

                        if loader_id in train_loaders:
                            del train_loaders[loader_id]
                        train_loaders[loader_id]=iter(train_loader)
                        if loader_id in active_batches:
                            del active_batches[loader_id]
                        active_batches[loader_id]=batch_s
                        gc.collect()
                        sum_read_elapse = time.time()
                        data,target = next(train_loaders[loader_id])
                        sum_read_elapse = time.time() - sum_read_elapse

                    gc.collect()
                else:
                    sum_read_elapse = time.time()
                    data,target = next(train_loader)
                    sum_read_elapse = time.time() - sum_read_elapse

                self.callback_kwargs['batch'] = i_batch

                # Catch Segmentation fault errors and nan grads
                while True:
                    forward = False
                    backward = False
                    try:
                        # [forward] making next step
                        torch.cuda.empty_cache()
                        sum_forward_elapse = time.time()
                        outputs, losses = self.forward(data, target, precision=precision)
                        sum_forward_elapse = time.time() - sum_forward_elapse
                        forward = True

                        # [backward]
                        optimiser.zero_grad()
                        sum_backward_elapse = time.time()
                        for loss in losses[:1]:
                            if precision=='mixed':
                                scaler.scale(loss).backward()
                            else:
                                loss.backward()
                        sum_backward_elapse = time.time() - sum_backward_elapse
                        lr = lr_scheduler.update()
                        batch_size = tuple(data.size())
                        self.adjust_learning_rate(optimiser=optimiser,lr=lr)
                        if precision=='mixed':
                            scaler.step(optimiser)
                            scaler.update()
                        else:
                            optimiser.step()
                        backward = True
                        break

                    except Exception as e:
                        # Create new data loader in the (rare) case of segmentation fault
                        # Or nan loss
                        logging.info(e)
                        logging.warning('Error in forward/backward: forward executed: {} , backward executed: {}'.format(forward,backward))
                        logging.warning('Creating dataloader for batch of size ({},{},{},{})'.format(b,*batch_s))
                        train_iter.shuffle(i_epoch+i_batch+int(time.time()))

                        if loader_id in train_loaders:
                            del train_loaders[loader_id]
                        if loader_id in active_batches:
                            del active_batches[loader_id]
                        if loss in locals():
                            del loss

                        gc.collect()
                        optimiser.zero_grad()
                        torch.cuda.empty_cache()
                        train_loader = iter(torch.utils.data.DataLoader(train_iter,batch_size=b, shuffle=True,num_workers=n_workers, pin_memory=False))
                        train_loaders[loader_id]=train_loader
                        active_batches[loader_id]=batch_s
                        sum_read_elapse = time.time()
                        data,target = next(train_loaders[loader_id])
                        sum_read_elapse = time.time() - sum_read_elapse
                        gc.collect()

                # [evaluation] update train metric
                metrics.update([outputs[0].data.cpu().float()],
                               target.cpu(),
                               [losses[0].data.cpu()],
                               lr,
                               batch_size)

                # update train metrics for sampler
                for s,s_m in enumerate(sampler_metrics_list):
                    s_m.update([outputs[s+1].data.cpu().float()],
                               target.cpu(),
                               [losses[s+1].data.cpu()],
                               lr,
                               batch_size)

                    # Append matrices
                    sm = s_m.get_name_value()
                    train_top1_sum['samp_'+str(s)].append(sm[1][0][2])
                    train_top5_sum['samp_'+str(s)].append(sm[2][0][2])
                    train_loss_sum['samp_'+str(s)].append(sm[0][0][2])

                # Append matrices
                m = metrics.get_name_value()
                train_top1_sum['cl'].append(m[1][0][2])
                train_top5_sum['cl'].append(m[2][0][2])
                train_loss_sum['cl'].append(m[0][0][2])


                # timing each batch
                epoch_speeds += [sum_read_elapse,sum_forward_elapse,sum_backward_elapse]

                if (i_batch % self.step_callback_freq) == 0:
                    # retrive eval results and reset metic
                    self.callback_kwargs['namevals'] = metrics.get_name_value()
                    metrics.reset()
                    # speed monitor
                    self.callback_kwargs['read_elapse'] = sum_read_elapse / data.shape[0]
                    self.callback_kwargs['forward_elapse'] = sum_forward_elapse / data.shape[0]
                    self.callback_kwargs['backward_elapse'] = sum_backward_elapse / data.shape[0]
                    sum_read_elapse = 0.
                    sum_forward_elapse = 0.
                    sum_backward_elapse = 0.
                    # callbacks
                    self.step_end_callback()

            # Epoch end
            self.callback_kwargs['epoch_elapse'] = time.time() - epoch_start_time
            self.callback_kwargs['optimiser_dict'] = optimiser.state_dict()
            self.epoch_end_callback(directory)
            train_loaders = {}
            active_batches = {}

            l = len(train_top1_sum['cl'])
            row = {'Epoch':str(i_epoch)}
            for key in train_top1_sum.keys():
                train_top1_sum[key] = sum(train_top1_sum[key])/l
                row['Top1_'+key] = str(train_top1_sum[key])
                train_top5_sum[key] = sum(train_top5_sum[key])/l
                row['Top5_'+key] = str(train_top5_sum[key])
                train_loss_sum[key] = sum(train_loss_sum[key])/l
                row['Loss_'+key] = str(train_loss_sum[key])

            reordered_row = {k: row[k] for k in fieldnames}
            train_writer.writerow(reordered_row)
            logging.info('Epoch [{:d}]  (train)  average top-1 acc: {:.5f}   average top-5 acc: {:.5f}   average loss: {:.5f}'.format(i_epoch,train_top1_sum['cl'],train_top5_sum['cl'],train_loss_sum['cl']))

            # Evaluation happens here
            if (eval_iter is not None):
                logging.info("Start evaluating epoch {:d}:".format(i_epoch))
                metrics.reset()
                self.net.eval()
                sum_read_elapse = time.time()
                sum_forward_elapse = 0.

                for i_batch, (data, target) in enumerate(eval_iter):

                    f_completed = False
                    while not f_completed:
                        try:
                            sum_read_elapse = time.time()
                            self.callback_kwargs['batch'] = i_batch
                            sum_forward_elapse = time.time()

                            # [forward] making next step
                            torch.cuda.empty_cache()
                            outputs, losses = self.forward(data, target, precision=precision)
                            f_completed = True

                        except Exception as e:
                            logging.info(e)
                            logging.warning('Error in forward/backward: forward executed: {} , backward executed: {}'.format(forward,backward))


                    sum_forward_elapse = time.time() - sum_forward_elapse

                    metrics.update([outputs[0].data.cpu().float()],
                                    target.cpu(),
                                   [losses[0].data.cpu()])

                    # update train metrics for sampler
                    for s,s_m in enumerate(sampler_metrics_list):
                        s_m.update([outputs[s+1].data.cpu().float()],
                                   target.cpu(),
                                   [losses[s+1].data.cpu()])

                        # Append matrices
                        sm = s_m.get_name_value()
                        val_top1_sum['samp_'+str(s)].append(sm[1][0][2])
                        val_top5_sum['samp_'+str(s)].append(sm[2][0][2])
                        val_loss_sum['samp_'+str(s)].append(sm[0][0][2])

                    # Append matrices
                    m = metrics.get_name_value()
                    val_top1_sum['cl'].append(m[1][0][2])
                    val_top5_sum['cl'].append(m[2][0][2])
                    val_loss_sum['cl'].append(m[0][0][2])

                    val_top1_avg = sum(val_top1_sum['cl'])/(i_batch+1)
                    val_top5_avg = sum(val_top5_sum['cl'])/(i_batch+1)
                    val_loss_avg = sum(val_loss_sum['cl'])/(i_batch+1)

                    if (i_batch%50 == 0):
                        logging.info('Epoch [{:d}]: Iteration [{:d}]:  (val)  average top-1 acc: {:.5f}   average top-5 acc: {:.5f}   average loss {:.5f}'.format(i_epoch,i_batch,val_top1_avg,val_top5_avg,val_loss_avg))

                # evaluation callbacks
                self.callback_kwargs['read_elapse'] = sum_read_elapse / data.shape[0]
                self.callback_kwargs['forward_elapse'] = sum_forward_elapse / data.shape[0]
                self.callback_kwargs['namevals'] = metrics.get_name_value()
                self.step_end_callback()

                l = len(val_top1_sum['cl'])
                row = {'Epoch':str(i_epoch)}
                for key in val_top1_sum.keys():
                    val_top1_sum[key] = sum(val_top1_sum[key])/l
                    row['Top1_'+key] = str(val_top1_sum[key])
                    val_top5_sum[key] = sum(val_top5_sum[key])/l
                    row['Top5_'+key] = str(val_top5_sum[key])
                    val_loss_sum[key] = sum(val_loss_sum[key])/l
                    row['Loss_'+key] = str(val_loss_sum[key])

                reordered_row = {k: row[k] for k in fieldnames}
                val_writer.writerow(row)
                logging.info('Epoch [{:d}]:  (val)  average top-1 acc: {:.5f}   average top-5 acc: {:.5f}   average loss {:.5f}'.format(i_epoch,val_top1_avg,val_top5_avg,val_loss_avg))

                # Save best model (regardless of `save_frequency`)
                if val_top1_avg > best_val_top1:
                    best_val_top1 = val_top1_avg
                    self.save_checkpoint(epoch=self.callback_kwargs['epoch']+1,
                                         optimiser_state=self.callback_kwargs['optimiser_dict'],
                                         base_directory=directory,
                                         best=True)

                train_file.close()
                val_file.close()


        logging.info("--- Finished ---")

    def inference(self,
                  eval_iter=None,
                  save_directory=None,
                  workers=4,
                  metrics=metric.Accuracy(topk=1),
                  sampler_metrics_list = [metric.Accuracy(topk=1) for _ in range(4)],
                  precision='mixed',
                  samplers=4,
                  labels_dir=None,
                  fp_topk=3,
                  **kwargs):

            logging.info("Running inference")
            metrics.reset()
            self.net.eval()
            sum_read_elapse = time.time()
            sum_forward_elapse = 0.
            accurac_dict = {} # (!!!) For per-class accuracy the batch size should be 1

            # load json class ids to text
            with open(os.path.join(labels_dir,'dictionary.json')) as f:
                class_dict = json.load(f)

            val_top1_sum = {'cl':[]}
            val_top5_sum = {'cl':[]}
            val_loss_sum = {'cl':[]}
            for k in range(samplers):
                val_top1_sum['samp_'+str(k)] = []
                val_top5_sum['samp_'+str(k)] = []
                val_loss_sum['samp_'+str(k)] = []

            for i_batch, (data, target, path) in enumerate(eval_iter):

                label = path[0].split('/')[-2]

                sum_read_elapse = time.time()
                sum_forward_elapse = time.time()

                # [forward] making next step
                torch.cuda.empty_cache()
                outputs, losses = self.forward(data, target, precision=precision)

                sum_forward_elapse = time.time() - sum_forward_elapse

                metrics.update([outputs[0].data.cpu().float()],
                                target.cpu(),
                               [losses[0].data.cpu()])

                # update train metrics for sampler
                for s,s_m in enumerate(sampler_metrics_list):
                    s_m.update([outputs[s+1].data.cpu().float()],
                               target.cpu(),
                               [losses[s+1].data.cpu()])

                    # Append matrices
                    sm = s_m.get_name_value()
                    val_top1_sum['samp_'+str(s)].append(sm[1][0][2])
                    val_top5_sum['samp_'+str(s)].append(sm[2][0][2])
                    val_loss_sum['samp_'+str(s)].append(sm[0][0][2])

                # Append matrices
                m = metrics.get_name_value()
                val_top1_sum['cl'].append(m[1][0][2])
                val_top5_sum['cl'].append(m[2][0][2])
                val_loss_sum['cl'].append(m[0][0][2])

                # Append rates
                if label in accurac_dict:
                    accurac_dict[label]['TP'] += m[1][0][1]
                    accurac_dict[label]['num'] += 1
                else:
                    accurac_dict[label] = {'TP': m[1][0][1] , 'num':1}

                # Append False positives
                if m[1][0][1] < 1:
                    probs, indices = torch.topk(outputs[0].detach(), fp_topk)
                    probs = probs.cpu().numpy()[0]
                    indices = indices.cpu().numpy()[0]
                    fp_labels = []
                    for index in indices:
                        for k,v in class_dict.items():
                            if v == index:
                                fp_labels.append(k)
                                break
                    if 'FP' in accurac_dict[label]:
                        accurac_dict[label]['FP'][path[0].split('/')[-1]] = {str(fp_label) : float(prob) for (fp_label,prob) in zip(fp_labels,probs)}
                    else:
                        accurac_dict[label]['FP'] = {path[0].split('/')[-1] : {str(fp_label) : float(prob) for (fp_label,prob) in zip(fp_labels,probs)}}

                for s,s_m in enumerate(sampler_metrics_list):
                    sm = s_m.get_name_value()
                    sampler_id = 'samp_'+str(s)
                    if sampler_id in accurac_dict[label].keys():
                        accurac_dict[label][sampler_id]['TP'] += sm[1][0][1]
                    else:
                        accurac_dict[label][sampler_id] = {'TP': sm[1][0][1]}

                    # Append False positives
                    if sm[1][0][1] < 1:
                        probs, indices = torch.topk(outputs[s+1].detach(), fp_topk)
                        probs = probs.cpu().numpy()[0]
                        indices = indices.cpu().numpy()[0]
                        fp_labels = []
                        for index in indices:
                            for k,v in class_dict.items():
                                if v == index:
                                    fp_labels.append(k)
                                    break
                        if 'FP' in accurac_dict[label]['samp_'+str(s)].keys():
                            accurac_dict[label]['samp_'+str(s)]['FP'][path[0].split('/')[-1]] = {str(fp_label) : float(prob) for (fp_label,prob) in zip(fp_labels,probs)}
                        else:
                            accurac_dict[label]['samp_'+str(s)]['FP'] = {path[0].split('/')[-1] : {str(fp_label) : float(prob) for (fp_label,prob) in zip(fp_labels,probs)}}

                line = "Video:: {:d}/{:d} videos, `{}` top-1 acc: [{:.3f} | {:.3f}]".format(i_batch,len(eval_iter.dataset),label, m[1][0][1], sum(val_top1_sum['cl'])/(i_batch + 1))
                print(' '*(len(line)+20), end='\r')
                print(line, end='\r')

            logging.info('Inference: average top-1 acc: {:.5f} average top-5 acc: {:.5f} average loss {:.5f}'.format(sum(val_top1_sum['cl'])/(i_batch+1),sum(val_top5_sum['cl'])/(i_batch+1),sum(val_loss_sum['cl'])/(i_batch+1)))
            for s,sm in enumerate(sampler_metrics_list):
                logging.info('Inference: >> Sampler {} average top-1 acc: {:.5f} average top-5 acc: {:.5f} average loss {:.5f}'.format(s, sum(val_top1_sum['samp_'+str(s)])/(i_batch+1),sum(val_top5_sum['samp_'+str(s)])/(i_batch+1),sum(val_loss_sum['samp_'+str(s)])/(i_batch+1)))

            avg_class_accuracy = 0
            avg_class_accuracy_samplers = [0 for _ in range(samplers)]
            for label in accurac_dict.keys():
                accurac_dict[label]['acc'] = accurac_dict[label]['TP'] / accurac_dict[label]['num']
                logging.info('Inference: Label: `{}` average accuracy: {:.5f} num:{}'.format(label,accurac_dict[label]['acc'],accurac_dict[label]['num']))
                avg_class_accuracy += accurac_dict[label]['acc']
                for s in range(samplers):
                    accurac_dict[label]['samp_'+str(s)]['acc'] = accurac_dict[label]['samp_'+str(s)]['TP'] / accurac_dict[label]['num']
                    avg_class_accuracy_samplers[s] += accurac_dict[label]['samp_'+str(s)]['acc']

            logging.info('> Avg class accuracy: {:.5f}'.format(avg_class_accuracy/len(accurac_dict.keys())))
            accurac_dict['acc_top1_class'] = avg_class_accuracy/len(accurac_dict.keys())
            for s in range(samplers):
                logging.info('>> Avg class accuracy for sampler `{}`: {:.5f}'.format(s,avg_class_accuracy_samplers[s]/len(accurac_dict.keys())))
                accurac_dict['acc_top1_class_sampler_`{}`'.format(s)] = avg_class_accuracy_samplers[s]/len(accurac_dict.keys())

            accurac_dict['acc_top1_cl'] = sum(val_top1_sum['cl'])/len(val_top1_sum['cl'])
            accurac_dict['acc_top5_cl'] = sum(val_top5_sum['cl'])/len(val_top5_sum['cl'])
            for s in range(samplers):
                accurac_dict['acc_top1_samp_'+str(s)] = sum(val_top1_sum['samp_'+str(s)])/len(val_top1_sum['samp_'+str(s)])
                accurac_dict['acc_top5_samp_'+str(s)] = sum(val_top5_sum['samp_'+str(s)])/len(val_top5_sum['samp_'+str(s)])


            # Save to dictionary
            if not os.path.isdir(save_directory):
                os.makedirs(save_directory)
            with open(os.path.join(save_directory,'class_accuracies.json'), 'w') as jsonf:
                json.dump(accurac_dict, jsonf,indent=4, sort_keys=True)

            logging.info("--- Finished ---")

'''
===  E N D  O F  C L A S S  M O D E L ===
'''
