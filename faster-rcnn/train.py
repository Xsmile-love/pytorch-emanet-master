#-------------------------------------#
#       Training on the dataset
#-------------------------------------#
import os
import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.frcnn import FasterRCNN
from nets.frcnn_training import (FasterRCNNTrainer, get_lr_scheduler,
                                 set_optimizer_lr, weights_init)
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import FRCNNDataset, frcnn_dataset_collate
from utils.utils import get_classes, show_config
from utils.utils_fit import fit_one_epoch

'''
Training your own target detection model must require attention to the following points:
1、Double check if your format meets the requirements before training. The library requires the dataset format to be in VOC format,
   and what needs to be prepared are input images and labels. The input image is a .jpg image, no need to fix the size, it will be automatically
   resized before passing in for training. grayscale image will be automatically converted to RGB image for training, no need to modify it by yourself. 
   If the input image has a non-jpg suffix, you need to convert it to jpg by batch before starting training.The tags are in .xml format, and the file
   will contain information about the target to be detected, and the tag file corresponds to the input image file.
   

2、The size of the loss value is used to determine whether there is convergence. What is more important is that there is a trend of convergence,
   i.e., the loss in the validation set keeps decreasing, and the model basically converges if the loss in the validation set basically does not change. 
   The exact size of the loss value is not meaningful, as large or small only depends on the way the loss is calculated, and not close to 0. If you want 
   to make the loss look better, you can divide it by 10,000 in the corresponding loss function. the loss values during training are saved in the logs folder
   in the loss_%Y_%m_%d_%H_%M_%S folder

   
3、The trained weights file is saved in the logs folder. Each training generation (Epoch) contains several training steps (Step), and each training step 
   (Step) performs a gradient descent. If only a few Steps are trained, they are not saved. The concepts of Epoch and Step should be clarified.
'''
if __name__ == "__main__":
    #-------------------------------#
    #   Whether to use Cuda, no GPU can be set to False
    #-------------------------------#
    Cuda            = True
    #---------------------------------------------------------------------#
    #   train_gpu:  The GPUs used for training are the first card by default, [0, 1] for dual cards, and [0, 1, 2] for triple cards. When using multiple
    #               GPUs, the batch on each card is the total batch divided by the number of cards.   
    #---------------------------------------------------------------------#
    train_gpu       = [0,]
    #---------------------------------------------------------------------#
    #   fp16:        Whether to use mixed precision training, it can reduce the video memory by about half, and requires pytorch1.7.1 or above
    #---------------------------------------------------------------------#
    fp16            = False
    #---------------------------------------------------------------------#
    #   classes_path:    Point to the txt under model_data, which corresponds to the dataset you trained,  Be sure to modify the classes_path
    #                    before training so that it corresponds to your own dataset
    #---------------------------------------------------------------------#
    classes_path    = 'model_data/voc_classes.txt'
    #----------------------------------------------------------------------------------------------------------------------------#
    #   Please see README for the download of the weights file, which can be downloaded via Netflix. The pre-training weights of the model are common to different datasets, because the features are common.
    #   The more important part of the pre-training weights of the model is the weight part of the backbone feature extraction network, which is used to perform feature extraction.
    #   Pre-training weights must be used for 99% of cases, if not, the weights of the main part are too random, the feature extraction effect is not obvious, and the network training results will not be good
    #
    #   If there is an operation to interrupt the training process, you can set the model_path to the weights file in the logs folder to load the weights that have been partially trained again.
    #   Also modify the parameters of the freeze phase or thaw phase below to ensure the continuity of the model epoch.
    #   
    #   Do not load the weights of the whole model when model_path = ''.
    #
    #   The weights used here are the weights of the whole model, so they are loaded in train.py. The following pretrain does not affect the loading of the weights here.
    #   If you want the model to start training from the pre-training weights of the backbone, set model_path = '' with pretrain = True below, when only the backbone is loaded.
    #   If you want the model to be trained from 0, set model_path = '', pretrain = Fasle below, and Freeze_Train = Fasle, at which point the training starts from 0 and there is no process of freezing the backbone.
    #   
    #   Generally speaking, the network will be poorly trained from 0 because the weights are too random and feature extraction is not effective, so it is highly, highly, highly discouraged to start training from 0!
    #   If you must start from 0, you can understand the imagenet dataset and first train the classification model to obtain the backbone part weights of the network, the backbone part of the classification model and that model in general, based on which training is performed.
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path      = 'model_data/voc_weights_resnet.pth'
    #------------------------------------------------------#
    #   input_shape:     Input shape size
    #------------------------------------------------------#
    input_shape     = [600, 600]
    #---------------------------------------------#
    #   vgg
    #   resnet50
    #---------------------------------------------#
    backbone        = "resnet50"
    #----------------------------------------------------------------------------------------------------------------------------#
    #   pretrained:     Whether to use the pre-training weights of the backbone network, here the weights of the backbone are used and are therefore loaded at the time of model construction.
    #                   If model_path is set, the weights of the trunk need not be loaded and the value of pretrained is meaningless.
    #                   If no model_path is set and pretrained = True, only the backbone is loaded to start training.
    #                   If model_path is not set, pretrained = False and Freeze_Train = Fasle, the training starts from 0 and there is no process of freezing the backbone.
    #----------------------------------------------------------------------------------------------------------------------------#
    pretrained      = False
    #------------------------------------------------------------------------#
    #   anchors_size is used to set the size of the a priori box. 9 a priori boxes exist for each feature point.
    #   Each number of anchors_size corresponds to 3 a priori boxes.
    #   When anchors_size = [8, 16, 32], the generated a priori box width and height are approximately
    #   [90, 180] ; [180, 360]; [360, 720]; [128, 128]; 
    #   [256, 256]; [512, 512]; [180, 90] ; [360, 180]; 
    #   [720, 360]; See anchors.py for details
    #   If you want to detect small objects, you can reduce the number of anchors_size in front of them.
    #   For example, set anchors_size = [4, 16, 32]
    #------------------------------------------------------------------------#
    anchors_size    = [8, 16, 32]

    #----------------------------------------------------------------------------------------------------------------------------#
    #   The training is divided into two phases, the freezing phase and the unfreezing phase. The freeze phase is set to meet the training needs of students with insufficient machine performance.
    #   Freeze training requires less video memory, and in the case of very poor graphics cards, Freeze_Epoch can be set equal to UnFreeze_Epoch, when only freeze training is performed.
    #      
    #   Here are some suggestions for parameter settings that trainers can adjust flexibly to suit their needs:
    #   （一）Training starts with pre-training weights for the entire model:
    #       Adam：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True，optimizer_type = 'adam'，Init_lr = 1e-4. (Frozen)
    #           Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False，optimizer_type = 'adam'，Init_lr = 1e-4. (No freezing)
    #       SGD：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 150，Freeze_Train = True，optimizer_type = 'sgd'，Init_lr = 1e-2. (Frozen)
    #           Init_Epoch = 0，UnFreeze_Epoch = 150，Freeze_Train = False，optimizer_type = 'sgd'，Init_lr = 1e-2。（不冻结）
    #       Where: UnFreeze_Epoch can be adjusted between 100 and 300.
    #   （二）Training starts from the pre-training weights of the backbone network:
    #       Adam：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True，optimizer_type = 'adam'，Init_lr = 1e-4. (Frozen)
    #           Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False，optimizer_type = 'adam'，Init_lr = 1e-4. (No freezing)
    #       SGD：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 150，Freeze_Train = True，optimizer_type = 'sgd'，Init_lr = 1e-2. (No freezing)
    #           Init_Epoch = 0，UnFreeze_Epoch = 150，Freeze_Train = False，optimizer_type = 'sgd'，Init_lr = 1e-2. (No freezing)
    #       Among them: Since the training starts from the pre-training weights of the backbone network, the weights of the backbone are not necessarily suitable for the target detection, and more training is needed to jump out the local optimal solution.
    #             UnFreeze_Epoch can be adjusted between 150-300, and 300 is recommended for both YOLOV5 and YOLOX.
    #             Adam converges a bit faster compared to SGD. Therefore UnFreeze_Epoch can be smaller in theory, but still more Epoch is recommended.
    #   （三）Setting of batch_size.
    #       Within the acceptable range of the graphics card, the larger is better. Insufficient memory has nothing to do with dataset size, please reduce the batch_size when it indicates insufficient memory (OOM or CUDA out of memory).
    #       faster rcnn's Batch BatchNormalization layer has been frozen and the batch_size can be 1
    #----------------------------------------------------------------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Freeze phase training parameters.
    #   At this point the backbone of the model is frozen and the feature extraction network does not change.
    #   Smaller memory footprint, fine-tuned for network only.
    #   Init_Epoch:         The model currently starts with a training generation whose value can be greater than Freeze_Epoch, e.g. setting: Init_Epoch = 60, Freeze_Epoch = 50, UnFreeze_Epoch = 100, will skip the freeze phase and start directly from
    #                       generation 60 and adjust the corresponding learning rate.
    #                       (for use during breakpoint practice)
    #   Freeze_Epoch        Freeze_Epoch for model freeze training
    #                       (disabled when Freeze_Train=False)
    #   Freeze_batch_size   Model freeze training batch_size
    #                       (disabled when Freeze_Train=False)
    #------------------------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 4
    #------------------------------------------------------------------#
    #   Thawing phase training parameters.
    #   At this time, the backbone of the model is not frozen, the feature extraction network will be changed, occupying a large amount of video memory, and all parameters of the network will be changed.
    #   UnFreeze_Epoch:          The total training epoch of the model, SGD takes longer to converge, so set a larger UnFreeze_Epoch, Adam can use a relatively small UnFreeze_Epoch
    #   Unfreeze_batch_size:     Batch_size of the model after thawing
    #------------------------------------------------------------------#
    UnFreeze_Epoch      = 100
    Unfreeze_batch_size = 2
    #------------------------------------------------------------------#
    #   Freeze_Train:   Whether to freeze the training, default freeze the backbone training first and then unfreeze the training. If set Freeze_Train=False, it is recommended to use the optimizer as sgd. 
    #------------------------------------------------------------------#
    Freeze_Train        = True
    
    #------------------------------------------------------------------#
    #   Other training parameters: learning rate, optimizer, learning rate decrease related
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Init_lr:         Maximum learning rate of the model, It is recommended to set Init_lr=1e-4 when using Adam optimizer, and Init_lr=1e-2 when using SGD optimizer.
    #   Min_lr:          Minimum learning rate of the model, default is 0.01 of the maximum learning rate
    #------------------------------------------------------------------#
    Init_lr             = 1e-4
    Min_lr              = Init_lr * 0.01
    #------------------------------------------------------------------#
    #   optimizer_type:  The types of optimizers used are adam, sgd, and Init_lr=1e-4 when using the Adam optimizer, and Init_lr=1e-2 when using the SGD optimizer.
    #   momentum:        Optimizer internal use of the momentum parameter
    #   weight_decay:    The weight decay can prevent overfitting, adam will cause weight_decay error, it is recommended to set to 0 when using adam.
    #------------------------------------------------------------------#
    optimizer_type      = "adam"
    momentum            = 0.9
    weight_decay        = 0
    #------------------------------------------------------------------#
    #   lr_decay_type:   The learning rate decline method used, optionally 'step', 'cos'
    #------------------------------------------------------------------#
    lr_decay_type       = 'cos'
    #------------------------------------------------------------------#
    #   save_period:     How many epochs to save the weights once
    #------------------------------------------------------------------#
    save_period         = 5
    #------------------------------------------------------------------#
    #   save_dir:        The folder where the rights and log files are saved
    #------------------------------------------------------------------#
    save_dir            = 'logs'
    #------------------------------------------------------------------#
    #   eval_flag:       Whether to evaluate at training time, evaluate on the validation set, install the pycocotools library for a better evaluation experience.
    #   eval_period:     It is not recommended to evaluate frequently, as it takes more time to evaluate, and frequent evaluation will lead to very slow training.
    #   The mAP obtained here will be different from that obtained by get_map.py for two reasons:
    #   （一）The mAP obtained here is the mAP of the validation set.
    #   （二）The evaluation parameters are set here conservatively, with the aim of speeding up the evaluation.
    #------------------------------------------------------------------#
    eval_flag           = True
    eval_period         = 5
    #------------------------------------------------------------------#
    #   num_workers:     Used to set whether to use multi-threading to read data, 1 means turn off multi-threading, after opening will speed up the data reading
    #                    speed, but will occupy more memory, and then open multi-threading when the IO is the bottleneck, that is, the GPU computing speed is much 
    #                    faster than the speed of reading images.
    #------------------------------------------------------------------#
    num_workers         = 4
    #----------------------------------------------------#
    #   Get image paths and tags
    #----------------------------------------------------#
    train_annotation_path   = '2007_train.txt'
    val_annotation_path     = '2007_val.txt'
    
    #----------------------------------------------------#
    #   Get classes and anchors
    #----------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)

    #------------------------------------------------------#
    #   Set up the graphics card used
    #------------------------------------------------------#
    os.environ["CUDA_VISIBLE_DEVICES"]  = ','.join(str(x) for x in train_gpu)
    ngpus_per_node                      = len(train_gpu)
    print('Number of devices: {}'.format(ngpus_per_node))
    
    model = FasterRCNN(num_classes, anchor_scales = anchors_size, backbone = backbone, pretrained = pretrained)
    if not pretrained:
        weights_init(model)
    if model_path != '':
        print('Load weights {}.'.format(model_path))
        
        #------------------------------------------------------#
        #   Load according to the Key of pre-trained weights and the Key of the model.
        #------------------------------------------------------#
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        #------------------------------------------------------#
        #   Show keys that do not match.
        #------------------------------------------------------#
        print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
        print("\n\033[1;33;44mWarm tip, the head part is not loaded is normal, Backbone part is not loaded is wrong.\033[0m")

    #----------------------#
    #   Record Loss
    #----------------------#
    time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
    log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
    loss_history    = LossHistory(log_dir, model, input_shape=input_shape)

    #------------------------------------------------------------------#
    #   torch 1.2 does not support amp, it is recommended to use torch 1.7.1 and above to use fp16 correctly, so torch 1.2 shows "could not be resolve" here.
    #------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model_train)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    #---------------------------#
    #   Read the txt corresponding to the dataset
    #---------------------------#
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)
    
    show_config(
        classes_path = classes_path, model_path = model_path, input_shape = input_shape, \
        Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
        Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
        save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
    )
    #---------------------------------------------------------#
    #   Total training generations refers to the total number of iterations over the entire data.
    #   The total training step refers to the total number of gradient descents.
    #   Each training generation contains several training steps, and each training step performs one gradient descent.
    #   Only the minimum training generation is recommended here, and the top is not capped, and only the thawing part is considered in the calculation.
    #----------------------------------------------------------#
    wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
    total_step  = num_train // Unfreeze_batch_size * UnFreeze_Epoch
    if total_step <= wanted_step:
        if num_train // Unfreeze_batch_size == 0:
            raise ValueError('The dataset is too small for training, please expand the dataset.')
        wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
        print("\n\033[1;33;44m[Warning] When using the %s optimizer, it is recommended to set the total training step size to %d or higher.\033[0m"%(optimizer_type, wanted_step))
        print("\033[1;33;44m[Warning] The total training data volume for this run is %d, Unfreeze_batch_size is %d, a total of %d Epochs are trained, and the total training step size is calculated as %d.\033[0m"%(num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
        print("\033[1;33;44m[Warning] Since the total training step is %d, which is less than the recommended total step %d, it is recommended to set the total generation to %d.\033[0m"%(total_step, wanted_step, wanted_epoch))

    #------------------------------------------------------#
    #   Trunk feature extraction network features generic, freeze training can speed up the training speed, also can prevent the weights from being destroyed in the early stage of training.
    #   Init_Epoch is the starting generation.
    #   Freeze_Epoch for freeze training generations.
    #   UnFreeze_Epoch Total Training Generation.
    #   Prompt OOM or insufficient video memory, please reduce the Batch_size.
    #------------------------------------------------------#
    if True:
        UnFreeze_flag = False
        #------------------------------------#
        #   Freeze a certain part of the training
        #------------------------------------#
        if Freeze_Train:
            for param in model.extractor.parameters():
                param.requires_grad = False
        # ------------------------------------#
        #   Freeze bn layer
        # ------------------------------------#
        model.freeze_bn()

        #-------------------------------------------------------------------#
        #   If you don't freeze the training, set the batch_size to Unfreeze_batch_size directly
        #-------------------------------------------------------------------#
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        #-------------------------------------------------------------------#
        #   Determine the current batch_size and adjust the learning rate adaptively
        #-------------------------------------------------------------------#
        nbs             = 16
        lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 5e-2
        lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
        
        #---------------------------------------#
        #   Optimizer selection based on optimizer_type
        #---------------------------------------#
        optimizer = {
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
        }[optimizer_type]

        #---------------------------------------#
        #   The formula for obtaining a decrease in learning rate
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
        #---------------------------------------#
        #   Determine the length of each generation
        #---------------------------------------#
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("The dataset is too small to continue the training, please expand the dataset.")

        train_dataset   = FRCNNDataset(train_lines, input_shape, train = True)
        val_dataset     = FRCNNDataset(val_lines, input_shape, train = False)

        gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=frcnn_dataset_collate)
        gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=frcnn_dataset_collate)

        train_util      = FasterRCNNTrainer(model_train, optimizer)
        #----------------------#
        #   Record eval's map curve
        #----------------------#
        eval_callback   = EvalCallback(model_train, input_shape, class_names, num_classes, val_lines, log_dir, Cuda, \
                                        eval_flag=eval_flag, period=eval_period)

        #---------------------------------------#
        #   Start model training
        #---------------------------------------#
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            #---------------------------------------#
            #   If the model has a frozen learning part, then unfreeze it and set the parameters
            #---------------------------------------#
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                #-------------------------------------------------------------------#
                #   Determine the current batch_size and adjust the learning rate adaptively
                #-------------------------------------------------------------------#
                nbs             = 16
                lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 5e-2
                lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                #---------------------------------------#
                #   The formula for obtaining a decrease in learning rate
                #---------------------------------------#
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                
                for param in model.extractor.parameters():
                    param.requires_grad = True
                # ------------------------------------#
                #   Freeze bn layer
                # ------------------------------------#
                model.freeze_bn()

                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("The dataset is too small to continue the training, please expand the dataset.")

                gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last=True, collate_fn=frcnn_dataset_collate)
                gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                            drop_last=True, collate_fn=frcnn_dataset_collate)

                UnFreeze_flag = True
                
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
            
            fit_one_epoch(model, train_util, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, fp16, scaler, save_period, save_dir)
            
        loss_history.writer.close()
