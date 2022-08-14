import colorsys
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

from nets.frcnn import FasterRCNN
from utils.utils import (cvtColor, get_classes, get_new_img_size, resize_image,
                         preprocess_input, show_config)
from utils.utils_bbox import DecodeBox


#--------------------------------------------#
#   Prediction using your own trained model requires modification of 2 parameters
#   Both model_path and classes_path need to be modified!
#   If there is a shape mismatch, make sure to pay attention to the NUM_CLASSES, model_path and classes_path parameters during training
#--------------------------------------------#
class FRCNN(object):
    _defaults = {
        #--------------------------------------------------------------------------#
        #   Use your own trained model for prediction make sure to modify model_path and classes_path!
        #   model_path points to the weights file under logs folder, classes_path points to the txt under model_data
        #
        #   After training there are multiple weights files in the logs folder, just choose the one with lower loss in the validation set.
        #   A lower loss in the validation set does not mean a higher mAP, it only means that the weight has better generalization performance on the validation set.
        #   If there is a shape mismatch, also pay attention to the modification of the model_path and classes_path parameters during training
        #--------------------------------------------------------------------------#
        "model_path"    : 'model_data/voc_weights_resnet.pth',
        "classes_path"  : 'model_data/voc_classes.txt',
        #---------------------------------------------------------------------#
        #   The network's backbone feature extraction network, resnet50 or vgg
        #---------------------------------------------------------------------#
        "backbone"      : "resnet50",
        #---------------------------------------------------------------------#
        #   Only prediction frames with scores greater than the confidence level will be retained
        #---------------------------------------------------------------------#
        "confidence"    : 0.5,
        #---------------------------------------------------------------------#
        #   The size of nms_iou used for non-extreme suppression
        #---------------------------------------------------------------------#
        "nms_iou"       : 0.3,
        #---------------------------------------------------------------------#
        #   Used to specify the size of the a priori box
        #---------------------------------------------------------------------#
        'anchors_size'  : [8, 16, 32],
        #-------------------------------#
        #   Whether to use Cuda, no GPU can be set to False
        #-------------------------------#
        "cuda"          : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   Initialize faster RCNN
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value 
        #---------------------------------------------------#
        #   Get the number of types and a priori boxes
        #---------------------------------------------------#
        self.class_names, self.num_classes  = get_classes(self.classes_path)

        self.std    = torch.Tensor([0.1, 0.1, 0.2, 0.2]).repeat(self.num_classes + 1)[None]
        if self.cuda:
            self.std    = self.std.cuda()
        self.bbox_util  = DecodeBox(self.std, self.num_classes)

        #---------------------------------------------------#
        #   Set different colors for picture frames
        #---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

        show_config(**self._defaults)

    #---------------------------------------------------#
    #   Load Model
    #---------------------------------------------------#
    def generate(self):
        #-------------------------------#
        #   Loading models and weights
        #-------------------------------#
        self.net    = FasterRCNN(self.num_classes, "predict", anchor_scales = self.anchors_size, backbone = self.backbone)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        
        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()
    
    #---------------------------------------------------#
    #   Test pictures
    #---------------------------------------------------#
    def detect_image(self, image, crop = False, count = False):
        #---------------------------------------------------#
        #   Calculate the height and width of the input image
        #---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------#
        #   Calculate the size of the image after resize, the short side of the image after resize is 600
        #---------------------------------------------------#
        input_shape = get_new_img_size(image_shape[0], image_shape[1])
        #---------------------------------------------------------#
        #   Converting images to RGB images here prevents grayscale maps from reporting errors when predicting. The code only supports RGB image prediction, all other types of images will be converted to RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   Resize the original image, resize to a size of 600 on the short side
        #---------------------------------------------------------#
        image_data  = resize_image(image, [input_shape[1], input_shape[0]])
        #---------------------------------------------------------#
        #   Add on the batch_size dimension
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            
            #-------------------------------------------------------------#
            #   roi_cls_locs  Suggested adjustment parameters for the box
            #   roi_scores    Suggestion box type score
            #   rois          Coordinates of the suggestion box
            #-------------------------------------------------------------#
            roi_cls_locs, roi_scores, rois, _ = self.net(images)
            #-------------------------------------------------------------#
            #   Decode the suggestion frame using the prediction result of the classifier to obtain the prediction frame
            #-------------------------------------------------------------#
            results = self.bbox_util.forward(roi_cls_locs, roi_scores, rois, image_shape, input_shape, 
                                                    nms_iou = self.nms_iou, confidence = self.confidence)
            #---------------------------------------------------------#
            #   If no object is detected, return to the original image
            #---------------------------------------------------------#           
            if len(results[0]) <= 0:
                return image
                
            top_label   = np.array(results[0][:, 5], dtype = 'int32')
            top_conf    = results[0][:, 4]
            top_boxes   = results[0][:, :4]
        
        #---------------------------------------------------------#
        #   Set font and border thickness
        #---------------------------------------------------------#
        font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness   = int(max((image.size[0] + image.size[1]) // np.mean(input_shape), 1))
        #---------------------------------------------------------#
        #   Counting
        #---------------------------------------------------------#
        if count:
            print("top_label:", top_label)
            classes_nums    = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
        #---------------------------------------------------------#
        #   Whether to perform target tailoring
        #---------------------------------------------------------#
        if crop:
            for i, c in list(enumerate(top_label)):
                top, left, bottom, right = top_boxes[i]
                top     = max(0, np.floor(top).astype('int32'))
                left    = max(0, np.floor(left).astype('int32'))
                bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
                right   = min(image.size[0], np.floor(right).astype('int32'))
                
                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)
        #---------------------------------------------------------#
        #   Image plotting
        #---------------------------------------------------------#
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            # print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image

    def get_FPS(self, image, test_interval):
        #---------------------------------------------------#
        #   Calculate the height and width of the input image
        #---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        input_shape = get_new_img_size(image_shape[0], image_shape[1])
        #---------------------------------------------------------#
        #   Here the image is converted to RGB image to prevent the grayscale map from reporting errors in prediction.
        #   The code only supports RGB image prediction, all other types of images will be converted to RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        
        #---------------------------------------------------------#
        #   Resize the original image, resize to a size of 600 on the short side
        #---------------------------------------------------------#
        image_data  = resize_image(image, [input_shape[1], input_shape[0]])
        #---------------------------------------------------------#
        #   Add on the batch_size dimension
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            roi_cls_locs, roi_scores, rois, _ = self.net(images)
            #-------------------------------------------------------------#
            #   Decode the suggestion frame using the prediction result of the classifier to obtain the prediction frame
            #-------------------------------------------------------------#
            results = self.bbox_util.forward(roi_cls_locs, roi_scores, rois, image_shape, input_shape, 
                                                    nms_iou = self.nms_iou, confidence = self.confidence)
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                roi_cls_locs, roi_scores, rois, _ = self.net(images)
                #-------------------------------------------------------------#
                #   Decode the suggestion frame using the prediction result of the classifier to obtain the prediction frame
                #-------------------------------------------------------------#
                results = self.bbox_util.forward(roi_cls_locs, roi_scores, rois, image_shape, input_shape, 
                                                        nms_iou = self.nms_iou, confidence = self.confidence)
                
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    #---------------------------------------------------#
    #  Testing pictures
    #---------------------------------------------------#
    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w")
        #---------------------------------------------------#
        #   Calculate the height and width of the input image
        #---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        input_shape = get_new_img_size(image_shape[0], image_shape[1])
        #---------------------------------------------------------#
        #   Here the image is converted to RGB image to prevent the grayscale map from reporting errors in prediction.
        #   The code only supports RGB image prediction, all other types of images will be converted to RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        
        #---------------------------------------------------------#
        #   Resize the original image, resize to a size of 600 on the short side
        #---------------------------------------------------------#
        image_data  = resize_image(image, [input_shape[1], input_shape[0]])
        #---------------------------------------------------------#
        #   Add on the batch_size dimension
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            roi_cls_locs, roi_scores, rois, _ = self.net(images)
            #-------------------------------------------------------------#
            #   Decode the suggestion frame using the prediction result of the classifier to obtain the prediction frame
            #-------------------------------------------------------------#
            results = self.bbox_util.forward(roi_cls_locs, roi_scores, rois, image_shape, input_shape, 
                                                    nms_iou = self.nms_iou, confidence = self.confidence)
            #--------------------------------------#
            #   If no object is detected, the original image is returned
            #--------------------------------------#
            if len(results[0]) <= 0:
                return 

            top_label   = np.array(results[0][:, 5], dtype = 'int32')
            top_conf    = results[0][:, 4]
            top_boxes   = results[0][:, :4]
        
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 
