import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm

from utils.utils import get_classes
from utils.utils_map import get_coco_map, get_map
from frcnn import FRCNN

if __name__ == "__main__":
    '''
    Recall and Precision are not an area concept like AP, so the Recall and Precision values of the network are different when the threshold (Confidence) is different.
    By default, the Recall and Precision calculated by this code represent the corresponding Recall and Precision values when the threshold value (Confidence) is 0.5.

    Due to the limitation of mAP calculation principle, the network needs to obtain nearly all prediction boxes when calculating mAP so that it can calculate Recall 
    and Precision values under different threshold conditions, therefore, the number of boxes of txt inside map_out/detection-results/ obtained by this code will 
    generally be more than the direct predict, in order to list all possible prediction boxes.
    '''
    #------------------------------------------------------------------------------------------------------------------#
    #   map_mode is used to specify what is calculated when the file is run
    #   A map_mode of 0 represents the whole map calculation process, including getting the prediction result, getting the real frame, and calculating VOC_map.
    #   A map_mode of 1 means that only predicted results are obtained.
    #   A map_mode of 2 means that only real boxes are obtained.
    #   A map_mode of 3 means that only the VOC_map is calculated.
    #   A map_mode of 4 means that a 0.50:0.95 map is calculated for the current dataset using the COCO toolbox. you need to get the prediction results, get the real box and install pycocotools to do so
    #-------------------------------------------------------------------------------------------------------------------#
    map_mode        = 0
    #--------------------------------------------------------------------------------------#
    #   The classes_path here is used to specify the class of the VOC_map to be measured, which is generally the same as the classes_path used for training and prediction.
    #--------------------------------------------------------------------------------------#
    classes_path    = 'model_data/voc_classes.txt'
    #--------------------------------------------------------------------------------------#
    #   MINOVERLAP is used to specify the mAP0.x that you want to get. mAP0.x What is the meaning of the mAP0.x Please ask your students to Baidu. For example, to calculate mAP0.75, you can set MINOVERLAP = 0.75.
    #
    #   When the overlap between a prediction frame and the real frame is greater than MINOVERLAP, the prediction frame is considered a positive sample, otherwise it is a negative sample.
    #   Therefore, the larger the value of MINOVERLAP, the more accurate the prediction frame has to be in order to be considered a positive sample, and the lower the calculated mAP value is at this time.
    #--------------------------------------------------------------------------------------#
    MINOVERLAP      = 0.5
    #--------------------------------------------------------------------------------------#
    #   Restricted by the principle of mAP calculation, the network needs to obtain nearly all prediction frames when calculating mAP so that it can calculate mAP
    #   Therefore, the value of confidence should be set as small as possible to obtain all possible prediction frames.
    #   
    #   This value is generally not adjusted. Because the calculation of mAP requires obtaining nearly all the prediction frames, the confidence here cannot be changed arbitrarily.
    #   To get Recall and Precision values at different threshold values, please modify score_threhold below.
    #--------------------------------------------------------------------------------------#
    confidence      = 0.02
    #--------------------------------------------------------------------------------------#
    #   The magnitude of the non-extreme suppression value used in the prediction, with larger indicating less stringent non-extreme suppression.
    #   
    #   This value is generally not adjusted.
    #--------------------------------------------------------------------------------------#
    nms_iou         = 0.5
    #---------------------------------------------------------------------------------------------------------------#
    #   Recall and Precision are not an area concept like AP, so the Recall and Precision values of the network are different when the threshold values are different.
    #   
    #   By default, Recall and Precision calculated by this code represent the Recall and Precision values when the threshold value is 0.5 (defined here as score_threhold).
    #   Because the calculation of mAP requires obtaining nearly all the prediction frames, the confidence defined above cannot be changed arbitrarily.
    #   Here a score_threhold is defined to represent the threshold value, which in turn finds the Recall and Precision values corresponding to the threshold value when calculating the mAP.
    #---------------------------------------------------------------------------------------------------------------#
    score_threhold  = 0.5
    #-------------------------------------------------------#
    #   map_vis is used to specify whether to turn on the visualization of VOC_map calculations
    #-------------------------------------------------------#
    map_vis         = False
    #-------------------------------------------------------#
    #   Point to the folder where the VOC dataset is located, the default is to point to the VOC dataset in the root directory
    #-------------------------------------------------------#
    VOCdevkit_path  = 'VOCdevkit'
    #-------------------------------------------------------#
    #   The folder where the results are output, the default is map_out
    #-------------------------------------------------------#
    map_out_path    = 'map_out'

    image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Main/test.txt")).read().strip().split()

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    class_names, _ = get_classes(classes_path)

    if map_mode == 0 or map_mode == 1:
        print("Load model.")
        frcnn = FRCNN(confidence = confidence, nms_iou = nms_iou)
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
            image       = Image.open(image_path)
            if map_vis:
                image.save(os.path.join(map_out_path, "images-optional/" + image_id + ".jpg"))
            frcnn.get_map_txt(image_id, image, class_names, map_out_path)
        print("Get predict result done.")
        
    if map_mode == 0 or map_mode == 2:
        print("Get ground truth result.")
        for image_id in tqdm(image_ids):
            with open(os.path.join(map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
                root = ET.parse(os.path.join(VOCdevkit_path, "VOC2007/Annotations/"+image_id+".xml")).getroot()
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult')!=None:
                        difficult = obj.find('difficult').text
                        if int(difficult)==1:
                            difficult_flag = True
                    obj_name = obj.find('name').text
                    if obj_name not in class_names:
                        continue
                    bndbox  = obj.find('bndbox')
                    left    = bndbox.find('xmin').text
                    top     = bndbox.find('ymin').text
                    right   = bndbox.find('xmax').text
                    bottom  = bndbox.find('ymax').text

                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print("Get ground truth result done.")

    if map_mode == 0 or map_mode == 3:
        print("Get map.")
        get_map(MINOVERLAP, True, score_threhold = score_threhold, path = map_out_path)
        print("Get map done.")

    if map_mode == 4:
        print("Get map.")
        get_coco_map(class_names = class_names, path = map_out_path)
        print("Get map done.")
