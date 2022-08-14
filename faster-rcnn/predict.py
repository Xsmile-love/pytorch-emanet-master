#----------------------------------------------------#
#   Single image prediction, camera detection and FPS test functions are integrated into one py file, and mode modification is performed by specifying mode.
#----------------------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image

from frcnn import FRCNN

if __name__ == "__main__":
    frcnn = FRCNN()
    #----------------------------------------------------------------------------------------------------------#
    #   mode is used to specify the mode of the test:
    #   'predict'           Indicates a single image prediction, if you want to modify the prediction process, such as saving images, intercepting objects, etc., you can first read the detailed comments below
    #   'video'             Indicates video detection, you can call the camera or video for detection, check the comments below for details.
    #   'fps'               Indicates test fps, the image used is street.jpg inside img, check the comments below for details.
    #   'dir_predict'       Indicates that the folder is traversed for inspection and saved. Default traverses the img folder and saves the img_out folder, see the comments below for details.
    #----------------------------------------------------------------------------------------------------------#
    mode = "predict"
    #-------------------------------------------------------------------------#
    #   crop:                Specifies whether to intercept the target after a single image prediction
    #   count:               Specifies whether to do a count of the targets
    #   crop, count are only valid when mode='predict'
    #-------------------------------------------------------------------------#
    crop            = False
    count           = False
    #----------------------------------------------------------------------------------------------------------#
    #   video_path:          Used to specify the path of the video, when video_path=0 means detect the camera.
    #                        If you want to detect the video, set e.g. video_path = "xxx.mp4", which means read out the xxx.mp4 file in the root directory.
    #   video_save_path:     Indicates the path where the video is saved, when video_save_path="" means not saved.
    #                        If you want to save the video, set e.g. video_save_path = "yyyy.mp4", which means save as yyyy.mp4 file in the root directory.
    #   video_fps:           The fps of the video used for saving.
    #
    #   video_path, video_save_path and video_fps are only valid when mode='video'.
    #   Saving the video requires ctrl+c to exit or run to the last frame to complete the full save step.
    #----------------------------------------------------------------------------------------------------------#
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0
    #----------------------------------------------------------------------------------------------------------#
    #   test_interval:       Used to specify the number of times the image will be tested when measuring fps. Theoretically the larger the test_interval, the more accurate the fps.
    #   fps_image_path:      The fps image used to specify the test
    #   
    #   test_interval and fps_image_path are only valid for mode='fps'
    #----------------------------------------------------------------------------------------------------------#
    test_interval   = 100
    fps_image_path  = "img/street.jpg"
    #-------------------------------------------------------------------------#
    #   dir_origin_path:     Specifies the folder path of the images to be detected
    #   dir_save_path:       Specifies the path to save the detected images
    #   
    #   dir_origin_path and dir_save_path are only valid when mode='dir_predict'
    #-------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path   = "img_out/"

    if mode == "predict":
        '''
        1、The code can't do batch prediction directly, if you want to do batch prediction, you can use os.listdir() to traverse the folder and use Image.open to 
           open the image file for prediction. The specific process can be referred to get_dr_txt.py, in get_dr_txt.py that achieves the traversal also achieves 
           the saving of the target information.
        2、If you want to save the detected image, use r_image.save("img.jpg") to save it, and modify it directly in predict.py. 
        3、If you want to get the coordinates of the prediction box, you can go to the frcnn.detect_image function and read the four values of top, left, bottom, and right in the drawing section.
        4、If you want to use the prediction box to intercept the lower target, you can enter the frcnn.detect_image function and use the top, left, bottom, and right values obtained in the drawing
           section to intercept the original image in the form of a matrix.
        5、If you want to write additional words on the predicted image, such as the number of specific targets detected, you can go to the frcnn.detect_image function
           and make a judgment on predicted_class in the drawing section. For example, judge if predicted_class == 'car': that is, you can determine whether the current
           target is a car, and then just record the number. Use draw.text to write.
        '''
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = frcnn.detect_image(image, crop = crop, count = count)
                r_image.show()

    elif mode == "video":
        capture=cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        fps = 0.0
        while(True):
            t1 = time.time()
            # Read a frame
            ref,frame=capture.read()
            # Format transformation, BGRtoRGB
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # Convert to Image
            frame = Image.fromarray(np.uint8(frame))
            # Perform testing
            frame = np.array(frcnn.detect_image(frame))
            # RGBtoBGR meets the opencv display format
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break
        capture.release()
        out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = frcnn.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":
        import os
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = frcnn.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)

    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
