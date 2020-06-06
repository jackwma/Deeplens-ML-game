#*****************************************************
#                                                    *
# Copyright 2018 Amazon.com, Inc. or its affiliates. *
# All Rights Reserved.                               *
#                                                    *
#*****************************************************
import os
import json
import time
import numpy as np
import awscam
import cv2
import mo
import greengrasssdk
import math
from utils import LocalDisplay
from PIL import Image
from PIL import ImageEnhance


def lambda_handler(event, context):
    """Empty entry point to the Lambda function invoked from the edge."""
    return

# Create an IoT client for sending to messages to the cloud.
client = greengrasssdk.client('iot-data')
iot_topic = '$aws/things/{}/infer'.format(os.environ['AWS_IOT_THING_NAME'])

INPUT_WIDTH = 128
INPUT_HEIGHT = 128

def infinite_infer_run():
    """ Run the DeepLens inference loop frame by frame"""
    
    try:
        # Number of top classes to output
        num_top_k = 2
        
        model_type = 'classification'
        model_name = 'image-classification'
        
        with open('labels.txt', 'r') as f:
	        output_map = [l for l in f]

        # Create a local display instance that will dump the image bytes to a FIFO
        # file that the image can be rendered locally.
        local_display = LocalDisplay('480p')
        local_display.start()

        # Optimize the model
        error, model_path = mo.optimize(model_name,INPUT_WIDTH,INPUT_HEIGHT)
        
        # Load the model onto the GPU.
        client.publish(topic=iot_topic, payload='Loading model')
        model = awscam.Model(model_path, {'GPU': 1})
        client.publish(topic=iot_topic, payload='Model loaded')
        
        while True:
            # Get a frame from the video stream
            ret, frame = awscam.getLastFrame()
            if not ret:
                raise Exception('Failed to get frame from the stream')
                
            # Fixing eyefish calibration
            undistorted_img = undistort(frame)
            
            # Isolate the drawing (get frames)
            isolated_frame = isolate_image(undistorted_img)
            
            # # Check if cropping a frame is successful
            # if (isolated_frame == None)
            #     print("Cannot crop the frame")
            #     break
            
            # Resize frame to the same size as the training set
            frame_resize = cv2.resize(isolated_frame, (INPUT_HEIGHT, INPUT_WIDTH))
            
            # Pre processing the input image
            # Convert the image to grap scale and invert the color
            gray = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2GRAY)
            gray = np.array(gray, float)
            gray /= 255
            normImage = 1 - gray
            
            # crop horizontal frames
            normImage[0:20] = 0;
            normImage[108:] = 0;

            # crop vertical frames
            normImage[:, 0:20] = 0;
            normImage[:, 108:] = 0;
            
            normImage = crop_image(normImage, 0.4)
            top, bottom, left, right = [5] * 4
            img_with_border = cv2.copyMakeBorder(normImage, top, bottom, left, right, 
            cv2.BORDER_CONSTANT, value=[0, 0])
            border_image = cv2.resize(img_with_border, (INPUT_HEIGHT, INPUT_WIDTH))
            kernel = np.ones((3, 3), np.uint8)
            dilated_image = cv2.dilate(border_image, kernel, iterations=2)
            
            stacked_img = 255*np.stack((dilated_image,)*3, axis=-1)
            finalOutput = stacked_img.astype(np.uint8)
            # Run the images through the inference engine and parse the results using
            # the parser API, note it is possible to get the output of doInference
            # and do the parsing manually, but since it is a classification model,
            # a simple API is provided.
            parsed_inference_results = model.parseResult(model_type,
                                                         model.doInference(finalOutput))
            # Get top k results with highest probabilities
            top_k = parsed_inference_results[model_type][0:num_top_k]
            # Add the label of the top result to the frame used by local display.
            # See https://docs.opencv.org/3.4.1/d6/d6e/group__imgproc__draw.html
            # for more information about the cv2.putText method.
            # Method signature: image, text, origin, font face, font scale, color, and thickness
            output_text = '{} : {:.2f}'.format(output_map[top_k[0]['label']], top_k[0]['prob'])
            cv2.putText(frame, output_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 165, 20), 8)
            # Set the next frame in the local display stream.
            local_display.set_frame_data(frame)
            # Send the top k results to the IoT console via MQTT
            cloud_output = {}
            for obj in top_k:
                cloud_output[output_map[obj['label']]] = obj['prob']
            client.publish(topic=iot_topic, payload=json.dumps(cloud_output))
    except Exception as ex:
      	print('Error in lambda {}'.format(ex))
        client.publish(topic=iot_topic, payload='Error in lambda: {}'.format(ex))
            
##### PREPROCESSING #####
class Config:
    def __init__(self):
        pass
    
    min_area = 100000
    min_contours = 8
    threshold_thresh = 50
    epsilon_start = 50
    epsilon_step = 10
    
def undistort(frame): 
    frame_height, frame_width, _ = frame.shape
    K=np.array([[511.98828907136766, 0.0, 426.48016197546474], [0.0, 513.8644747557715, 236.89875770956868], [0.0, 0.0, 1.0]])
    D=np.array([[-0.10969105781526832], [0.03463562293251206], [-0.2341226037892333], [0.34335682066685935]])
    DIM = (int(frame_width/3), int(frame_height/3))
    frame_resize = cv2.resize(frame, DIM)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(frame_resize, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img
    
def isolate_image(frame):
    frame = Image.fromarray(frame)
    enh_color = ImageEnhance.Color(frame)
    colorBalance = 0.1000
    img_colored = enh_color.enhance(colorBalance)
    # enhance the contrast
    enh_con = ImageEnhance.Contrast(img_colored)
    contrast = 5.01
    img_contrasted = enh_con.enhance(contrast)
    image = img_contrasted
    image = np.array(image)
    
    # these constants are carefully picked
    MORPH = 9
    CANNY = 84
    HOUGH = 25
    
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.GaussianBlur(img, (3,3), 0, img)
    
    # this is to recognize white on white
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(MORPH,MORPH))
    dilated = cv2.dilate(img, kernel)
    
    edges = cv2.Canny(dilated, 0, CANNY, apertureSize=3)
    
    lines = cv2.HoughLinesP(edges, 1,  3.14/180, HOUGH)
    for line in lines[0]:
        cv2.line(edges, (line[0], line[1]), (line[2], line[3]), (255,0,0), 2, 8)
    
    # finding contours
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = filter(lambda cont: cv2.arcLength(cont, False) > 100, contours)
    contours = filter(lambda cont: cv2.contourArea(cont) > 10000, contours)
    
    result = None
    for idx, c in enumerate(contours):
        if len(c) < Config.min_contours:
            continue
    
        epsilon = Config.epsilon_start
        while True:
            approx = cv2.approxPolyDP(c, epsilon, True)
            approx = approx.reshape((len(approx), 2))
            new_approx = []
            for i in range(len(approx)):
                if 80 < approx[i][0] < 750:
                    new_approx.append(approx[i])
            approx = np.array(new_approx)
            
            if (len(approx) < 4):
                break
            if math.fabs(cv2.contourArea(approx)) > Config.min_area:
                if (len(approx) > 4):
                    epsilon += Config.epsilon_step
                    continue
                
                else:
                    # for p in approx:
                    #    cv2.circle(binary,(p[0][0],p[0][1]),8,(255,255,0),thickness=-1)
                    approx = approx.reshape((4, 2))
                    # [top-left, top-right, bottom-right, bottom-left]
                    src_rect = order_points(approx)
    
                    cv2.drawContours(image, c, -1, (0, 255, 255), 1)
                    cv2.line(image, (src_rect[0][0], src_rect[0][1]), (src_rect[1][0], src_rect[1][1]),
                             color=(100, 255, 100))
                    cv2.line(image, (src_rect[2][0], src_rect[2][1]), (src_rect[1][0], src_rect[1][1]),
                             color=(100, 255, 100))
                    cv2.line(image, (src_rect[2][0], src_rect[2][1]), (src_rect[3][0], src_rect[3][1]),
                             color=(100, 255, 100))
                    cv2.line(image, (src_rect[0][0], src_rect[0][1]), (src_rect[3][0], src_rect[3][1]),
                             color=(100, 255, 100))
    
                    # get the min rectangle contour
                    rect = cv2.minAreaRect(approx)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    box = box.reshape(4, 2)
                    box = order_points(box)
                    w, h = point_distance(box[0], box[1]), point_distance(box[1], box[2])
    
                    # perspective transformation
                    dst_rect = np.array([
                        [0, 0],
                        [w - 1, 0],
                        [w - 1, h - 1],
                        [0, h - 1]],
                        dtype="float32")
                    M = cv2.getPerspectiveTransform(src_rect, dst_rect)
                    warped = cv2.warpPerspective(image, M, (w, h))
                    # cv2.imwrite("cropped.png", warped, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
    
                    imgAfterCropping = warped
                    imgAfterCropping = Image.fromarray(imgAfterCropping)
                    enh_con = ImageEnhance.Contrast(imgAfterCropping)
                    contrast = 1.0
                    contrastedAfter = enh_con.enhance(contrast)
                    # contrastedAfter.save("finalCroppedImage.jpg")  # This image is ready for model!
                    result = np.array(contrastedAfter)
                    return result
                    break
            else:
                print("failed %d area=%f" % (idx, math.fabs(cv2.contourArea(approx))))
                break
        
    return result

def get_new(old):
    new = np.ones(old.shape, np.uint8)
    cv2.bitwise_not(new,new)
    return new
        
def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect

def point_distance(a,b):
    return int(np.sqrt(np.sum(np.square(a - b))))

def crop_image(img, tol):
    # img is 2D image data
    # tol  is tolerance
    mask = img>tol
    return img[np.ix_(mask.any(1), mask.any(0))]

infinite_infer_run()
