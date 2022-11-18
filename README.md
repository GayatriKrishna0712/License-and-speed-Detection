# ABOUT THE REPOSITORY
This repository focuses on evaluating real-time traffic videos and identifying license plate numbers and matching vehicle speeds.
For the project: 
1. Software used: Spyder
2. Methodology: YOLOv5 and DeepSort(for vehicle tracking)
3. License Image Dataset: kaagle.com
4. UI software: streamlit

# OBJECTIVE
The objective of the project is to create a UI that will display the speed and number plate of a vehicle.
1. Type of vehicle: For a given input video data, classify the vehicle as car,truck,bus etc
2. Number Plate Detection: To draw a rectangle around the number plate of the vehicle and also display the number 
3. Speed Detection: To detect the speed of the vehicle. 
4. Generate an Excel: The excel should  will record all the obervation.

# PLAN OF ATTACK
![image](https://user-images.githubusercontent.com/93417245/201829005-49d3f136-aeb9-4da2-8c0b-c9f99d68c39b.png)

# METHODOLOGY DESCRIPTION
1. Optical character recognition is a technology that recognizes text within a digital image. It is commonly used to recognize text in scanned documents and images. OCR software can be used to convert a physical paper document, or an image into an accessible electronic version with text. A Python library called EasyOCR makes it possible to turn images into text. It has access to over 70 languages, including English, Chinese, Japanese, Korean, Hindi, and many more are being added. It is by far the simplest approach to implement OCR.
```python
!pip install easyocr
import easyocr
```

2. YOLO an acronym for 'You only look once', is an object detection algorithm that divides images into a grid system. Each cell in the grid is responsible for detecting objects within itself. After the release of YOLOv4 Glenn Jocher introduced YOLOv5 using the Pytorch framework. YOLOv5 is one of the most famous object detection algorithms due to its speed and accuracy.
```python
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
import torch
```

3. Performance is frequently improved by pre-training a model on a very large dataset to acquire useful representations, and then fine-tuning it on the relevant task. This is done using Transfer learning

4. The computer vision research community frequently uses the benchmarking dataset Common Objects in Context (COCO). Even general practitioners working in the field can use it. COCO contains over 330,000 images, of which more than 200,000 are labelled, across dozens of categories of objects. The COCO dataset is designed to represent a vast array of things that we regularly encounter in everyday life, from vehicles like bikes to animals like dogs to people.
```python 
label_names = {2: 'car', 5: 'bus', 7: 'truck'}
```


5. Object tracking is an important task in computer vision. Object trackers are an integral part of many computer vision applications that process the video stream of cameras.  SORT stands for Simple Online Real-time Tracking. The simple Sort Algorithm using bounding box prediction, Kalman filter and IoU matching. 
    * Bounding box prediction, The first step is detect where are the images present in the image. This can be accomplished using any CNN architecture, YOLO,R-CNN etc.
    * Kalman filter is a linear approximation. It predicts what is the future location of the dected object. Uses of these future predictions are:
        * Predict whether the object that we were tracking was the same object or not. 
        * Deal with the problem of occlusion

# BUSINESS POTENTIAL 

# RESULT
Tested my model on real-time traffic videos and recognised numbers from detected number plates. Further I also estimated vehicles speed using distance-time formulae and also classified the vehcile type.




https://user-images.githubusercontent.com/93417245/201831958-14034f28-bf02-4688-82f0-811b1fa317b7.mp4



<u>The report generated:</u>

![image](https://user-images.githubusercontent.com/93417245/201830703-2300163d-0b3f-47f9-a988-52528aac944a.png)




























## REFERENCE 
1. https://github.com/ultralytics/yolov5
2. https://learnopencv.com/edge-detection-using-opencv/
3. https://www.geeksforgeeks.org/python-bilateral-filtering/
4. https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
5. https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html                                          
6. https://www.oreilly.com/library/view/mastering-opencv-4/9781789344912/16b55e96-1027-4765-85d8-ced8fa071473.xhtml                                 
7. https://www.geeksforgeeks.org/python-opencv-cv2-puttext-method/                                                  
8. https://pyimagesearch.com/2020/09/14/getting-started-with-easyocr-for-optical-character-recognition/                                                           
9. https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html
10. https://cocodataset.org/#home

