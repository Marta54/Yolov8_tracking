from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import numpy as np

# load video to test
model = YOLO("yolov8l-visdrone.pt")

# load video
video_path = "C:\\Users\\marta.sofia.oliveira\\Downloads\\track_japan_crosswalk\\istockphoto-473333595-640_adpp_is.mp4"
#video_path = "C:\\Users\\marta.sofia.oliveira\\Downloads\\track_japan_crosswalk\\pexels-timo-volz-5544073 (360p).mp4"

video = cv2.VideoCapture(video_path)

save_frames = []
# read frames
has_frame = True
n = 0

shape = 384, 640
array = np.zeros(shape)

while has_frame:
  has_frame, frame = video.read()
  n += 1 
  if has_frame and n % 1 == 0:
    # detect and track objects
    #results = model.predict(frame, classes=[0,1], show = True)
    results = model.track(frame, persist=True, classes = [0,1], conf=0)
    #print(results[0])
    #results = model.track(source="https://youtu.be/LNwODJXcvt4", conf=0.3, iou=0.5, show=True)
    frame_ = results[0].plot(labels = False)
    xywhn = results[0].boxes.xyxy
    print(results[0].boxes.id)
    
    points = [[round(x2.item()), round((y1+(y2-y1)//2).item())] for x1,y1,x2,y2 in xywhn]
    x = [round((x1+(x2-x1)//2).item()) for x1,y1,x2,y2 in xywhn]
    y = [round(y2.item()) for x1,y1,x2,y2 in xywhn]

    for i, j in points:
      array[j-1][i-1] = 1

    # visualize
    #https://docs.ultralytics.com/modes/track/#multithreaded-tracking
    #plt.imshow(frame)
    cv2.imshow('frame', frame_)
    #plt.scatter(x, y, s = 3, c = "yellow")

    #plt.show()
    #cv2.imshow('points', points)

    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
