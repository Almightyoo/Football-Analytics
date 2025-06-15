import numpy as np
import cv2
import supervision as sv
import math
import os 
import sys
import matplotlib.pyplot as plt





vis = True
vis1 = True

video_path = 'assets/videos/15sec_input_720p.mp4'

def get_frame(video_path: str = 'assets/videos/15sec_input_720p.mp4', frame_no: int = 10, vis: bool = False) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    _, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if vis:
        plt.imshow(frame_rgb)
        plt.show()
    return frame


# * fps: 25.0 , w : 1280.0, h : 720.0, pos : 0.0
cap = cv2.VideoCapture(video_path)
frame_number = 150

# get frame
frame = get_frame(video_path=video_path, frame_no=frame_number, vis=False)
original_frame = frame.copy()


# convert to hsv
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
if vis: sv.plot_image(hsv)

# mask green color to use field only
lower_green = np.array([30, 30, 30])
upper_green = np.array([90, 255, 255])
mask = cv2.inRange(hsv, lower_green, upper_green)
if vis: sv.plot_image(mask)

# Morphological image processing
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
# mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
if vis: sv.plot_image(mask)


# Get field boundaries
green_rows = np.where(mask.any(axis=1))[0]
if len(green_rows) > 0:
    field_top = green_rows.min()
    field_bottom = green_rows.max()
    field_height = field_bottom - field_top
    min_vertical_length = field_height * 0.9
# if vis:
#     cv2.line(original_frame, (0, field_top), (original_frame.shape[1], field_top), (255, 0, 0), 2)
#     cv2.line(original_frame, (0, field_bottom), (original_frame.shape[1], field_bottom), (255, 0, 0), 2)
#     cv2.imshow('original frame', original_frame)
#     cv2.waitKey(0)

# bgr to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
if vis: sv.plot_image(gray)

# Mask white lines
_, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
if vis: sv.plot_image(white_mask)

# Combine whitemask and greenmask
white_mask = cv2.bitwise_and(white_mask, mask)
if vis: sv.plot_image(white_mask)

# Clean up white lines
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
white_mask = cv2.dilate(white_mask, kernel, iterations=1)
if vis: sv.plot_image(white_mask)

blurred = cv2.GaussianBlur(white_mask, (5, 5), 0)
if vis: sv.plot_image(blurred)

edges = cv2.Canny(blurred, 30, 100)
if vis: sv.plot_image(edges)

edges = cv2.dilate(edges, kernel, iterations=1)
if vis: sv.plot_image(edges)

lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, maxLineGap=20, minLineLength=min_vertical_length*0.7)
print(len(lines))

if lines is not None:
    center_line = []
    maxi = 0
    longest_line = lines[0]
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        


        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angle = abs(angle)
        
        if (45 <= angle <= 135) and (length >= min_vertical_length):
            if length>maxi:
                maxi = length
                longest_line = line
            center_line.append(line)
        else:
            if vis1: cv2.line(original_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for other lines

    x1, y1, x2, y2 = longest_line[0]
    print(x1, y1, x2, y2)
    m=0.56
    n=2-m
    offset = 400
    center_x = int((m * x1 + n * x2)/(m+n))
    center_y = int((m * y1 + n * y2)/(m+n))
    print(center_x, center_y)
    if vis1: cv2.line(original_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red for center line
    if vis1: cv2.circle(original_frame, (center_x + offset, center_y), radius = 3, color=(0, 0, 255), thickness=3)
        
    print(len(center_line))

if vis1: 
    cv2.imshow('Original Frame', original_frame)
    cv2.waitKey(0)
