import numpy as np
import cv2
import matplotlib.pyplot as plt
from plot_field import SoccerPitchConfiguration, draw_pitch
import supervision as sv
from typing import Tuple

x1 = 334 
y1 = 719 
x2 = 345 
y2 = 114

m = 0.56
n = 2 - m
offset = 400
center_x = int((m * x1 + n * x2)/(m + n))
center_y = int((m * y1 + n * y2)/(m + n))

print(center_x, center_y)

image_points = np.array([
    [x2, y2],  # Bottom end of the center line
    [x1, y1],  # Top end of the center line
    [center_x - offset, center_y], # left center of circle
    [center_x + offset, center_y], # right edge of circle
    [center_x, center_y]
], dtype=np.float32)


pitch_points = np.array([
    [6000, 0],     # Bottom end of the center line
    [6000, 7000],  # Top end of the center line
    [5085, 3500],  # center of circle
    [6915, 3500],  # right edge of circle
    [6000, 3500]
], dtype=np.float32)

H, _ = cv2.findHomography(image_points, pitch_points, method=cv2.RANSAC, ransacReprojThreshold=5.0)

print(H)

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


reprojected = cv2.perspectiveTransform(image_points.reshape(-1,1,2), H)
error = np.mean(np.abs(reprojected - pitch_points.reshape(-1,1,2)))
print("Reprojection Error:", error)

# * fps: 25.0 , w : 1280.0, h : 720.0, pos : 0.0
cap = cv2.VideoCapture(video_path)
frame_number = 150

# get frame
frame = get_frame(video_path=video_path, frame_no=frame_number, vis=False)
image = frame.copy()

h, w = image.shape[:2]
corners_img = np.array([
    [[0, 0]],
    [[w, 0]],
    [[w, h]],
    [[0, h]]
], dtype=np.float32)

print(corners_img)

corners_pitch = cv2.perspectiveTransform(corners_img, H)
print(corners_pitch)

pitch_config = SoccerPitchConfiguration()

scale = 0.1
padding = 50

pitch_image = draw_pitch(
    config=pitch_config,
    background_color=sv.Color(34, 139, 34), 
    line_color=sv.Color.WHITE,
    padding=padding,
    line_thickness=4,
    point_radius=8,
    scale=scale
)


scaled_corners = [(int(k[0][0] * scale) + padding, int(k[0][1] * scale) + padding) for k in corners_pitch]
cv2.polylines(
    img=pitch_image,
    pts=[np.array(scaled_corners, dtype=np.int32)],
    isClosed=True,
    color=(0, 0, 255),
    thickness=2
)



from ultralytics import YOLO

import supervision as sv
import cv2
import matplotlib.pyplot as plt
import numpy as np

def get_frame(video_path: str = 'assets/videos/15sec_input_720p.mp4', frame_no: int = 10, vis: bool = False) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    _, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if vis:
        plt.imshow(frame_rgb)
        plt.show()
    return frame


player_detection_model = YOLO('models/finetuned_best.pt')
video_path = 'assets/videos/15sec_input_720p.mp4'
# * fps: 25.0 , w : 1280.0, h : 720.0, pos : 0.0
frame_number = 150
frame = get_frame(video_path, frame_number, vis=False)

player_detection_model.conf = 0.3
results = player_detection_model(frame)[0]
detections = sv.Detections.from_ultralytics(results)

ball_detections = detections[detections.class_id == 0]
ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

all_detections = detections[detections.class_id != 0]
all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
all_detections.class_id -= 1

player_detections = all_detections[all_detections.class_id == 1]
referee_detection = all_detections[all_detections.class_id == 2]
print(len(referee_detection))

def transform_to_pitch(points):
    """Transform image coordinates to pitch coordinates"""
    points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(points, H)
    return [(int(p[0][0]*scale)+padding, int(p[0][1]*scale)+padding) for p in transformed]

print(len(player_detections))
for xyxy in player_detections.xyxy:
    x_center = (xyxy[0] + xyxy[2]) / 2
    y_center = xyxy[3]
    pitch_coords = transform_to_pitch([[x_center, y_center]])
    print(len(pitch_coords))
    cv2.circle(pitch_image, pitch_coords[0], 5, (255, 0, 0), -1)

for xyxy in ball_detections.xyxy:
    x_center = (xyxy[0] + xyxy[2]) / 2
    y_center = (xyxy[1] + xyxy[3]) / 2
    pitch_coords = transform_to_pitch([[x_center, y_center]])
    cv2.circle(pitch_image, pitch_coords[0], 5, (0, 0, 255), -1)


for xyxy in referee_detection.xyxy:
    x_center = (xyxy[0] + xyxy[2]) / 2
    y_center = (xyxy[1] + xyxy[3]) / 2
    pitch_coords = transform_to_pitch([[x_center, y_center]])
    cv2.circle(pitch_image, pitch_coords[0], 5, (255, 255, 0), -1)

cv2.imshow('nani', pitch_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# plt.figure(figsize=(12, 8))
# plt.imshow(cv2.cvtColor(pitch_image, cv2.COLOR_BGR2RGB))
# plt.title("Bird's-Eye View of Players and Ball")
# plt.axis('off')
# plt.show()


# pitch_width = 10500  # cm
# pitch_height = 6800  # cm

# warped = cv2.warpPerspective(image, H, (pitch_width, pitch_height))
# plt.figure(figsize=(12, 8))
# plt.imshow(warped)
# plt.title("Warped Image (Debug Homography)")
# plt.xlabel("Pitch X (cm)")
# plt.ylabel("Pitch Y (cm)")
# plt.grid(True)
# plt.show()