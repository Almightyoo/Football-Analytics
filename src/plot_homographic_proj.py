import numpy as np
import cv2
import supervision as sv
import math
import os 
import sys
import matplotlib.pyplot as plt

from ultralytics import YOLO

import supervision as sv
import cv2
import matplotlib.pyplot as plt
import numpy as np

from draw_pitch import SoccerPitchConfiguration, draw_pitch
from get_homograph_H import get_homograph_matrix_H
from get_frame import get_frame

vis = False
vis1 = True

video_path = 'assets/videos/15sec_input_720p.mp4'

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

output_path = 'output/mapped_pitch_video1.mp4'
os.makedirs(os.path.dirname(output_path), exist_ok=True)

pitch_height, pitch_width = pitch_image.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 25.0, (pitch_width, pitch_height))


# * fps: 25.0 , w : 1280.0, h : 720.0, pos : 0.0
cap = cv2.VideoCapture(video_path)

H = None
for i in range(100):
    frame_number = i

    # get frame
    frame = get_frame(video_path=video_path, frame_no=frame_number, vis=False)
    original_frame = frame.copy()

    if frame_number % 5 == 0:
        H_temp = get_homograph_matrix_H(frame, frame_number)
        if H_temp is not None:
            H = H_temp
    
    if H is not None:
        h, w = original_frame.shape[:2]
        corners_img = np.array([
            [[0, 0]],
            [[w, 0]],
            [[w, h]],
            [[0, h]]
        ], dtype=np.float32)
            


        corners_pitch = cv2.perspectiveTransform(corners_img, H)

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


        player_detection_model = YOLO('models/finetuned_best.pt')

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

        def transform_to_pitch(points):
            """Transform image coordinates to pitch coordinates"""
            points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(points, H)
            return [(int(p[0][0]*scale)+padding, int(p[0][1]*scale)+padding) for p in transformed]

        for xyxy in player_detections.xyxy:
            x_center = (xyxy[0] + xyxy[2]) / 2
            y_center = xyxy[3]
            pitch_coords = transform_to_pitch([[x_center, y_center]])
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

        out.write(pitch_image)

out.release()
cap.release()
cv2.destroyAllWindows()