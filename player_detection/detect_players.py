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



for i in range(35):
    frame_number = i * 10
    frame = get_frame(video_path, frame_number, False)
    player_detection_model.conf = 0.3
    results = player_detection_model(frame)[0]

    ellipse_annotator = sv.EllipseAnnotator(color=sv.ColorPalette.from_hex(['#0000FF', '#FFFF00', '#FFA500']),thickness=2)
    triangle_annotator = sv.TriangleAnnotator(color=sv.Color.from_hex('#FF0000'),base=25,height=21,outline_thickness=1)
    label_annotator = sv.LabelAnnotator(color=sv.ColorPalette.from_hex(['#FF0000', '#0000FF', '#FFFF00', '#FFA500' ]),text_color=sv.Color.from_hex('#000000'))
    detections = sv.Detections.from_ultralytics(results)

    ball_detections = detections[detections.class_id == 0]
    ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

    all_detections = detections[detections.class_id != 0]
    all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
    all_detections.class_id -= 1

    labels = [f"{class_name} {confidence:.2f}" for class_name, confidence in zip(detections['class_name'], detections.confidence)]


    annotated_frame = ellipse_annotator.annotate(scene = frame, detections = all_detections)
    annotated_frame = triangle_annotator.annotate(scene = annotated_frame, detections = ball_detections)
    annotated_frame = label_annotator.annotate(scene = annotated_frame, detections = detections, labels = labels)
    
    sv.plot_image(annotated_frame)