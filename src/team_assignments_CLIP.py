from draw_pitch import SoccerPitchConfiguration, draw_pitch
from get_homograph_H import get_homograph_matrix_H
from get_frame import get_frame
import supervision as sv
from ultralytics import YOLO
from sklearn.cluster import DBSCAN, KMeans
import cv2
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel
import torch
import umap


#! not useful

video_path = 'assets/videos/15sec_input_720p.mp4'
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

for i in range(1):
    frame_number = i

    frame = get_frame(video_path=video_path, frame_no=frame_number, vis=False)
    original_frame = frame.copy()

    player_detection_model = YOLO('models/finetuned_best.pt')

    player_detection_model.conf = 0.3
    results = player_detection_model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)

    ball_detections = detections[detections.class_id == 0]
    ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

    all_detections = detections[detections.class_id != 0]
    all_detections = all_detections.with_nms(threshold=0.3, class_agnostic=True)
    all_detections.class_id -= 1

    player_detections = all_detections[all_detections.class_id == 1]
    referee_detection = all_detections[all_detections.class_id == 2]

    players_crops = [sv.crop_image(frame, xyxy) for xyxy in player_detections.xyxy]

    player_embeddings = []

    for crop in players_crops:
        inputs = processor(images=crop, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            features = model.get_image_features(**inputs)
        
        player_embeddings.append(features.cpu().numpy().flatten())

    player_embeddings = np.array(player_embeddings)
    print(player_embeddings.shape)

    # reducer = umap.UMAP(n_components=3, random_state=42)
    # embeddings_3d = reducer.fit_transform(player_embeddings)

    kmeans = KMeans(n_clusters=2, random_state=42)
    labels = kmeans.fit_predict(player_embeddings)

    plt.figure(figsize=(15, 8))
    for i, (crop, cluster) in enumerate(zip(players_crops, labels)):
        plt.subplot(5, 10, i+1)
        plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        plt.title(f"Team {cluster+1}", color='red' if cluster else 'blue')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
