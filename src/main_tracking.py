import supervision as sv
from ultralytics import YOLO
import torchreid
import torch
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
from collections import deque
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import gc
import matplotlib.pyplot as plt
import math
from typing import Tuple



# * Global Variables
score_threshold_old = 0.9
score_threshold_new = 0.8






# * Initialization
# video_path = 'assets/videos/0-10.mp4'
# video_path = 'assets/videos/0-100.mp4'
video_path = 'assets/videos/15sec_input_720p.mp4'
output_path = "output/final_tracking_with_reid1.mp4"
player_detection_model_path = 'models/best.pt'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# * Load yolo model
yolo_model = YOLO(player_detection_model_path).to(device)
yolo_model.conf = 0.3

# * Load Reid model
reid_model = torchreid.models.build_model('osnet_x1_0', num_classes=1000, pretrained=True)
reid_model.eval().to(device)

# * Load Byte Track
tracker = sv.ByteTrack(frame_rate = 25)
tracker.reset()


# 
player_db = {}

# * ReID transform
reid_transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# * reid image transform
def extract_features(crop: np.ndarray) -> np.ndarray:
    img = Image.fromarray(crop)
    img = reid_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        return reid_model(img).cpu().numpy()
    



# * Get player bounding boxes Detection [just class Players for now]    
def get_player_detections(frame: np.ndarray) -> sv.Detections:
    results = yolo_model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[detections.class_id == 2]
    # detections = detections.with_nms(threshold=0.5, class_agnostic=False)

    detections.xyxy = sv.pad_boxes(detections.xyxy, px=10)
    detections = tracker.update_with_detections(detections)
    return detections[detections.class_id == 2]


# * utility functions for calculating distance
def get_center(xyxy: np.ndarray) -> Tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def euclidean_dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return np.linalg.norm(np.array(p1) - np.array(p2))





# * torch reid osnet with the formula 
def torch_reid(current_id: str, crop: np.ndarray, xyxy: np.ndarray, flag: bool) -> Tuple[str, np.ndarray, Tuple[float, float]]:
    """
    Performs re-identification of a player using appearance and spatial information.
    
    Args:
        current_id (str): Current tracker ID of the player
        crop (np.ndarray): Image crop of the player
        xyxy (np.ndarray): Bounding box coordinates of the player
        flag (bool): Whether this is a new player (True) or existing player (False)
        
    Returns:
        Tuple[str, np.ndarray, Tuple[float, float]]: 
            - Best matching player ID
            - Feature vector of the current crop
            - Center position of the player
    """
    
    best_similarity_threshold = score_threshold_new if flag else score_threshold_old
    current_crop_feat = extract_features(crop)
    current_center = get_center(xyxy)

    best_match_id = current_id
    best_score = 0

    for existing_id, data in player_db.items():
        if existing_id == current_id or len(data['features']) == 0:
            continue

        player_crop_feat = data['features'][-1]
        similarity = cosine_similarity(current_crop_feat, player_crop_feat)[0][0]
        if similarity < best_similarity_threshold:
            continue

        appearances = data.get('appearances', 1)
        past_center = data.get('last_position', current_center)

        distance = euclidean_dist(current_center, past_center)
        distance_score = 1 / (1 + distance)

        score = similarity * math.log(1 + appearances) * distance_score

        if score > best_score:
            best_score = score
            best_match_id = existing_id

    return best_match_id, current_crop_feat, current_center


# * visualization
def plot_reid_comparison(query_crop: np.ndarray, gallery_crop: np.ndarray, query_id: str, gallery_id: str, similarity: float) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(query_crop)
    ax1.set_title(f"Query Player (ID: {query_id})")
    ax1.axis('off')

    ax2.imshow(gallery_crop)
    ax2.set_title(f"Gallery Player (ID: {gallery_id})\nSim: {similarity:.2f}")
    ax2.axis('off')

    plt.suptitle(f"ReID Match: {query_id} → {gallery_id}")
    plt.show()


# * init annotators and frame count
frame_counter = 0
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# * Process frame callback function
def process_frame(frame: np.ndarray, _: int) -> np.ndarray:
    """
    This processes a single video frame for player detection, tracking and re-identification.
    
    Args:
        frame (np.ndarray): Input video frame
        _ (int): Frame index (idk why sv frame requires an additional int variable)
        
    Returns:
        np.ndarray: Annotated frame with player bounding boxes and IDs
    """

    global frame_counter
    frame_counter += 1
    print(f"Processing frame {frame_counter}")

    player_detections = get_player_detections(frame)

    for idx, (xyxy, tracker_id) in enumerate(zip(player_detections.xyxy, player_detections.tracker_id)):
        crop = sv.crop_image(frame, xyxy)
        original_id = str(tracker_id)
        current_id = original_id

        is_new = current_id not in player_db
        current_id, features, center = torch_reid(current_id, crop, xyxy, flag=is_new)

        if current_id not in player_db:
            player_db[current_id] = {
                'crops': deque(maxlen=3),
                'features': deque(maxlen=3),
                'appearances': 0,
                'last_position': center
            }
        
        if current_id != original_id:
            player_detections.tracker_id[idx] = int(current_id)
        
        player_db[current_id]['crops'].append(crop)
        player_db[current_id]['features'].append(features)
        player_db[current_id]['appearances'] += 1
        player_db[current_id]['last_position'] = center



    gc.collect()
    labels = [f"#{tracker_id}" for tracker_id in player_detections.tracker_id]
    annotated = box_annotator.annotate(frame.copy(), detections=player_detections)
    return label_annotator.annotate(annotated, detections=player_detections, labels=labels)


# < =============================================== Test ===============================================>


if __name__ == "__main__":
    print("Starting tracking and ReID process...")
    sv.process_video(
        source_path=video_path,
        target_path=output_path,
        callback=process_frame
    )

    print("Processing complete. Output saved to:", output_path)

