import supervision as sv
from get_frame import get_frame
from ultralytics import YOLO
import torchreid
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from collections import deque
import gc

# --- Initialize components ---
video_path = 'assets/videos/15sec_input_720p.mp4'
output_raw_segment = 'assets/videos/0-10.mp4'
output_annotated = 'output/final_tracking_with_reid.mp4'
player_detection_model_path = 'models/finetuned_best.pt'
os.makedirs(os.path.dirname(output_annotated), exist_ok=True)

tracker = sv.ByteTrack()
tracker.reset()
player_db = {}





# torch reid model osnet
reid_model = torchreid.models.build_model('osnet_x1_0', num_classes=1000, pretrained=True)
reid_model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
reid_model = reid_model.to(device)

reid_transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# *  --- Helper Functions ---


def extract_features(crop):
    img = Image.fromarray(crop)
    img = reid_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        return reid_model(img).cpu().numpy()
    

def get_player_detections(frame, model_path = player_detection_model_path):
    model = YOLO(model_path)
    model.conf = 0.3
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[detections.class_id == 2]
    detections.xyxy = sv.pad_boxes(detections.xyxy, px=10)
    detections = tracker.update_with_detections(detections)
    player_detections = detections[detections.class_id == 2]
    return player_detections


def plot_reid_comparison(query_crop, gallery_crop, query_id, gallery_id, similarity):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    ax1.imshow(query_crop)
    ax1.set_title(f"Query Player (ID: {query_id})")
    ax1.axis('off')
    
    ax2.imshow(gallery_crop)
    ax2.set_title(f"Gallery Player (ID: {gallery_id})\nSimilarity: {similarity:.2f}")
    ax2.axis('off')
    
    plt.suptitle(f"ReID Match: {query_id} → {gallery_id}")
    plt.tight_layout()
    plt.show()


def torch_reid(best_similarity: int, player_db, current_id, crop):
    best_match_id = None
    similarity_scores = []
    
    for existing_id, data in player_db.items():
        if existing_id == current_id or len(data['crops']) == 0:
            continue
        gallery_crop = data['crops'][-1]
        similarity = cosine_similarity(extract_features(crop), extract_features(gallery_crop))[0][0]
        similarity_scores.append(similarity)

        if similarity > best_similarity:
            best_similarity = similarity
            best_match_id = existing_id
            best_gallery_crop = gallery_crop                
    
    if best_match_id:
        print(f"Reassigned {current_id} -> {best_match_id} (sim: {best_similarity:.2f})")
        # plot_reid_comparison(crop, best_gallery_crop, current_id, best_match_id, best_similarity)
        current_id = best_match_id

    return current_id


def process_frame(frame: np.ndarray, _: int) -> np.ndarray:
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()


    player_detections = get_player_detections(frame)
    for idx, (xyxy, tracker_id) in enumerate(zip(player_detections.xyxy, player_detections.tracker_id)):
        crop = sv.crop_image(frame, xyxy)
        original_id = str(tracker_id)
        current_id = original_id

        if current_id not in player_db:
            current_id = torch_reid(best_similarity = 0.80, player_db = player_db, current_id = current_id, crop=crop)
            player_db[current_id] = {'crops': deque(maxlen=10)}
        
        if current_id!= original_id:
            player_detections.tracker_id[idx] = int(current_id)
            
        player_db[current_id]['crops'].append(crop)

    del crop
    gc.collect()

    labels = [f"#{tracker_id}" for tracker_id in player_detections.tracker_id]
    annotated_frame = box_annotator.annotate(frame.copy(), detections=player_detections)
    return label_annotator.annotate(annotated_frame, detections=player_detections, labels=labels)


# for frame_number in range(100, 102):
#     frame = get_frame(video_path, frame_number, vis = False)

#     #acquire player detections
#     player_detections = get_player_detections(frame, player_detection_model_path)
    
#     #create players_db
#     for idx, (xyxy, tracker_id) in enumerate(zip(player_detections.xyxy, player_detections.tracker_id)):
#         crop = sv.crop_image(frame, xyxy)
#         original_id = str(tracker_id)
#         current_id = original_id

#         if current_id not in player_db:
#             current_id = torch_reid(best_similarity = 0.80, player_db = player_db, current_id = current_id)
#             player_db[current_id] = {'crops': deque(maxlen=10)}
        
#         if current_id!= original_id:
#             player_detections.tracker_id[idx] = int(current_id)
            
#         player_db[current_id]['crops'].append(crop)



# < =========================================== Test ===========================================>
import os

input_path = 'assets/videos/15sec_input_720p.mp4'
output_path = 'assets/videos/0-10.mp4'
max_frames = 10

os.makedirs(os.path.dirname(output_path), exist_ok=True)

cap = cv2.VideoCapture(input_path)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_count = 0

while frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)
    frame_count += 1

cap.release()
out.release()
print(f"Saved first {frame_count} frames to {output_path}")

sv.process_video(
    source_path=output_path,
    target_path="output/final_tracking_with_reid.mp4",
    callback=process_frame
)
