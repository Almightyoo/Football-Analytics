# import supervision as sv
# from get_frame import get_frame
# from ultralytics import YOLO
# import torchreid
# import torch
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from torchvision import transforms
# from PIL import Image
# import cv2

# reid_model = torchreid.models.build_model('osnet_x1_0', num_classes=1000, pretrained=True)
# reid_model.eval()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# reid_model = reid_model.to(device)

# reid_transform = transforms.Compose([
#     transforms.Resize((256, 128)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# def extract_features(crop):
#     """Extract re-ID features from player crop"""
#     img = Image.fromarray(crop)
#     img = reid_transform(img).unsqueeze(0).to(device)
#     with torch.no_grad():
#         return reid_model(img).cpu().numpy()

# video_path = 'assets/videos/15sec_input_720p.mp4'
# output_video_path = 'output/annotated_players2.mp4'

# video_info = sv.VideoInfo.from_video_path(video_path)
# frame_generator = sv.get_video_frames_generator(video_path, start=0)

# tracker = sv.ByteTrack()
# tracker.reset()

# box_annotator = sv.BoundingBoxAnnotator()
# label_annotator = sv.LabelAnnotator()

# player_db = {}  


# with sv.VideoSink(output_video_path, video_info) as sink:
#     for frame_number, frame in enumerate(frame_generator):
#         if frame_number >= 20:
#             break
            
#         model = YOLO('models/best.pt')
#         model.conf = 0.3
#         results = model(frame)[0]
#         detections = sv.Detections.from_ultralytics(results)
        
#         detections = detections[detections.class_id != 0]
#         detections = detections.with_nms(threshold=0.3, class_agnostic=True)
#         detections.xyxy = sv.pad_boxes(detections.xyxy, px=10)
#         detections = tracker.update_with_detections(detections)
#         detections.class_id -= 1
#         player_detections = detections[detections.class_id == 1]

#         labels = [
#             f"ID: {tracker_id}"
#             for tracker_id in player_detections.tracker_id
#         ]

#         annotated_frame = box_annotator.annotate(
#         frame.copy(), detections=player_detections)
#         annotated_frame = label_annotator.annotate(
#         annotated_frame, detections=player_detections, labels=labels)

#         sink.write_frame(annotated_frame)

# print(f"Video saved to {output_video_path}")



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


# ---------- Initialization ----------
video_path = '/kaggle/input/videos/15sec_input_720p.mp4'
# video_path = '/kaggle/input/100frames/0-100.mp4'
# video_path = '/kaggle/input/10frames/0-10.mp4'
output_path = "/kaggle/working/final_tracking_with_reid.mp4"
player_detection_model_path = '/kaggle/input/best_model_football/pytorch/default/1/best.pt'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# Load YOLO model once
yolo_model = YOLO(player_detection_model_path).to(device)
yolo_model.conf = 0.3

# Initialize ReID model
reid_model = torchreid.models.build_model('osnet_x1_0', num_classes=1000, pretrained=True)
reid_model.eval().to(device)

# Trackers and database
tracker = sv.ByteTrack(frame_rate = 25)
tracker.reset()
player_db = {}

# ReID transform
reid_transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ---------- Utility Functions ----------
def extract_features(crop):
    img = Image.fromarray(crop)
    img = reid_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        return reid_model(img).cpu().numpy()




def get_player_detections(frame):
    results = yolo_model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[detections.class_id == 2]

    # ✅ Apply correct NMS
    # detections = detections.with_nms(threshold=0.5, class_agnostic=False)

    detections.xyxy = sv.pad_boxes(detections.xyxy, px=10)
    detections = tracker.update_with_detections(detections)
    return detections[detections.class_id == 2]


import math

def get_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def torch_reid(current_id, crop, xyxy, flag: bool):
    best_similarity_threshold = 0.80 if flag else 0.90
    current_features = extract_features(crop)
    current_center = get_center(xyxy)

    best_match_id = current_id
    best_score = 0

    for existing_id, data in player_db.items():
        if existing_id == current_id or len(data['features']) == 0:
            continue

        gallery_features = data['features'][-1]
        similarity = cosine_similarity(current_features, gallery_features)[0][0]
        if similarity < best_similarity_threshold:
            continue

        appearances = data.get('appearances', 1)
        past_center = data.get('last_position', current_center)

        distance = euclidean(current_center, past_center)
        distance_score = 1 / (1 + distance)

        score = similarity * math.log(1 + appearances) * distance_score

        if score > best_score:
            best_score = score
            best_match_id = existing_id

    return best_match_id, current_features, current_center





def plot_reid_comparison(query_crop, gallery_crop, query_id, gallery_id, similarity):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(query_crop)
    ax1.set_title(f"Query Player (ID: {query_id})")
    ax1.axis('off')

    ax2.imshow(gallery_crop)
    ax2.set_title(f"Gallery Player (ID: {gallery_id})\nSim: {similarity:.2f}")
    ax2.axis('off')

    plt.suptitle(f"ReID Match: {query_id} → {gallery_id}")
    plt.tight_layout()
    plt.show()


# ---------- Frame Processor ----------
frame_counter = 0
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()


def process_frame(frame: np.ndarray, _: int) -> np.ndarray:
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


# ---------- Run Tracking ----------
print("Starting tracking and ReID process...")
sv.process_video(
    source_path=video_path,
    target_path=output_path,
    callback=process_frame
)

print("Processing complete. Output saved to:", output_path)
