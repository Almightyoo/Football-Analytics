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


# Reid mdoel initialization
reid_model = torchreid.models.build_model('osnet_x1_0', num_classes=1000, pretrained=True)
reid_model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
reid_model = reid_model.to(device)


# Reid mdoel transform
reid_transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# transform from to give reid model
def extract_features(crop):
    img = Image.fromarray(crop)
    img = reid_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        return reid_model(img).cpu().numpy()


# variables
video_path = 'assets/videos/15sec_input_720p.mp4'
tracker = sv.ByteTrack()
tracker.reset()

# {player_id: {'crops': [list]}
player_db = {}  



for frame_number in range(100, 110):
    frame = get_frame(video_path, frame_number, vis=False)
    
    model = YOLO('models/finetuned_best.pt')
    model.conf = 0.3
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    detections = detections[detections.class_id != 0]
    detections = detections.with_nms(threshold=0.3, class_agnostic=True)
    detections.xyxy = sv.pad_boxes(detections.xyxy, px=10)
    detections = tracker.update_with_detections(detections)
    player_detections = detections[detections.class_id == 2]
    print(player_detections.tracker_id)

    for xyxy, tracker_id in zip(player_detections.xyxy, player_detections.tracker_id):
        crop = sv.crop_image(frame, xyxy)
        current_features = extract_features(crop)
        current_id = f"#{tracker_id}"
        
        if current_id not in player_db:
            best_match_id = None
            best_similarity = 0.9  # similarity threshold
            
            for existing_id, data in player_db.items():
                if not data['features']:
                    continue
                    
                # Compare with last stored features
                similarity = cosine_similarity(
                    current_features, 
                    data['features'][-1]
                )[0][0]
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = existing_id
            
            # Reassign ID if found good match
            if best_match_id:
                print(f"Reassigned {current_id} -> {best_match_id} (sim: {best_similarity:.2f})")
                current_id = best_match_id
        
        # Update database
        if current_id not in player_db:
            player_db[current_id] = {'crops': [], 'features': []}
        
        player_db[current_id]['crops'].append(crop)
        player_db[current_id]['features'].append(current_features)


# < =========================================== Test ===========================================>

# # Visualization
# for player_id, data in player_db.items():
#     print(f"\nPlayer {player_id} history:")
    
#     # Filter out empty/invalid crops
#     valid_crops = []
#     valid_titles = []
    
#     for i, crop in enumerate(data['crops']):
#         if crop is not None and isinstance(crop, np.ndarray) and crop.size > 0:
#             # Convert from BGR to RGB if needed
#             if crop.shape[-1] == 3:  # Color image
#                 valid_crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
#             else:  # Grayscale or unexpected format
#                 valid_crops.append(crop)
#             valid_titles.append(f"Frame {i}")
    
#     if valid_crops:
#         sv.plot_images_grid(
#             valid_crops,
#             grid_size=(1, len(valid_crops)),
#             size=(16, 2),
#             titles=valid_titles
#         )
#         plt.show()
#     else:
#         print(f"No valid crops found for {player_id}")