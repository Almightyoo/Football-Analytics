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



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

reid_model = torchreid.models.build_model('osnet_ibn_x1_0', num_classes=1000, pretrained=True)
reid_model.eval().to(device)
print(f"Total parameters: {sum(p.numel() for p in reid_model.parameters())}")
