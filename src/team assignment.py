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
from sklearn.mixture import GaussianMixture

def extract_team_assignments(
    video_path: str,
    frame_indices: list,
    model_path: str = 'models/finetuned_best.pt',
    num_teams: int = 2,
    display: bool = True
):
    """
    Detects and classifies soccer players into team clusters based on jersey color using GMM.

    Args:
        video_path (str): Path to the input video file.
        frame_indices (list): List of frame indices to process.
        model_path (str): Path to the YOLO model for player detection.
        num_teams (int): Number of teams to cluster.
        display (bool): Whether to visualize the clustering results.

    Returns:
        dict: A dictionary mapping frame indices to cluster assignments and team color data:
    """


    results_by_frame = {}
    reference_team_colors = None
    first_frame_processed = False

    model = YOLO(model_path)
    model.conf = 0.3

    def process_player_crop(crop):
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lower_green = np.array([30, 30, 30])
        upper_green = np.array([90, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        blurred_mask = cv2.GaussianBlur(green_mask, (5, 5), 0)
        player_mask = cv2.bitwise_not(blurred_mask)
        kernel = np.ones((3, 3), np.uint8)
        player_mask = cv2.morphologyEx(player_mask, cv2.MORPH_OPEN, kernel)
        player_mask = cv2.morphologyEx(player_mask, cv2.MORPH_DILATE, kernel, iterations=1)
        masked_player = cv2.bitwise_and(crop, crop, mask=player_mask)
        return masked_player, player_mask

    def get_dominant_color(img, mask=None):
        if mask is not None:
            img = cv2.bitwise_and(img, img, mask=mask)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3)
        pixels = pixels[np.all(pixels != [0, 0, 0], axis=1)]
        return np.median(pixels, axis=0)

    for frame_index in frame_indices:
        frame = get_frame(video_path=video_path, frame_no=frame_index, vis=False)
        detections = sv.Detections.from_ultralytics(model(frame)[0])
        detections = detections[detections.class_id == 2]
        detections.xyxy = sv.pad_boxes(detections.xyxy, px=10)

        player_crops = [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]
        processed_crops, all_colors = [], []

        for crop in player_crops:
            masked, mask = process_player_crop(crop)
            dominant = get_dominant_color(masked, mask)
            processed_crops.append(masked)
            all_colors.append(dominant)

        player_colors = np.array(all_colors)
        features = player_colors[:, :2]

        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)

        gmm = GaussianMixture(n_components=num_teams, covariance_type='spherical')
        raw_clusters = gmm.fit_predict(scaled)

        
        team_colors = [np.median(player_colors[raw_clusters == i], axis=0) for i in range(num_teams)]

        if not first_frame_processed:
            reference_team_colors = team_colors
            first_frame_processed = True
            color_mapping = {i: i for i in range(num_teams)}
        else:
            
            distances = np.array([
                [np.linalg.norm(team_colors[i] - reference_team_colors[j]) for j in range(num_teams)]
                for i in range(num_teams)
            ])
            color_mapping = {i: int(np.argmin(distances[i])) for i in range(num_teams)}

        consistent_clusters = np.array([color_mapping[c] for c in raw_clusters])

        results_by_frame[frame_index] = {
            'team_assignments': consistent_clusters.tolist(),
            'team_colors': team_colors
        }

        if display:
            cols = 10
            rows = int(np.ceil(len(processed_crops) / cols))
            plt.figure(figsize=(20, rows * 2.5))
            for i, (crop, cluster_id) in enumerate(zip(processed_crops, consistent_clusters)):
                plt.subplot(rows, cols, i + 1)
                plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                color = 'red' if cluster_id == 0 else 'blue'
                plt.title(f"Team {cluster_id + 1}", color=color)
                plt.axis('off')
            plt.suptitle(f"Frame {frame_index} - Team Assignments", fontsize=16)
            plt.tight_layout()
            plt.show()

    return results_by_frame






# < =========================================== Test ===========================================>
if __name__ == "__main__":
    frame_indices = [i for i in range(5)]
    results = extract_team_assignments(
        video_path='assets/videos/15sec_input_720p.mp4',
        frame_indices=frame_indices,
        model_path='models/finetuned_best.pt',
        num_teams=2,
        display=True
    )
