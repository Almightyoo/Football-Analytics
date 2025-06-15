import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Optional


def get_frame(video_path: str = 'assets/videos/15sec_input_720p.mp4', frame_no: int = 10, vis: bool = False) -> Optional[np.ndarray]:
    """
    Extracts a specific frame from a video file with optional visualization.
    
    Args:
        video_path ('assets/videos/15sec_input_720p.mp4'): Path to the video file.
        frame_no (10): Frame number to extract (0-indexed).
        vis (False): Whether to display the frame using matplotlib.
        
    Returns:
        np.ndarray: The frame in RGB format
    """
    
    
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    cool, frame = cap.read()
    if not cool:
        return None
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if vis:
        plt.imshow(frame_rgb)
        plt.show()
    return frame

# < =========================================== Test ===========================================>
if __name__ == "__main__":
    frame1 = get_frame(vis=True)
    print("Handle error cases")
    invalid_frame = get_frame(frame_no=9999)
    if invalid_frame is None:
        print("Successfully handled out-of-bounds frame request")