import numpy as np
import cv2
import supervision as sv
import math
import os 
import sys
import matplotlib.pyplot as plt
from typing import Tuple

from get_frame import get_frame

video_path = 'assets/videos/15sec_input_720p.mp4'




def get_homograph_matrix_H(frame: np.ndarray, frame_number: int = 0):
    H = None

    def midpoint(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float]:
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    

    # 1. Field Detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 30, 30])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Get field boundaries
    green_rows = np.where(mask.any(axis=1))[0]
    if len(green_rows) > 0:
        field_top = green_rows.min()
        field_bottom = green_rows.max()
        field_height = field_bottom - field_top
        min_vertical_length = field_height * 0.8

    # Line Detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    white_mask = cv2.bitwise_and(white_mask, mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    white_mask = cv2.dilate(white_mask, kernel, iterations=1)

    # Edge Detection
    blurred = cv2.GaussianBlur(white_mask, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    #Line Detection
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, maxLineGap=20, minLineLength=min_vertical_length*0.55)
    
    # Line Classification
    if lines is not None:
        center_line = []
        penalty_line = []
        maxi = 0
        longest_line = None
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)

            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            angle = abs(angle)

            # Center line criteria
            if (60 <= angle <= 120) and (length >= min_vertical_length):
                if length>maxi:
                    maxi = length
                    longest_line = line
                center_line.append(line)

            # Penalty area line criteria
            if (10 <= angle <= 20) and (length >= min_vertical_length * 0.7):
                penalty_line.append(line)
            
        # Homography Calculation for CenterLine 
        if longest_line is not None:
            x1, y1, x2, y2 = longest_line[0]
            m=0.56
            n=2-m
            offset = 400
            center_x = int((m * x1 + n * x2)/(m+n))
            center_y = int((m * y1 + n * y2)/(m+n))

            image_points = np.array([
                [x2, y2],  # Bottom end of the center line
                [x1, y1],  # Top end of the center line
                [center_x - offset, center_y], # left center of circle
                [center_x + offset, center_y], # right edge of circle
                [center_x, center_y]
            ], dtype=np.float32)

            pitch_points = np.array([
                [6000, 0],     # Bottom end of the center line
                [6000, 7000],  # Top end of the center line
                [5085, 3500],  # center of circle
                [6915, 3500],  # right edge of circle
                [6000, 3500]
            ], dtype=np.float32)

            H, _ = cv2.findHomography(image_points, pitch_points, method=cv2.RANSAC, ransacReprojThreshold=5.0)
            return H
        
        epsilon = 15 
        line_clusters = []

        c_lines = []

        # Homography Calculation for Penalty Line 
        if penalty_line:
            for line in penalty_line:
                x1, y1, x2, y2 = line[0]
                c_val = (x2 * y1 - x1 * y2) / (x2 - x1)
                length = np.hypot(x2 - x1, y2 - y1)
                c_lines.append((c_val, line, length))
            c_lines.sort(key=lambda x: x[0])

            clusters = []
            current_cluster = [c_lines[0]]

            for c_val, line, length in c_lines[1:]:
                last_c_val = current_cluster[-1][0]
                if abs(c_val - last_c_val) < epsilon:
                    current_cluster.append((c_val, line, length))
                else:
                    clusters.append(current_cluster)
                    current_cluster = [(c_val, line, length)]

            clusters.append(current_cluster) 

            selected_lines = []

            # cluster similar lines so that you have only the longest, best line as penalty line, expected to have 2-3 lines in selected lines 
            # because there are 3 parallel straight lines in the penalty area
            for cluster in clusters:
                best_line = max(cluster, key=lambda x: x[2])
                selected_lines.append(best_line[1])

            c_values = []
            for line in selected_lines:
                x1, y1, x2, y2 = line[0]
                c_val = (x2 * y1 - x1 * y2) / (x2 - x1)
                c_values.append(c_val)

            lines_with_c = [(c, line[0]) for c, line in zip(c_values, selected_lines)]
            lines_sorted = sorted(lines_with_c, key=lambda x: x[0])
            keypoints = []

            # number of parallel lines is 2

            # if len(selected_lines) == 2:
            #     left_line = lines_sorted[1][1]
            #     right_line = lines_sorted[0][1]

            #     lx1, ly1, lx2, ly2 = left_line
            #     left_points = sorted([(lx1, ly1), (lx2, ly2)], key=lambda p: p[1])
            #     top_left = left_points[0]
            #     bottom_left = left_points[1]
            #     mid_left = midpoint(*top_left, *bottom_left)

            #     rx1, ry1, rx2, ry2 = right_line
            #     right_points = sorted([(rx1, ry1), (rx2, ry2)], key=lambda p: p[1])
            #     top_right = right_points[0]

            #     keypoints = [top_right, top_left, bottom_left, mid_left]

            #     image_points = np.array(keypoints, dtype=np.float32)

            #     pitch_points = np.array([
            #         [12000, 0],     
            #         [9985, 1480],  
            #         [9985, 3500],  
            #         [9985, 5550],
            #     ], dtype=np.float32)

            #     H, _ = cv2.findHomography(image_points, pitch_points, method=cv2.RANSAC, 5.0)
            #     return H


            # number of parallel lines is 3 
            if len(selected_lines) == 3:
                left_line = lines_sorted[2][1]
                middle_line = lines_sorted[1][1]
                right_line = lines_sorted[0][1]

                lx1, ly1, lx2, ly2 = left_line
                left_points = sorted([(lx1, ly1), (lx2, ly2)], key=lambda p: p[1])
                top_left = left_points[0]
                bottom_left = left_points[1]

                mx1, my1, mx2, my2 = middle_line
                mid_points = sorted([(mx1, my1), (mx2, my2)], key=lambda p: p[1])
                top_mid = mid_points[0]
                bottom_mid = mid_points[1]

                rx1, ry1, rx2, ry2 = right_line
                right_points = sorted([(rx1, ry1), (rx2, ry2)], key=lambda p: p[1])
                top_right = right_points[0]

                keypoints = [top_right, top_left, bottom_left, bottom_left, bottom_mid]

                image_points = np.array(keypoints, dtype=np.float32)

                pitch_points = np.array([
                    [12000, 0],       # top_right
                    [9985, 1480],     # top_left
                    [9985, 3500],     # bottom_left  
                    [11450, 2584],    # bottom_left
                    [11450, 4416]     # bottom_mid
                ], dtype=np.float32)

                H, _ = cv2.findHomography(image_points, pitch_points, method=cv2.RANSAC, ransacReprojThreshold=5.0)
                return H
    return H

# < =========================================== Test ===========================================>



if __name__ == "__main__":
    #Test 1
    video_path = 'assets/videos/15sec_input_720p.mp4'
    cap = cv2.VideoCapture(video_path)

    for i in range(70):
        frame_number = i*5
        frame = get_frame(video_path=video_path, frame_no=frame_number, vis = False)
        H = get_homograph_matrix_H(frame)
        if H is not None:
            print(1)
        else:
            print(0)


    #Test 2
    for i in range(50):
        frame_number = 0 + i
        frame = get_frame(video_path=video_path, frame_no=frame_number, vis=False)

        H = get_homograph_matrix_H(frame, frame_number)

        if H is not None:
            print("Homography matrix:\n", H)

            output_size = (12000, 7000)
            warped = cv2.warpPerspective(frame, H, output_size)

            cv2.imshow("Original Frame", frame)
            cv2.imshow("Warped Frame", warped)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            break
        else:
            print("Homography could not be computed.")