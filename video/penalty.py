import numpy as np
import cv2
import supervision as sv
import math
import os 
import sys
import matplotlib.pyplot as plt





vis = False
vis1 = True

video_path = 'assets/videos/15sec_input_720p.mp4'

def get_frame(video_path: str = 'assets/videos/15sec_input_720p.mp4', frame_no: int = 10, vis: bool = False) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    _, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if vis:
        plt.imshow(frame_rgb)
        plt.show()
    return frame


# * fps: 25.0 , w : 1280.0, h : 720.0, pos : 0.0
cap = cv2.VideoCapture(video_path)

for i in range(70):
    frame_number = i*5
    # if frame_number!=300: 
    #     continue

    # get framew
    frame = get_frame(video_path=video_path, frame_no=frame_number, vis=False)
    original_frame = frame.copy()


    # convert to hsv
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if vis: sv.plot_image(hsv)

    # mask green color to use field only
    lower_green = np.array([30, 30, 30])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    if vis: sv.plot_image(mask)

    # Morphological image processing
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    if vis: sv.plot_image(mask)


    # Get field boundaries
    green_rows = np.where(mask.any(axis=1))[0]
    if len(green_rows) > 0:
        field_top = green_rows.min()
        field_bottom = green_rows.max()
        field_height = field_bottom - field_top
        min_vertical_length = field_height * 0.8
    # if vis:
    #     cv2.line(original_frame, (0, field_top), (original_frame.shape[1], field_top), (255, 0, 0), 2)
    #     cv2.line(original_frame, (0, field_bottom), (original_frame.shape[1], field_bottom), (255, 0, 0), 2)
    #     cv2.imshow('original frame', original_frame)
    #     cv2.waitKey(0)

    # bgr to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if vis: sv.plot_image(gray)

    # Mask white lines
    _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    if vis: sv.plot_image(white_mask)

    # Combine whitemask and greenmask
    white_mask = cv2.bitwise_and(white_mask, mask)
    if vis: sv.plot_image(white_mask)

    # Clean up white lines
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    white_mask = cv2.dilate(white_mask, kernel, iterations=1)
    if vis: sv.plot_image(white_mask)

    blurred = cv2.GaussianBlur(white_mask, (5, 5), 0)
    if vis: sv.plot_image(blurred)

    edges = cv2.Canny(blurred, 30, 100)
    if vis: sv.plot_image(edges)

    edges = cv2.dilate(edges, kernel, iterations=1)
    if vis: sv.plot_image(edges)

    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, maxLineGap=20, minLineLength=min_vertical_length*0.55)
    # print(len(lines))

    def compute_line_angle(x1, y1, x2, y2):
        return math.degrees(math.atan2(y2 - y1, x2 - x1))


    def midpoint(x1, y1, x2, y2):
        return ((x1 + x2) / 2, (y1 + y2) /2)

    def line_distance(mp1, mp2):
        return math.hypot(mp1[0] - mp2[0], mp1[1] - mp2[1])
    
    def get_line_params(x1, y1, x2, y2):
        # Line: ax + by + c = 0
        a = y2 - y1
        b = x1 - x2
        c = x2*y1 - x1*y2
        return a, b, c
    
    def perpendicular_distance(line1, line2):
        # Line1 is reference, measure distance from line2's midpoint
        a, b, c = get_line_params(*line1)
        mx, my = midpoint(*line2)
        numerator = abs(a * mx + b * my + c)
        denominator = math.sqrt(a ** 2 + b ** 2)
        return numerator / denominator


    if lines is not None:
        center_line = []
        penalty_line = []
        maxi = 0
        longest_line = None
        sum = 0
        lst = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)

            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            angle = abs(angle)
            sum += angle

            
            if (60 <= angle <= 120) and (length >= min_vertical_length):
                if length>maxi:
                    maxi = length
                    longest_line = line
                center_line.append(line)
            else:
                if vis1: cv2.line(original_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for other lines

            
            if (10 <= angle <= 20) and (length >= min_vertical_length * 0.7):
                lst.append((x2*y1 - x1*y2)/(x2-x1))
                penalty_line.append(line)


            
        
            # for linea in lines:
            #     xa1, ya1, xa2, ya2 = linea[0]
            #     line1 = (x1, y1, x2, y2)
            #     line2 = (xa1, ya1, xa2, ya2)

            #     a1 = compute_line_angle(x1, y1, x2, y2)
            #     a2 = compute_line_angle(xa1, ya1, xa2, ya2)

            #     if abs(a1-a2)<5 and  perpendicular_distance(line1, line2) > 10:
            #         if vis1: cv2.line(original_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            #         if vis1: cv2.line(original_frame, (xa1, ya1), (xa2, ya2), (0, 0, 255), 3)


        # sum/=len(lines)
        # print(sum)
        # print(lst)
        
        # print(penalty_line)
        epsilon = 15  # tolerance for grouping similar c values
        line_clusters = []

        c_lines = []

        if penalty_line:
            for line in penalty_line:
                x1, y1, x2, y2 = line[0]
                c_val = (x2 * y1 - x1 * y2) / (x2 - x1)
                length = np.hypot(x2 - x1, y2 - y1)
                c_lines.append((c_val, line, length))

            # print(c_lines)
            
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
            for cluster in clusters:
                best_line = max(cluster, key=lambda x: x[2])
                selected_lines.append(best_line[1])
            print(len(selected_lines))

            for line in selected_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(original_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            c_values = []
            for line in selected_lines:
                x1, y1, x2, y2 = line[0]
                c_val = (x2 * y1 - x1 * y2) / (x2 - x1)
                c_values.append(c_val)

            lines_with_c = [(c, line[0]) for c, line in zip(c_values, selected_lines)]
            lines_sorted = sorted(lines_with_c, key=lambda x: x[0])
            keypoints = []

            if len(selected_lines) == 2:
                left_line = lines_sorted[1][1]
                right_line = lines_sorted[0][1]

                lx1, ly1, lx2, ly2 = left_line
                left_points = sorted([(lx1, ly1), (lx2, ly2)], key=lambda p: p[1])
                top_left = left_points[0]
                bottom_left = left_points[1]
                mid_left = midpoint(*top_left, *bottom_left)

                rx1, ry1, rx2, ry2 = right_line
                right_points = sorted([(rx1, ry1), (rx2, ry2)], key=lambda p: p[1])
                top_right = right_points[0]

                keypoints = [top_right, top_left, bottom_left, mid_left]

            elif len(selected_lines) == 3:
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

                keypoints = [top_right, top_left, bottom_left, top_mid, bottom_mid]

            for (x, y) in keypoints:
                cv2.circle(original_frame, (int(x), int(y)), radius=5, color=(255, 0, 0), thickness=-1)
                    

        



        

        # for pline in penalty_line:
        #     x1, y1, x2, y2 = pline[0]
        #     if vis1: cv2.line(original_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red for center line



        if longest_line is not None:
            x1, y1, x2, y2 = longest_line[0]
            print(x1, y1, x2, y2)
            m=0.56
            n=2-m
            offset = 400
            center_x = int((m * x1 + n * x2)/(m+n))
            center_y = int((m * y1 + n * y2)/(m+n))
            print(center_x, center_y)
            if vis1: cv2.line(original_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red for center line
            if vis1: cv2.circle(original_frame, (center_x + offset, center_y), radius = 3, color=(0, 0, 255), thickness=3)
            if vis1: cv2.circle(original_frame, (center_x - offset, center_y), radius = 3, color=(0, 0, 255), thickness=3)
            
            # print(len(center_line))
    if vis1: cv2.putText(
                        img=original_frame,
                        text=str(frame_number),
                        org=(50, 100),  # x, y
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(0, 0, 255),
                        thickness=2,
                        lineType=cv2.LINE_AA
                    )

    if vis1: 
        cv2.imshow('Original Frame', original_frame)
        cv2.waitKey(0)
    
