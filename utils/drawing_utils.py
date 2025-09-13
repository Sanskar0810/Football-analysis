import cv2
import numpy as np
from .bbox_utils import *

def draw_ellipse(frame, bbox, color, track_id=None):
    y2 = int(bbox[3])
    x_center, _ = get_center_of_bbox(bbox)
    width = get_bbox_width(bbox)
    
    cv2.ellipse(
        frame,
        center=(x_center, y2),
        axes=(int(width), int(0.35 * width)),
        angle=0.0,
        startAngle=-45,
        endAngle=235,
        color=color,
        thickness=2,
        lineType=cv2.LINE_4
    )

    rectangle_width = 40
    rectangle_height = 20
    x1_rect = x_center - rectangle_width // 2
    x2_rect = x_center + rectangle_width // 2
    y1_rect = (y2 - rectangle_height // 2) + 15
    y2_rect = (y2 + rectangle_height // 2) + 15

    if track_id is not None:
        cv2.rectangle(frame,
                      (int(x1_rect), int(y1_rect)),
                      (int(x2_rect), int(y2_rect)),
                      color,
                      cv2.FILLED)
        
        x1_text = x1_rect + 12
        if track_id > 99:
            x1_text -= 10
        
        cv2.putText(
            frame,
            f"{track_id}",
            (int(x1_text), int(y1_rect + 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

    return frame

def draw_triangle(frame, bbox, color):
    y = int(bbox[1])
    x, _ = get_center_of_bbox(bbox)

    triangle_points = np.array([
        [x, y],
        [x - 10, y - 20],
        [x + 10, y - 20],
    ])
    cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
    cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

    return frame

def draw_team_ball_control(frame, frame_num, team_ball_control):
    # Draw a semi-transparent rectangle
    overlay = frame.copy()
    cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    team_ball_control_till_frame = team_ball_control[:frame_num+1]
    # Get the number of times each team had ball control
    team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
    team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
    team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
    team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)

    cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.1f}%",(1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.1f}%",(1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

    return frame