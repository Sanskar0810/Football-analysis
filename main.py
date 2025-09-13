from utils.video_utils import read_video, save_video
from utils import draw_ellipse, draw_triangle, draw_team_ball_control
import os
import numpy as np
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from heatmap_generator import HeatMapGenerator

def main():
    # Read Video
    video_path = r"C:\Users\sanskar\OneDrive\Desktop\FA\project\input_videos\match_trim.mp4"
    output_path = r"C:\Users\sanskar\OneDrive\Desktop\FA\project\output_videos\output_video.avi"
    
    print(f"🔍 Attempting to read: {video_path}")

    if not os.path.exists(video_path):
        print(f"❌ Error: File not found at {video_path}")
        return
    
    video_frames = read_video(video_path)
    print(f"✅ Successfully read {len(video_frames)} frames.")
    
    # Initialize Tracker
    tracker = Tracker('models/best.pt')
    
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl'
                                       )
    
    # Get object positions 
    tracker.add_position_to_tracks(tracks)

    # Camera movement estimator
    camera_movement_estimator = CameraMovementEstimator()
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=True,
                                                                                stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # View Transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate ball positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks["players"][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Assign Ball Acquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['ball'][frame_num][1]['has_ball'] = True
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)

    # Draw output 
    ## Draw object Tracks
    output_video_frames = []
    for frame_num, frame in enumerate(video_frames):
        frame = frame.copy()

        player_dict = tracks["players"][frame_num]
        ball_dict = tracks["ball"][frame_num]
        referee_dict = tracks["referees"][frame_num]

        # Draw players
        for track_id, player in player_dict.items():
            color = player.get("team_color", (0, 0, 255))
            frame = draw_ellipse(frame, player["bbox"], color, track_id)

            if player.get('has_ball', False):
                frame = draw_triangle(frame, player["bbox"], (0, 0, 255))

        # Draw Referee
        for _, referee in referee_dict.items():
            frame = draw_ellipse(frame, referee["bbox"], (0, 255, 255))
            
        # Draw ball 
        for track_id, ball in ball_dict.items():
            frame = draw_triangle(frame, ball["bbox"], (0, 255, 0))

        # Draw Team Ball Control
        frame = draw_team_ball_control(frame, frame_num, team_ball_control)

        output_video_frames.append(frame)

    # Draw camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    # Draw speed and distance
    output_video_frames = speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # Generate Heat Maps
    print("🗺️ Generating heat maps...")
    heatmap_generator = HeatMapGenerator()
    
    # Generate individual player heat maps
    heatmap_generator.generate_individual_heatmaps(tracks, 'output_heatmaps/individual')
    
    # Generate team heat maps
    heatmap_generator.generate_team_heatmaps(tracks, 'output_heatmaps/teams')
    
    # Generate combined team heat map
    heatmap_generator.generate_combined_team_heatmap(tracks, 'output_heatmaps/combined')
    
    # Generate video with heat map overlay (optional - creates a second video)
    print("🎥 Generating heat map overlay video...")
    heatmap_video_frames = heatmap_generator.generate_video_overlay_heatmap(tracks, video_frames, 
                                                                           'output_videos/heatmap_overlay.avi')
    
    # Save both videos
    print(f"🔍 Attempting to save main video to: {output_path}")
    save_video(output_video_frames, output_path)
    
    print(f"🔍 Attempting to save heatmap video to: output_videos/heatmap_overlay.avi")
    save_video(heatmap_video_frames, 'output_videos/heatmap_overlay.avi')
    
    print("✅ Video analysis complete and saved successfully!")
    print("🗺️ Heat maps saved in 'output_heatmaps' directory!")
    print("🎥 Two videos generated:")
    print("   1. Main analysis video with tracking")
    print("   2. Heat map overlay video showing movement patterns")

if __name__ == '__main__':
    main()
    
    
    #iugtgtriuetiuiu