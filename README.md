# Football Analysis Project

## Overview
This project analyzes football match videos using computer vision and deep learning. It tracks players and the ball, assigns teams, estimates camera movement, calculates speed and distance, and generates heatmaps and annotated videos for tactical analysis.

## Features
- **Player & Ball Detection:** YOLO-based detection in video frames.
- **Tracking:** Tracks objects across frames.
- **Team Assignment:** Assigns players to teams.
- **Ball Possession:** Identifies which player has the ball.
- **Camera Movement Estimation:** Corrects for panning/zooming.
- **Speed & Distance Estimation:** Calculates player metrics.
- **Heatmap Generation:** Visualizes player and team movement.
- **Output Videos:** Annotated and overlayed videos.

## Directory Structure
- input_videos: Raw match videos
- models: YOLO model weights
- output_heatmaps: Generated heatmaps (individual, team, combined)
- output_videos: Annotated videos
- runs: YOLO detection runs
- stubs: Intermediate results
- trackers: Tracking utilities
- training: Training scripts and datasets
- utils: Utility functions

## Input
- **Videos:** Place in input_videos (e.g., `match.mp4`)
- **YOLO Weights:** Place in models (e.g., `best.pt`)

## Output
- **Heatmaps:** In individual, `teams/`, and `combined/`
- **Videos:** In output_videos (e.g., `heatmap_overlay.avi`)

## How It Works
1. **YOLO Inference:** Detects players/ball (yolo_inference.py)
2. **Tracking:** Tracks objects (tracker.py)
3. **Team Assignment:** Assigns teams (team_assigner.py)
4. **Ball Possession:** Assigns ball (player_ball_assigner.py)
5. **Camera Movement:** Estimates movement (camera_movement_estimator.py)
6. **Speed & Distance:** Calculates metrics (speed_and_distance_estimator.py)
7. **Heatmaps:** Generates heatmaps (heatmap_generator.py)
8. **Video Generation:** Annotates videos (view_transformer.py)

## Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Input
- Place video in input_videos
- Place YOLO weights in models

### 3. Run Main Pipeline
```bash
python main.py
```

### 4. View Outputs
- Heatmaps: output_heatmaps
- Videos: output_videos

## Customization
- Train your own YOLO models in training
- Adjust parameters in scripts as needed

## Example Workflow
1. Place `match.mp4` in input_videos
2. Place `best.pt` in models
3. Run `python main.py`
4. Find outputs in output_heatmaps and output_videos

## File Descriptions
- main.py: Main pipeline
- yolo_inference.py: YOLO detection
- tracker.py: Tracking
- team_assigner.py: Team assignment
- player_ball_assigner.py: Ball possession
- camera_movement_estimator.py: Camera movement
- speed_and_distance_estimator.py: Speed/distance
- heatmap_generator.py: Heatmaps
- view_transformer.py: Perspective transforms
- utils: Utility functions

## Requirements
- Python 3.10+
- OpenCV, PyTorch, Ultralytics YOLO, NumPy, Matplotlib
- See `requirements.txt` for full list

## Notes
- Use high-quality videos and trained models for best results
- Outputs are saved automatically
- Intermediate results cached in stubs

## License
For educational and research purposes.

## Contact
For questions or contributions, contact the repository owner.

---

Let me know if you want this written directly to your README.md file.