import sys
import os
import subprocess
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QLabel, QFileDialog, QProgressBar,
                             QTextEdit, QScrollArea, QGridLayout, QFrame, QSplitter,
                             QGroupBox, QMessageBox)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QPixmap, QFont, QPalette, QColor, QIcon

# Import your analysis modules
try:
    from utils.video_utils import read_video, save_video
    from utils import draw_ellipse, draw_triangle, draw_team_ball_control
    import numpy as np
    from trackers import Tracker
    from team_assigner import TeamAssigner
    from player_ball_assigner import PlayerBallAssigner
    from camera_movement_estimator import CameraMovementEstimator
    from view_transformer import ViewTransformer
    from speed_and_distance_estimator import SpeedAndDistance_Estimator
    from heatmap_generator import HeatMapGenerator
    ANALYSIS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Analysis modules not available: {e}")
    ANALYSIS_AVAILABLE = False


class AnalysisWorker(QThread):
    """Worker thread for running video analysis"""
    progress_update = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        
    def run(self):
        try:
            if not ANALYSIS_AVAILABLE:
                self.finished.emit(False, "Analysis modules not available")
                return
                
            self.progress_update.emit("🔍 Reading video...")
            
            # Create output directories
            os.makedirs('output_videos', exist_ok=True)
            os.makedirs('output_heatmaps/individual', exist_ok=True)
            os.makedirs('output_heatmaps/teams', exist_ok=True)
            os.makedirs('output_heatmaps/combined', exist_ok=True)
            
            # Read Video
            if not os.path.exists(self.video_path):
                self.finished.emit(False, f"File not found: {self.video_path}")
                return
                
            video_frames = read_video(self.video_path)
            self.progress_update.emit(f"✅ Successfully read {len(video_frames)} frames")
            
            # Initialize Tracker
            self.progress_update.emit("🎯 Initializing tracker...")
            tracker = Tracker('models/best.pt')
            
            self.progress_update.emit("🔄 Getting object tracks...")
            tracks = tracker.get_object_tracks(video_frames,
                                               read_from_stub=True,
                                               stub_path='stubs/track_stubs.pkl')
            
            # Get object positions 
            tracker.add_position_to_tracks(tracks)
            
            # Camera movement estimator
            self.progress_update.emit("📹 Estimating camera movement...")
            camera_movement_estimator = CameraMovementEstimator()
            camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
                video_frames, read_from_stub=True, stub_path='stubs/camera_movement_stub.pkl')
            camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)
            
            # View Transformer
            self.progress_update.emit("🔄 Transforming view...")
            view_transformer = ViewTransformer()
            view_transformer.add_transformed_position_to_tracks(tracks)
            
            # Interpolate ball positions
            self.progress_update.emit("⚽ Interpolating ball positions...")
            tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
            
            # Speed and distance estimator
            self.progress_update.emit("📊 Calculating speeds and distances...")
            speed_and_distance_estimator = SpeedAndDistance_Estimator()
            speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)
            
            # Assign Player Teams
            self.progress_update.emit("👥 Assigning player teams...")
            team_assigner = TeamAssigner()
            
            # Safety check for initial frame
            if tracks["players"] and len(tracks["players"]) > 0 and tracks["players"][0]:
                team_assigner.assign_team_color(video_frames[0], tracks["players"][0])
                
                # Ensure we don't exceed available frames
                max_frames = min(len(video_frames), len(tracks["players"]), len(tracks["ball"]), len(tracks["referees"]))
                
                for frame_num in range(max_frames):
                    player_track = tracks['players'][frame_num]
                    for player_id, track in player_track.items():
                        team = team_assigner.get_player_team(video_frames[frame_num], 
                                                           track['bbox'], player_id)
                        tracks['players'][frame_num][player_id]['team'] = team 
                        tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
            else:
                self.progress_update.emit("⚠️ No player tracks found for team assignment")
            
            # Assign Ball Acquisition
            self.progress_update.emit("⚽ Assigning ball possession...")
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
            self.progress_update.emit("🎨 Drawing analysis overlay...")
            output_video_frames = []

            # Use the minimum length across tracks to stay safe
            max_frames = min(
                len(video_frames),
                len(tracks.get("players", [])),
                len(tracks.get("ball", [])),
                len(tracks.get("referees", []))
            )

            for frame_num in range(max_frames):
                frame = video_frames[frame_num].copy()

                # Safely fetch dictionaries for this frame
                player_dict = tracks["players"][frame_num] if frame_num < len(tracks["players"]) else {}
                ball_dict = tracks["ball"][frame_num] if frame_num < len(tracks["ball"]) else {}
                referee_dict = tracks["referees"][frame_num] if frame_num < len(tracks["referees"]) else {}

                # Draw players
                for track_id, player in player_dict.items():
                    color = player.get("team_color", (0, 0, 255))
                    frame = draw_ellipse(frame, player["bbox"], color, track_id)

                    if player.get("has_ball", False):
                        frame = draw_triangle(frame, player["bbox"], (0, 0, 255))

                # Draw referees
                for _, referee in referee_dict.items():
                    frame = draw_ellipse(frame, referee["bbox"], (0, 255, 255))

                # Draw ball
                for track_id, ball in ball_dict.items():
                    frame = draw_triangle(frame, ball["bbox"], (0, 255, 0))

                # Draw team ball control (safe index check)
                if "team_ball_control" in locals() and frame_num < len(team_ball_control):
                    frame = draw_team_ball_control(frame, frame_num, team_ball_control)

                output_video_frames.append(frame)

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
            self.progress_update.emit("🗺️ Generating heat maps...")
            heatmap_generator = HeatMapGenerator()
            
            # Generate individual player heat maps
            heatmap_generator.generate_individual_heatmaps(tracks, 'output_heatmaps/individual')
            
            # Generate team heat maps
            heatmap_generator.generate_team_heatmaps(tracks, 'output_heatmaps/teams')
            
            # Generate combined team heat map
            heatmap_generator.generate_combined_team_heatmap(tracks, 'output_heatmaps/combined')
            
            # Generate video with heat map overlay
            self.progress_update.emit("🎥 Generating heat map overlay video...")
            heatmap_video_frames = heatmap_generator.generate_video_overlay_heatmap(
                tracks, video_frames, 'output_videos/heatmap_overlay.avi')
            
            # Save videos
            self.progress_update.emit("💾 Saving main analysis video...")
            output_path = 'output_videos/output_video.avi'
            save_video(output_video_frames, output_path)
            
            self.progress_update.emit("💾 Saving heatmap video...")
            save_video(heatmap_video_frames, 'output_videos/heatmap_overlay.avi')
            
            self.finished.emit(True, "Analysis completed successfully!")
            
        except Exception as e:
            self.finished.emit(False, f"Error during analysis: {str(e)}")


class ImageCard(QFrame):
    """Custom widget for displaying images with titles"""
    def __init__(self, image_path, title):
        super().__init__()
        self.setFrameStyle(QFrame.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                background-color: #3e3e3e;
                border-radius: 10px;
                border: 1px solid #555;
            }
            QFrame:hover {
                border: 2px solid #00aaff;
                background-color: #454545;
            }
        """)
        
        layout = QVBoxLayout()
        
        # Image label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(200, 150)
        self.image_label.setStyleSheet("border: none; background-color: transparent;")
        
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            scaled_pixmap = pixmap.scaled(200, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
        else:
            self.image_label.setText("Image not found")
            self.image_label.setStyleSheet("color: #888; border: none;")
        
        # Title label
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: white; font-weight: bold; margin: 5px; border: none;")
        
        layout.addWidget(self.image_label)
        layout.addWidget(title_label)
        self.setLayout(layout)
        
        # Make clickable
        self.mousePressEvent = lambda event: self.open_image(image_path)
    
    def open_image(self, path):
        if os.path.exists(path):
            if sys.platform.startswith('darwin'):  # macOS
                subprocess.run(['open', path])
            elif sys.platform.startswith('win'):   # Windows
                os.startfile(path)
            else:  # Linux
                subprocess.run(['xdg-open', path])


class VideoCard(QFrame):
    """Custom widget for video links"""
    def __init__(self, video_path, title):
        super().__init__()
        self.video_path = video_path
        self.setFrameStyle(QFrame.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                background-color: #2e2e2e;
                border-radius: 10px;
                border: 1px solid #555;
                min-height: 80px;
            }
            QFrame:hover {
                border: 2px solid #ff6b35;
                background-color: #3e3e3e;
            }
        """)
        
        layout = QHBoxLayout()
        
        # Video icon (using text for now)
        icon_label = QLabel("🎥")
        icon_label.setStyleSheet("font-size: 24px; border: none; padding: 10px;")
        
        # Video info
        info_layout = QVBoxLayout()
        title_label = QLabel(title)
        title_label.setStyleSheet("color: white; font-weight: bold; font-size: 14px; border: none;")
        
        status_label = QLabel("✅ Ready to play" if os.path.exists(video_path) else "❌ File not found")
        status_label.setStyleSheet("color: #888; font-size: 12px; border: none;")
        
        info_layout.addWidget(title_label)
        info_layout.addWidget(status_label)
        
        # Play button
        play_btn = QPushButton("▶ Play")
        play_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff6b35;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ff5722;
            }
            QPushButton:pressed {
                background-color: #e64a19;
            }
        """)
        play_btn.clicked.connect(self.play_video)
        
        layout.addWidget(icon_label)
        layout.addLayout(info_layout, 1)
        layout.addWidget(play_btn)
        self.setLayout(layout)
    
    def play_video(self):
        if os.path.exists(self.video_path):
            if sys.platform.startswith('darwin'):  # macOS
                subprocess.run(['open', self.video_path])
            elif sys.platform.startswith('win'):   # Windows
                os.startfile(self.video_path)
            else:  # Linux
                subprocess.run(['xdg-open', self.video_path])
        else:
            QMessageBox.warning(None, "File Not Found", f"Video file not found:\n{self.video_path}")


class FootballAnalysisUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("⚽ Football Video Analysis Dashboard")
        self.setGeometry(100, 100, 1400, 800)
        
        # Set dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: white;
            }
            QWidget {
                background-color: #2b2b2b;
                color: white;
            }
            QPushButton {
                background-color: #00aaff;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0088cc;
            }
            QPushButton:pressed {
                background-color: #0066aa;
            }
            QPushButton:disabled {
                background-color: #666;
                color: #999;
            }
            QTextEdit {
                background-color: #1e1e1e;
                border: 1px solid #555;
                border-radius: 5px;
                padding: 8px;
                font-family: 'Consolas', 'Monaco', monospace;
            }
            QProgressBar {
                border: 1px solid #555;
                border-radius: 5px;
                text-align: center;
                background-color: #3e3e3e;
            }
            QProgressBar::chunk {
                background-color: #00aaff;
                border-radius: 3px;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555;
                border-radius: 8px;
                margin: 10px 0;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QScrollArea {
                border: none;
                background-color: #2b2b2b;
            }
        """)
        
        self.init_ui()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout()
        
        # Header
        header_label = QLabel("⚽ Football Video Analysis Dashboard")
        header_label.setStyleSheet("font-size: 24px; font-weight: bold; padding: 20px; color: #00aaff;")
        header_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header_label)
        
        # Create splitter for better layout
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        
        # File selection group
        file_group = QGroupBox("📁 Video Selection")
        file_layout = QVBoxLayout()
        
        self.file_label = QLabel("No video selected")
        self.file_label.setStyleSheet("padding: 10px; background-color: #3e3e3e; border-radius: 5px; border: 1px solid #555;")
        
        select_btn = QPushButton("🎬 Select Video File")
        select_btn.clicked.connect(self.select_video)
        
        self.analyze_btn = QPushButton("🔍 Start Analysis")
        self.analyze_btn.clicked.connect(self.start_analysis)
        self.analyze_btn.setEnabled(False)
        
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(select_btn)
        file_layout.addWidget(self.analyze_btn)
        file_group.setLayout(file_layout)
        
        # Progress group
        progress_group = QGroupBox("📊 Analysis Progress")
        progress_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        self.log_text.append("🎯 Ready to analyze football videos!")
        self.log_text.append("📝 Select a video file to get started.")
        
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.log_text)
        progress_group.setLayout(progress_layout)
        
        left_layout.addWidget(file_group)
        left_layout.addWidget(progress_group)
        left_layout.addStretch()
        left_panel.setLayout(left_layout)
        
        # Right panel - Results
        self.results_panel = QWidget()
        results_layout = QVBoxLayout()
        
        results_label = QLabel("📊 Analysis Results")
        results_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #00aaff; padding: 10px;")
        results_layout.addWidget(results_label)
        
        # Scroll area for results
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        self.results_content = QWidget()
        self.results_layout = QVBoxLayout()
        self.results_content.setLayout(self.results_layout)
        scroll_area.setWidget(self.results_content)
        
        results_layout.addWidget(scroll_area)
        self.results_panel.setLayout(results_layout)
        
        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(self.results_panel)
        splitter.setSizes([400, 1000])  # Left panel smaller
        
        main_layout.addWidget(splitter)
        central_widget.setLayout(main_layout)
        
        self.selected_video = None
        
        # Check for existing results on startup
        QTimer.singleShot(1000, self.check_existing_results)
    
    def select_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", 
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv);;All Files (*)"
        )
        
        if file_path:
            self.selected_video = file_path
            self.file_label.setText(f"Selected: {os.path.basename(file_path)}")
            self.analyze_btn.setEnabled(True)
            self.log_text.append(f"✅ Video selected: {os.path.basename(file_path)}")
    
    def start_analysis(self):
        if not self.selected_video:
            return
        
        # Disable UI during analysis
        self.analyze_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        
        # Clear previous logs
        self.log_text.clear()
        self.log_text.append("🚀 Starting analysis...")
        
        # Start analysis in worker thread
        self.worker = AnalysisWorker(self.selected_video)
        self.worker.progress_update.connect(self.update_progress)
        self.worker.finished.connect(self.analysis_finished)
        self.worker.start()
    
    def update_progress(self, message):
        self.log_text.append(message)
        self.log_text.ensureCursorVisible()
    
    def analysis_finished(self, success, message):
        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True)
        
        if success:
            self.log_text.append(f"✅ {message}")
            self.load_results()
        else:
            self.log_text.append(f"❌ {message}")
            QMessageBox.critical(self, "Analysis Error", message)
    
    def load_results(self):
        # Clear previous results
        for i in reversed(range(self.results_layout.count())):
            self.results_layout.itemAt(i).widget().setParent(None)
        
        # Videos section
        videos_group = QGroupBox("🎥 Generated Videos")
        videos_layout = QVBoxLayout()
        
        video_files = [
            ('output_videos/output_video.avi', 'Main Analysis Video'),
            ('output_videos/heatmap_overlay.avi', 'Heatmap Overlay Video')
        ]
        
        for video_path, title in video_files:
            video_card = VideoCard(video_path, title)
            videos_layout.addWidget(video_card)
        
        videos_group.setLayout(videos_layout)
        self.results_layout.addWidget(videos_group)
        
        # Heatmaps section
        heatmaps_group = QGroupBox("🗺️ Generated Heatmaps")
        heatmaps_layout = QVBoxLayout()
        
        # Individual heatmaps
        individual_label = QLabel("Individual Player Heatmaps:")
        individual_label.setStyleSheet("font-weight: bold; margin: 10px 0;")
        heatmaps_layout.addWidget(individual_label)
        
        individual_grid = QGridLayout()
        individual_widget = QWidget()
        individual_widget.setLayout(individual_grid)
        
        individual_dir = 'output_heatmaps/individual'
        if os.path.exists(individual_dir):
            files = [f for f in os.listdir(individual_dir) if f.endswith(('.png', '.jpg'))]
            for i, filename in enumerate(files):
                image_path = os.path.join(individual_dir, filename)
                title = filename.replace('.png', '').replace('_', ' ').title()
                card = ImageCard(image_path, title)
                individual_grid.addWidget(card, i // 3, i % 3)
        
        heatmaps_layout.addWidget(individual_widget)
        
        # Team heatmaps
        team_label = QLabel("Team Heatmaps:")
        team_label.setStyleSheet("font-weight: bold; margin: 10px 0;")
        heatmaps_layout.addWidget(team_label)
        
        team_grid = QGridLayout()
        team_widget = QWidget()
        team_widget.setLayout(team_grid)
        
        team_dirs = ['output_heatmaps/teams', 'output_heatmaps/combined']
        col = 0
        for team_dir in team_dirs:
            if os.path.exists(team_dir):
                files = [f for f in os.listdir(team_dir) if f.endswith(('.png', '.jpg'))]
                for filename in files:
                    image_path = os.path.join(team_dir, filename)
                    title = filename.replace('.png', '').replace('_', ' ').title()
                    card = ImageCard(image_path, title)
                    team_grid.addWidget(card, 0, col)
                    col += 1
        
        heatmaps_layout.addWidget(team_widget)
        heatmaps_group.setLayout(heatmaps_layout)
        self.results_layout.addWidget(heatmaps_group)
        
        self.results_layout.addStretch()
    
    def check_existing_results(self):
        """Check if there are existing results to display"""
        if (os.path.exists('output_videos') and 
            (os.path.exists('output_videos/output_video.avi') or 
             os.path.exists('output_videos/heatmap_overlay.avi'))):
            self.load_results()


def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Set dark palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(43, 43, 43))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(30, 30, 30))
    palette.setColor(QPalette.AlternateBase, QColor(60, 60, 60))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(60, 60, 60))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)
    
    window = FootballAnalysisUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()