import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from collections import defaultdict
import os

class HeatMapGenerator:
    def __init__(self, field_width=1920, field_height=1080):
        self.field_width = field_width
        self.field_height = field_height
        self.heatmap_resolution = (108, 68)  # Standard football field proportions
        self.player_positions = defaultdict(list)
        self.team_positions = {1: [], 2: []}
        
    def collect_positions_from_tracks(self, tracks):
        """Collect all player positions from tracking data"""
        for frame_num, frame_tracks in enumerate(tracks['players']):
            for player_id, player_data in frame_tracks.items():
                if 'position_transformed' in player_data and player_data['position_transformed'] is not None:
                    # Use transformed positions (real-world coordinates)
                    position = player_data['position_transformed']
                    team = player_data.get('team', 1)
                    
                    self.player_positions[player_id].append(position)
                    self.team_positions[team].append(position)
                elif 'position' in player_data:
                    # Fallback to pixel positions
                    position = player_data['position']
                    # Normalize to field coordinates
                    norm_x = (position[0] / self.field_width) * self.heatmap_resolution[0]
                    norm_y = (position[1] / self.field_height) * self.heatmap_resolution[1]
                    team = player_data.get('team', 1)
                    
                    self.player_positions[player_id].append([norm_x, norm_y])
                    self.team_positions[team].append([norm_x, norm_y])

    def generate_individual_heatmaps(self, tracks, output_dir='heatmaps'):
        """Generate individual heat maps for each player"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        self.collect_positions_from_tracks(tracks)
        
        for player_id, positions in self.player_positions.items():
            if len(positions) < 10:  # Skip players with too few positions
                continue
                
            # Create 2D histogram
            positions = np.array(positions)
            heatmap, xedges, yedges = np.histogram2d(
                positions[:, 0], positions[:, 1], 
                bins=self.heatmap_resolution,
                range=[[0, self.heatmap_resolution[0]], [0, self.heatmap_resolution[1]]]
            )
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            plt.imshow(heatmap.T, origin='lower', cmap='hot', interpolation='gaussian')
            plt.colorbar(label='Frequency')
            plt.title(f'Player {player_id} Heat Map')
            plt.xlabel('Field Width')
            plt.ylabel('Field Length')
            
            # Add field markings
            self._add_field_markings(plt)
            
            plt.savefig(f'{output_dir}/player_{player_id}_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()

    def generate_team_heatmaps(self, tracks, output_dir='heatmaps'):
        """Generate heat maps for each team"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        self.collect_positions_from_tracks(tracks)
        
        colors = ['Blues', 'Reds']
        team_names = ['Team 1', 'Team 2']
        
        for team_id in [1, 2]:
            if len(self.team_positions[team_id]) < 10:
                continue
                
            positions = np.array(self.team_positions[team_id])
            heatmap, xedges, yedges = np.histogram2d(
                positions[:, 0], positions[:, 1],
                bins=self.heatmap_resolution,
                range=[[0, self.heatmap_resolution[0]], [0, self.heatmap_resolution[1]]]
            )
            
            plt.figure(figsize=(12, 8))
            plt.imshow(heatmap.T, origin='lower', cmap=colors[team_id-1], interpolation='gaussian')
            plt.colorbar(label='Player Density')
            plt.title(f'{team_names[team_id-1]} Heat Map')
            plt.xlabel('Field Width')
            plt.ylabel('Field Length')
            
            self._add_field_markings(plt)
            
            plt.savefig(f'{output_dir}/team_{team_id}_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()

    def generate_combined_team_heatmap(self, tracks, output_dir='heatmaps'):
        """Generate a combined heat map showing both teams"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        self.collect_positions_from_tracks(tracks)
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Team 1 heatmap
        if len(self.team_positions[1]) > 10:
            positions1 = np.array(self.team_positions[1])
            heatmap1, _, _ = np.histogram2d(
                positions1[:, 0], positions1[:, 1],
                bins=self.heatmap_resolution,
                range=[[0, self.heatmap_resolution[0]], [0, self.heatmap_resolution[1]]]
            )
            im1 = ax1.imshow(heatmap1.T, origin='lower', cmap='Blues', interpolation='gaussian')
            ax1.set_title('Team 1 Heat Map')
            plt.colorbar(im1, ax=ax1, label='Density')
            self._add_field_markings_ax(ax1)
        
        # Team 2 heatmap
        if len(self.team_positions[2]) > 10:
            positions2 = np.array(self.team_positions[2])
            heatmap2, _, _ = np.histogram2d(
                positions2[:, 0], positions2[:, 1],
                bins=self.heatmap_resolution,
                range=[[0, self.heatmap_resolution[0]], [0, self.heatmap_resolution[1]]]
            )
            im2 = ax2.imshow(heatmap2.T, origin='lower', cmap='Reds', interpolation='gaussian')
            ax2.set_title('Team 2 Heat Map')
            plt.colorbar(im2, ax=ax2, label='Density')
            self._add_field_markings_ax(ax2)
        
        # Combined heatmap
        if len(self.team_positions[1]) > 10 and len(self.team_positions[2]) > 10:
            # Create RGB image for combined view
            combined = np.zeros((self.heatmap_resolution[1], self.heatmap_resolution[0], 3))
            combined[:, :, 0] = heatmap2.T / heatmap2.max() if heatmap2.max() > 0 else 0  # Red for team 2
            combined[:, :, 2] = heatmap1.T / heatmap1.max() if heatmap1.max() > 0 else 0  # Blue for team 1
            
            ax3.imshow(combined, origin='lower')
            ax3.set_title('Combined Heat Map\n(Blue: Team 1, Red: Team 2)')
            self._add_field_markings_ax(ax3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/combined_team_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_video_overlay_heatmap(self, tracks, frames, output_path, fade_frames=300):
        """Generate video with real-time heat map overlay"""
        self.collect_positions_from_tracks(tracks)
        
        # Initialize heat map accumulator
        heatmap_accumulator = np.zeros((self.heatmap_resolution[1], 
                                self.heatmap_resolution[0]))

        output_frames = []
        
        for frame_num, frame in enumerate(frames):
            frame_copy = frame.copy()
            
            # Add current frame positions to accumulator
            if frame_num < len(tracks['players']):
                for player_id, player_data in tracks['players'][frame_num].items():
                    if 'position' in player_data:
                        pos = player_data['position']
                        # Convert to heatmap coordinates
                        hmap_x = int((pos[0] / self.field_width) * self.heatmap_resolution[0])
                        hmap_y = int((pos[1] / self.field_height) * self.heatmap_resolution[1])
                        
                        if 0 <= hmap_x < self.heatmap_resolution[0] and 0 <= hmap_y < self.heatmap_resolution[1]:
                            heatmap_accumulator[hmap_y, hmap_x] += 1
            
            # Apply fade effect
            if frame_num > fade_frames:
                heatmap_accumulator *= 0.995
            
            # Create heatmap overlay
            if np.max(heatmap_accumulator) > 0:
                heatmap_normalized = heatmap_accumulator / np.max(heatmap_accumulator)
                heatmap_colored = cv2.applyColorMap(
                    (heatmap_normalized * 255).astype(np.uint8), 
                    cv2.COLORMAP_JET
                )
                
                # Resize heatmap to frame size
                heatmap_resized = cv2.resize(heatmap_colored, (frame.shape[1], frame.shape[0]))
                
                # Blend with original frame
                alpha = 0.3
                frame_copy = cv2.addWeighted(frame_copy, 1-alpha, heatmap_resized, alpha, 0)
            
            output_frames.append(frame_copy)
        
        return output_frames

    def _add_field_markings(self, plt_obj):
        """Add football field markings to the plot"""
        # Center circle
        circle = plt.Circle((self.heatmap_resolution[0]/2, self.heatmap_resolution[1]/2), 
                           self.heatmap_resolution[0]/10, fill=False, color='white', linewidth=2)
        plt_obj.gca().add_patch(circle)
        
        # Center line
        plt_obj.axhline(y=self.heatmap_resolution[1]/2, color='white', linewidth=2)
        
        # Penalty areas
        penalty_width = self.heatmap_resolution[0] * 0.3
        penalty_height = self.heatmap_resolution[1] * 0.15
        
        # Left penalty area
        left_penalty = plt.Rectangle((0, self.heatmap_resolution[1]/2 - penalty_height/2), 
                                   penalty_width, penalty_height, 
                                   fill=False, color='white', linewidth=2)
        plt_obj.gca().add_patch(left_penalty)
        
        # Right penalty area
        right_penalty = plt.Rectangle((self.heatmap_resolution[0] - penalty_width, 
                                     self.heatmap_resolution[1]/2 - penalty_height/2), 
                                    penalty_width, penalty_height, 
                                    fill=False, color='white', linewidth=2)
        plt_obj.gca().add_patch(right_penalty)

    def _add_field_markings_ax(self, ax):
        """Add football field markings to a specific axis"""
        # Center circle
        circle = plt.Circle((self.heatmap_resolution[0]/2, self.heatmap_resolution[1]/2), 
                           self.heatmap_resolution[0]/10, fill=False, color='white', linewidth=2)
        ax.add_patch(circle)
        
        # Center line
        ax.axhline(y=self.heatmap_resolution[1]/2, color='white', linewidth=2)
        
        # Penalty areas
        penalty_width = self.heatmap_resolution[0] * 0.3
        penalty_height = self.heatmap_resolution[1] * 0.15
        
        # Left penalty area
        left_penalty = plt.Rectangle((0, self.heatmap_resolution[1]/2 - penalty_height/2), 
                                   penalty_width, penalty_height, 
                                   fill=False, color='white', linewidth=2)
        ax.add_patch(left_penalty)
        
        # Right penalty area
        right_penalty = plt.Rectangle((self.heatmap_resolution[0] - penalty_width, 
                                     self.heatmap_resolution[1]/2 - penalty_height/2), 
                                    penalty_width, penalty_height, 
                                    fill=False, color='white', linewidth=2)
        ax.add_patch(right_penalty)