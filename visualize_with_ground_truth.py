#!/usr/bin/env python3
"""
Comprehensive Beach Crowd Detection Pipeline Visualizer with Ground Truth

This script visualizes all detection pipeline steps AND compares results
against ground truth annotations to find the optimal blob level for each image.

Features:
- Step-by-step visualization of all transformations
- Ground truth overlay on all detection visualizations
- MAE calculation for each blob level
- Automatic selection of optimal level per image
- Detailed comparison charts

Usage:
    python visualize_with_ground_truth.py -i images -a coordinates.csv
    python visualize_with_ground_truth.py -i images -a coordinates.csv --all-steps
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def adjust_gamma(image, gamma=0.4):
    """Apply gamma correction."""
    img = image.astype(np.float32) / 255.0
    img = np.power(img, gamma)
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def get_blob_params_level(level):
    """Get blob detection parameters by accuracy level (1-10)."""
    blob_params = {
        1: {'min_area': 30, 'max_area': 60000, 'min_circularity': 0.02, 'min_convexity': 0.15, 'min_inertia': 0.005},
        2: {'min_area': 80, 'max_area': 50000, 'min_circularity': 0.05, 'min_convexity': 0.20, 'min_inertia': 0.01},
        3: {'min_area': 150, 'max_area': 40000, 'min_circularity': 0.08, 'min_convexity': 0.28, 'min_inertia': 0.02},
        4: {'min_area': 220, 'max_area': 32000, 'min_circularity': 0.12, 'min_convexity': 0.35, 'min_inertia': 0.035},
        5: {'min_area': 300, 'max_area': 25000, 'min_circularity': 0.15, 'min_convexity': 0.42, 'min_inertia': 0.05},
        6: {'min_area': 380, 'max_area': 20000, 'min_circularity': 0.20, 'min_convexity': 0.48, 'min_inertia': 0.07},
        7: {'min_area': 450, 'max_area': 17000, 'min_circularity': 0.25, 'min_convexity': 0.55, 'min_inertia': 0.09},
        8: {'min_area': 520, 'max_area': 14000, 'min_circularity': 0.30, 'min_convexity': 0.62, 'min_inertia': 0.11},
        9: {'min_area': 600, 'max_area': 12000, 'min_circularity': 0.38, 'min_convexity': 0.70, 'min_inertia': 0.14},
        10: {'min_area': 700, 'max_area': 10000, 'min_circularity': 0.45, 'min_convexity': 0.78, 'min_inertia': 0.18}
    }
    return blob_params.get(level, blob_params[5])


def get_level_name(level):
    """Get human-readable name for each level."""
    names = {
        1: 'Ultra Loose', 2: 'Very Loose', 3: 'Loose', 4: 'Mod. Loose',
        5: 'Balanced', 6: 'Mod. Strict', 7: 'Strict', 8: 'Very Strict',
        9: 'Extra Strict', 10: 'Ultra Strict'
    }
    return names.get(level, 'Balanced')


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class LevelResult:
    """Result of testing a single level on an image."""
    level: int
    level_name: str
    detected_count: int
    ground_truth_count: int
    mae: int
    error_percent: float
    detected_points: np.ndarray


@dataclass
class ImageResult:
    """Complete result for a single image across all levels."""
    image_name: str
    ground_truth_count: int
    ground_truth_points: np.ndarray
    best_level: int
    best_level_name: str
    best_mae: int
    best_detected: int
    all_levels: List[LevelResult]


# =============================================================================
# MAIN PIPELINE VISUALIZER CLASS
# =============================================================================

class PipelineVisualizerWithGroundTruth:
    """
    Comprehensive pipeline visualizer that shows all steps and compares
    detection results against ground truth annotations.
    """
    
    def __init__(self, images_dir='images', annotations_path='coordinates.csv',
                 output_dir='pipeline_gt_outputs'):
        self.images_dir = images_dir
        self.annotations_path = annotations_path
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Default parameters
        self.params = {
            'clahe_clip': 2.0,
            'gamma': 0.4,
            'gaussian_size': 5,
            'top_mask_percent': 0.45,
            'hsv_s_max': 50,
            'hsv_v_min': 100,
            'morph_size': 5,
            'adaptive_block_size': 11,
            'adaptive_c': 2
        }
        
        # Load annotations
        self.annotations = self._load_annotations()
    
    def _load_annotations(self) -> pd.DataFrame:
        """Load ground truth annotations from CSV."""
        if not os.path.exists(self.annotations_path):
            print(f"Warning: Annotations file not found: {self.annotations_path}")
            print("Will run without ground truth comparisons.")
            return pd.DataFrame()
        
        # Try different separators
        for sep in [';', ',', '\t']:
            try:
                df = pd.read_csv(self.annotations_path, sep=sep)
                if 'file' in df.columns and 'x' in df.columns and 'y' in df.columns:
                    print(f"Loaded {len(df)} annotations from {self.annotations_path}")
                    return df
            except:
                continue
        
        print(f"Warning: Could not parse annotations file: {self.annotations_path}")
        return pd.DataFrame()
    
    def get_image_annotations(self, image_name: str) -> np.ndarray:
        """Get ground truth coordinates for an image."""
        if self.annotations.empty:
            return np.array([])
        
        img_annotations = self.annotations[self.annotations['file'] == image_name]
        if len(img_annotations) > 0:
            return img_annotations[['x', 'y']].values
        return np.array([])
    
    # =========================================================================
    # PIPELINE STEPS
    # =========================================================================
    
    def step1_load_image(self, image_path):
        """Step 1: Load the original image."""
        return cv2.imread(image_path)
    
    def step2_clahe_enhancement(self, image):
        """Step 2: Apply CLAHE enhancement."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=self.params['clahe_clip'], tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        
        lab_clahe = cv2.merge([l_clahe, a, b])
        enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        
        return enhanced, l, l_clahe
    
    def step3_gamma_correction(self, image):
        """Step 3: Apply gamma correction."""
        return adjust_gamma(image, gamma=self.params['gamma'])
    
    def step4_gaussian_blur(self, image):
        """Step 4: Apply Gaussian blur."""
        blur_size = self.params['gaussian_size']
        if blur_size % 2 == 0:
            blur_size += 1
        return cv2.GaussianBlur(image, (blur_size, blur_size), 0)
    
    def step5_spatial_mask(self, image):
        """Step 5: Create and apply spatial mask."""
        h, w = image.shape[:2]
        mask = np.ones((h, w), dtype=np.uint8) * 255
        exclude_h = int(h * self.params['top_mask_percent'])
        mask[:exclude_h, :] = 0
        
        masked = image.copy()
        masked[mask == 0] = 0
        
        return masked, mask
    
    def step6_hsv_sand_filtering(self, image):
        """Step 6: HSV color space filtering."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        lower_sand = np.array([0, 0, self.params['hsv_v_min']])
        upper_sand = np.array([180, self.params['hsv_s_max'], 255])
        sand_mask = cv2.inRange(hsv, lower_sand, upper_sand)
        non_sand_mask = cv2.bitwise_not(sand_mask)
        
        return hsv, sand_mask, non_sand_mask
    
    def step7_morphological_ops(self, mask):
        """Step 7: Apply morphological operations."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                           (self.params['morph_size'], self.params['morph_size']))
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        return opened, closed
    
    def step8_adaptive_threshold(self, image, mask):
        """Step 8: Apply adaptive thresholding."""
        filtered = cv2.bitwise_and(image, image, mask=mask)
        gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
        
        block_size = self.params['adaptive_block_size']
        if block_size % 2 == 0:
            block_size += 1
        
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, block_size,
                                       self.params['adaptive_c'])
        return filtered, gray, thresh
    
    def step9_blob_detection(self, thresh_image, level):
        """Step 9: Blob detection with specified level parameters."""
        blob_params = get_blob_params_level(level)
        
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = blob_params['min_area']
        params.maxArea = blob_params['max_area']
        params.filterByCircularity = True
        params.minCircularity = blob_params['min_circularity']
        params.filterByConvexity = True
        params.minConvexity = blob_params['min_convexity']
        params.filterByInertia = True
        params.minInertiaRatio = blob_params['min_inertia']
        params.filterByColor = False
        
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(thresh_image)
        points = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints]) if keypoints else np.array([])
        
        return keypoints, points, len(keypoints)
    
    # =========================================================================
    # VISUALIZATION METHODS
    # =========================================================================
    
    def draw_annotations(self, image, ground_truth, detected_points, 
                         show_gt=True, show_detected=True):
        """Draw ground truth and detected points on image."""
        img = image.copy()
        
        # Draw ground truth (red circles)
        if show_gt and len(ground_truth) > 0:
            for x, y in ground_truth:
                cv2.circle(img, (int(x), int(y)), 12, (0, 0, 255), 2)  # Red outer
                cv2.circle(img, (int(x), int(y)), 4, (0, 0, 255), -1)  # Red center
        
        # Draw detected points (green circles)
        if show_detected and len(detected_points) > 0:
            for x, y in detected_points:
                cv2.circle(img, (int(x), int(y)), 10, (0, 255, 0), 2)  # Green outer
                cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)  # Green center
        
        return img
    
    def visualize_preprocessing_steps(self, image_path, ground_truth):
        """Generate visualization of preprocessing steps (1-5)."""
        image_name = Path(image_path).stem
        
        # Execute steps
        original = self.step1_load_image(image_path)
        clahe_enhanced, l_original, l_clahe = self.step2_clahe_enhancement(original)
        gamma_corrected = self.step3_gamma_correction(clahe_enhanced)
        blurred = self.step4_gaussian_blur(gamma_corrected)
        masked, spatial_mask = self.step5_spatial_mask(blurred)
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Pipeline Steps 1-5: Preprocessing\n{image_name} | Ground Truth: {len(ground_truth)} people', 
                     fontsize=16, fontweight='bold')
        
        # Step 1: Original with ground truth overlay
        img_with_gt = self.draw_annotations(original, ground_truth, [], show_detected=False)
        axes[0, 0].imshow(cv2.cvtColor(img_with_gt, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title(f'Step 1: Original Image\n(Red = Ground Truth: {len(ground_truth)})',
                            fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Step 2: CLAHE
        axes[0, 1].imshow(cv2.cvtColor(clahe_enhanced, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title(f'Step 2: CLAHE Enhanced\n(clip={self.params["clahe_clip"]})', 
                            fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Step 2b: L channel comparison
        l_comparison = np.hstack([l_original, l_clahe])
        axes[0, 2].imshow(l_comparison, cmap='gray')
        axes[0, 2].set_title('Step 2b: L Channel (Before | After CLAHE)', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Step 3: Gamma
        axes[1, 0].imshow(cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title(f'Step 3: Gamma Correction\n(γ={self.params["gamma"]})', 
                            fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Step 4: Blur
        axes[1, 1].imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title(f'Step 4: Gaussian Blur\n(size={self.params["gaussian_size"]})', 
                            fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        
        # Step 5: Spatial Mask with GT overlay
        masked_with_gt = self.draw_annotations(masked, ground_truth, [], show_detected=False)
        axes[1, 2].imshow(cv2.cvtColor(masked_with_gt, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title(f'Step 5: Spatial Mask Applied\n(top {int(self.params["top_mask_percent"]*100)}% excluded)', 
                            fontsize=12, fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        fig_path = os.path.join(self.output_dir, f'{image_name}_step1_preprocessing.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return masked, fig_path
    
    def visualize_segmentation_steps(self, image_path, masked_image, ground_truth):
        """Generate visualization of segmentation steps (6-8)."""
        image_name = Path(image_path).stem
        
        # Execute steps
        hsv, sand_mask, non_sand_mask = self.step6_hsv_sand_filtering(masked_image)
        morph_opened, morph_closed = self.step7_morphological_ops(non_sand_mask)
        filtered, gray, thresh = self.step8_adaptive_threshold(masked_image, morph_closed)
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Pipeline Steps 6-8: Segmentation\n{image_name} | Ground Truth: {len(ground_truth)} people', 
                     fontsize=16, fontweight='bold')
        
        # Step 6a: HSV
        axes[0, 0].imshow(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
        axes[0, 0].set_title('Step 6a: HSV Color Space', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Step 6b: Sand mask
        axes[0, 1].imshow(sand_mask, cmap='gray')
        axes[0, 1].set_title(f'Step 6b: Sand Mask\n(S<{self.params["hsv_s_max"]}, V>{self.params["hsv_v_min"]})', 
                            fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Step 6c: Non-sand mask with GT points overlay
        non_sand_rgb = cv2.cvtColor(non_sand_mask, cv2.COLOR_GRAY2BGR)
        if len(ground_truth) > 0:
            for x, y in ground_truth:
                cv2.circle(non_sand_rgb, (int(x), int(y)), 8, (0, 0, 255), -1)
        axes[0, 2].imshow(cv2.cvtColor(non_sand_rgb, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title('Step 6c: Non-Sand Mask\n(Red dots = Ground Truth)', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Step 7a: Morphological opening
        axes[1, 0].imshow(morph_opened, cmap='gray')
        axes[1, 0].set_title(f'Step 7a: Morphological Opening\n(kernel={self.params["morph_size"]})', 
                            fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Step 7b: Morphological closing
        axes[1, 1].imshow(morph_closed, cmap='gray')
        axes[1, 1].set_title('Step 7b: Morphological Closing', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        
        # Step 8: Adaptive threshold with GT points overlay
        thresh_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        if len(ground_truth) > 0:
            for x, y in ground_truth:
                cv2.circle(thresh_rgb, (int(x), int(y)), 8, (0, 0, 255), -1)
        axes[1, 2].imshow(cv2.cvtColor(thresh_rgb, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title(f'Step 8: Adaptive Threshold\n(Red dots = Ground Truth)', 
                            fontsize=12, fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        fig_path = os.path.join(self.output_dir, f'{image_name}_step2_segmentation.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return thresh, fig_path
    
    def visualize_blob_levels_comparison(self, image_path, thresh_image, ground_truth):
        """Generate visualization comparing all 10 blob levels with ground truth."""
        image_name = Path(image_path).stem
        original = cv2.imread(image_path)
        gt_count = len(ground_truth)
        
        # Test all levels
        level_results = []
        
        for level in range(1, 11):
            keypoints, points, count = self.step9_blob_detection(thresh_image, level)
            mae = abs(count - gt_count)
            error_pct = (mae / gt_count * 100) if gt_count > 0 else 0
            
            level_results.append(LevelResult(
                level=level,
                level_name=get_level_name(level),
                detected_count=count,
                ground_truth_count=gt_count,
                mae=mae,
                error_percent=error_pct,
                detected_points=points
            ))
        
        # Find best level (minimum MAE)
        best_result = min(level_results, key=lambda r: (r.mae, abs(r.level - 5)))
        
        # Create figure - 2 rows of 5
        fig, axes = plt.subplots(2, 5, figsize=(25, 10))
        fig.suptitle(f'Step 9: Blob Detection Comparison (All 10 Levels)\n{image_name} | '
                     f'Ground Truth: {gt_count} people | Best Level: {best_result.level} (MAE={best_result.mae})', 
                     fontsize=16, fontweight='bold')
        
        for i, result in enumerate(level_results):
            row = i // 5
            col = i % 5
            
            # Draw both ground truth and detected points
            img_annotated = self.draw_annotations(original, ground_truth, result.detected_points)
            img_rgb = cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB)
            
            axes[row, col].imshow(img_rgb)
            
            # Highlight best level
            title_color = 'green' if result.level == best_result.level else 'black'
            best_marker = ' ★ BEST' if result.level == best_result.level else ''
            
            title = f'Level {result.level}: {result.level_name}{best_marker}\n'
            title += f'Detected: {result.detected_count} | MAE: {result.mae}'
            
            axes[row, col].set_title(title, fontsize=11, fontweight='bold', color=title_color)
            axes[row, col].axis('off')
            
            # Add border for best level
            if result.level == best_result.level:
                for spine in axes[row, col].spines.values():
                    spine.set_edgecolor('green')
                    spine.set_linewidth(3)
                    spine.set_visible(True)
        
        # Add legend
        red_patch = mpatches.Patch(color='red', label=f'Ground Truth ({gt_count})')
        green_patch = mpatches.Patch(color='green', label='Detected')
        fig.legend(handles=[red_patch, green_patch], loc='lower center', ncol=2, fontsize=12)
        
        plt.tight_layout()
        fig_path = os.path.join(self.output_dir, f'{image_name}_step3_blob_levels.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return level_results, best_result, fig_path
    
    def visualize_mae_analysis(self, image_path, level_results, best_result, ground_truth):
        """Generate MAE analysis chart for all levels."""
        image_name = Path(image_path).stem
        gt_count = len(ground_truth)
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(f'Detection Analysis by Blob Level\n{image_name} | Ground Truth: {gt_count} people', 
                     fontsize=16, fontweight='bold')
        
        levels = [r.level for r in level_results]
        detected_counts = [r.detected_count for r in level_results]
        maes = [r.mae for r in level_results]
        
        # Color map - green for low MAE, red for high MAE
        max_mae = max(maes) if max(maes) > 0 else 1
        colors = plt.cm.RdYlGn_r(np.array(maes) / max_mae)
        
        # Chart 1: Detected vs Ground Truth
        ax1 = axes[0]
        bars1 = ax1.bar(levels, detected_counts, color=colors, alpha=0.8, label='Detected')
        ax1.axhline(y=gt_count, color='red', linestyle='--', linewidth=2, label=f'Ground Truth ({gt_count})')
        ax1.set_xlabel('Blob Level', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax1.set_title('Detected Count vs Ground Truth', fontsize=14, fontweight='bold')
        ax1.set_xticks(levels)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Add count labels
        for bar, count in zip(bars1, detected_counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Chart 2: MAE per Level
        ax2 = axes[1]
        bars2 = ax2.bar(levels, maes, color=colors, alpha=0.8)
        ax2.axhline(y=0, color='green', linestyle='-', linewidth=2)
        ax2.set_xlabel('Blob Level', fontsize=12, fontweight='bold')
        ax2.set_ylabel('MAE (Absolute Error)', fontsize=12, fontweight='bold')
        ax2.set_title('Mean Absolute Error per Level', fontsize=14, fontweight='bold')
        ax2.set_xticks(levels)
        ax2.grid(axis='y', alpha=0.3)
        
        # Highlight best level
        best_idx = best_result.level - 1
        bars2[best_idx].set_edgecolor('green')
        bars2[best_idx].set_linewidth(3)
        
        # Add MAE labels
        for bar, mae in zip(bars2, maes):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    str(mae), ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Chart 3: Parameters table with results
        ax3 = axes[2]
        ax3.axis('off')
        
        table_data = []
        for r in level_results:
            p = get_blob_params_level(r.level)
            row = [
                r.level,
                r.detected_count,
                r.mae,
                f"{r.error_percent:.1f}%",
                p['min_area'],
                f"{p['min_circularity']:.2f}",
                f"{p['min_convexity']:.2f}"
            ]
            table_data.append(row)
        
        table = ax3.table(
            cellText=table_data,
            colLabels=['Level', 'Detected', 'MAE', 'Error%', 'MinArea', 'Circ', 'Conv'],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Highlight best row
        for j in range(7):
            table[(best_idx + 1, j)].set_facecolor('#90EE90')
        
        ax3.set_title('Results & Parameters\n(Green row = Best Level)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        fig_path = os.path.join(self.output_dir, f'{image_name}_step4_analysis.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return fig_path
    
    def visualize_final_comparison(self, image_path, level_results, best_result, ground_truth):
        """Generate final comparison showing ground truth vs best detection."""
        image_name = Path(image_path).stem
        original = cv2.imread(image_path)
        gt_count = len(ground_truth)
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        fig.suptitle(f'Final Comparison: {image_name}\nOptimal Level: {best_result.level} ({best_result.level_name}) | MAE: {best_result.mae}', 
                     fontsize=16, fontweight='bold')
        
        # Panel 1: Ground Truth Only
        img_gt = self.draw_annotations(original, ground_truth, [], show_detected=False)
        axes[0].imshow(cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f'Ground Truth\n{gt_count} people (Red)', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Panel 2: Best Detection Only
        img_det = self.draw_annotations(original, [], best_result.detected_points, show_gt=False)
        axes[1].imshow(cv2.cvtColor(img_det, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f'Best Detection (Level {best_result.level})\n{best_result.detected_count} detected (Green)', 
                         fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # Panel 3: Overlay
        img_overlay = self.draw_annotations(original, ground_truth, best_result.detected_points)
        axes[2].imshow(cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB))
        axes[2].set_title(f'Overlay Comparison\nRed=GT ({gt_count}), Green=Detected ({best_result.detected_count})\nMAE: {best_result.mae}', 
                         fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        fig_path = os.path.join(self.output_dir, f'{image_name}_step5_final.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return fig_path
    
    # =========================================================================
    # MAIN PROCESSING METHODS
    # =========================================================================
    
    def process_single_image(self, image_path, show_all_steps=True):
        """Process a single image through the complete pipeline."""
        image_name = Path(image_path).name
        image_stem = Path(image_path).stem
        
        print(f"\n{'='*60}")
        print(f"Processing: {image_name}")
        print(f"{'='*60}")
        
        # Get ground truth
        ground_truth = self.get_image_annotations(image_name)
        gt_count = len(ground_truth)
        print(f"Ground Truth: {gt_count} people")
        
        # Step 1-5: Preprocessing
        print("\nSteps 1-5: Preprocessing...")
        masked, prep_path = self.visualize_preprocessing_steps(image_path, ground_truth)
        print(f"  Saved: {prep_path}")
        
        # Step 6-8: Segmentation
        print("Steps 6-8: Segmentation...")
        thresh, seg_path = self.visualize_segmentation_steps(image_path, masked, ground_truth)
        print(f"  Saved: {seg_path}")
        
        # Step 9: Blob detection at all levels
        print("Step 9: Testing all 10 blob levels...")
        level_results, best_result, blob_path = self.visualize_blob_levels_comparison(
            image_path, thresh, ground_truth)
        print(f"  Saved: {blob_path}")
        
        # Analysis chart
        print("Generating analysis chart...")
        analysis_path = self.visualize_mae_analysis(image_path, level_results, best_result, ground_truth)
        print(f"  Saved: {analysis_path}")
        
        # Final comparison
        print("Generating final comparison...")
        final_path = self.visualize_final_comparison(image_path, level_results, best_result, ground_truth)
        print(f"  Saved: {final_path}")
        
        # Print level results
        print(f"\n  {'Level':<6} {'Name':<14} {'Detected':>10} {'MAE':>6} {'Error%':>8}")
        print(f"  {'-'*50}")
        for r in level_results:
            marker = " ← BEST" if r.level == best_result.level else ""
            print(f"  {r.level:<6} {r.level_name:<14} {r.detected_count:>10} {r.mae:>6} {r.error_percent:>7.1f}%{marker}")
        
        return ImageResult(
            image_name=image_name,
            ground_truth_count=gt_count,
            ground_truth_points=ground_truth,
            best_level=best_result.level,
            best_level_name=best_result.level_name,
            best_mae=best_result.mae,
            best_detected=best_result.detected_count,
            all_levels=level_results
        )
    
    def process_all_images(self, show_all_steps=True):
        """Process all images in the directory."""
        image_files = sorted([f for f in os.listdir(self.images_dir)
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        if not image_files:
            print(f"No images found in {self.images_dir}")
            return []
        
        print(f"\n{'='*60}")
        print(f"BEACH CROWD DETECTION PIPELINE WITH GROUND TRUTH")
        print(f"{'='*60}")
        print(f"Images directory: {self.images_dir}")
        print(f"Annotations file: {self.annotations_path}")
        print(f"Output directory: {self.output_dir}")
        print(f"Found {len(image_files)} images to process")
        
        all_results = []
        
        for image_file in image_files:
            image_path = os.path.join(self.images_dir, image_file)
            result = self.process_single_image(image_path, show_all_steps)
            all_results.append(result)
        
        # Generate summary
        self._generate_summary(all_results)
        
        return all_results
    
    def _generate_summary(self, results: List[ImageResult]):
        """Generate summary report and comparison chart."""
        if not results:
            return
        
        print(f"\n{'='*60}")
        print("OPTIMIZATION SUMMARY")
        print(f"{'='*60}")
        
        # Summary table
        print(f"\n{'Image':<20} {'GT':>6} {'Best Lvl':>10} {'Detected':>10} {'MAE':>6} {'Error%':>8}")
        print("-"*65)
        
        total_gt = 0
        total_detected = 0
        total_mae = 0
        
        for r in results:
            print(f"{r.image_name:<20} {r.ground_truth_count:>6} "
                  f"{r.best_level:>3} ({r.best_level_name[:5]:>5}) "
                  f"{r.best_detected:>10} {r.best_mae:>6} "
                  f"{(r.best_mae/r.ground_truth_count*100 if r.ground_truth_count > 0 else 0):>7.1f}%")
            total_gt += r.ground_truth_count
            total_detected += r.best_detected
            total_mae += r.best_mae
        
        print("-"*65)
        avg_mae = total_mae / len(results)
        print(f"{'TOTAL':<20} {total_gt:>6} {'':>10} {total_detected:>10} {total_mae:>6}")
        print(f"{'AVERAGE MAE':<20} {avg_mae:>6.2f}")
        
        # Create comparison chart
        self._create_summary_chart(results)
        
        # Save results to JSON
        self._save_results_json(results)
    
    def _create_summary_chart(self, results: List[ImageResult]):
        """Create summary comparison chart across all images."""
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Summary: All Images Analysis', fontsize=16, fontweight='bold')
        
        # Chart 1: MAE per level per image (line chart)
        ax1 = axes[0, 0]
        cmap = plt.cm.tab10
        for i, r in enumerate(results):
            levels = [lr.level for lr in r.all_levels]
            maes = [lr.mae for lr in r.all_levels]
            label = r.image_name[:15]
            ax1.plot(levels, maes, 'o-', color=cmap(i), label=label, linewidth=2, markersize=8)
            
            # Mark best level
            ax1.scatter([r.best_level], [r.best_mae], color=cmap(i), s=200, marker='*', 
                       edgecolors='black', linewidths=1, zorder=5)
        
        ax1.set_xlabel('Blob Level', fontsize=12, fontweight='bold')
        ax1.set_ylabel('MAE', fontsize=12, fontweight='bold')
        ax1.set_title('MAE vs Blob Level per Image (★ = optimal)', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(1, 11))
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Chart 2: Best level distribution
        ax2 = axes[0, 1]
        best_levels = [r.best_level for r in results]
        level_counts = {l: best_levels.count(l) for l in range(1, 11)}
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, 10))
        
        bars = ax2.bar(level_counts.keys(), level_counts.values(), 
                      color=[colors[l-1] for l in level_counts.keys()])
        ax2.set_xlabel('Blob Level', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
        ax2.set_title('Best Level Distribution', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(1, 11))
        ax2.grid(axis='y', alpha=0.3)
        
        # Chart 3: Heatmap of MAE
        ax3 = axes[1, 0]
        mae_data = np.array([[lr.mae for lr in r.all_levels] for r in results])
        im = ax3.imshow(mae_data, aspect='auto', cmap='RdYlGn_r')
        ax3.set_xlabel('Blob Level', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Image', fontsize=12, fontweight='bold')
        ax3.set_title('MAE Heatmap (darker green = lower MAE)', fontsize=14, fontweight='bold')
        ax3.set_xticks(range(10))
        ax3.set_xticklabels(range(1, 11))
        ax3.set_yticks(range(len(results)))
        ax3.set_yticklabels([r.image_name[:12] for r in results])
        
        # Add MAE values
        for i in range(len(results)):
            for j in range(10):
                ax3.text(j, i, str(mae_data[i, j]), ha='center', va='center', 
                        fontsize=9, fontweight='bold')
        
        plt.colorbar(im, ax=ax3, label='MAE')
        
        # Chart 4: GT vs Detected comparison
        ax4 = axes[1, 1]
        x = range(len(results))
        width = 0.35
        
        gt_counts = [r.ground_truth_count for r in results]
        det_counts = [r.best_detected for r in results]
        
        bars1 = ax4.bar([i - width/2 for i in x], gt_counts, width, label='Ground Truth', color='red', alpha=0.7)
        bars2 = ax4.bar([i + width/2 for i in x], det_counts, width, label='Best Detection', color='green', alpha=0.7)
        
        ax4.set_xlabel('Image', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax4.set_title('Ground Truth vs Best Detection', fontsize=14, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels([r.image_name[:12] for r in results], rotation=45, ha='right')
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        fig_path = os.path.join(self.output_dir, 'all_images_summary.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nSummary chart saved: {fig_path}")
    
    def _save_results_json(self, results: List[ImageResult]):
        """Save results to JSON file."""
        output = {
            'summary': {
                'total_images': len(results),
                'total_ground_truth': sum(r.ground_truth_count for r in results),
                'total_detected': sum(r.best_detected for r in results),
                'total_mae': sum(r.best_mae for r in results),
                'average_mae': sum(r.best_mae for r in results) / len(results) if results else 0
            },
            'per_image': {}
        }
        
        for r in results:
            output['per_image'][r.image_name] = {
                'ground_truth': r.ground_truth_count,
                'best_level': r.best_level,
                'best_level_name': r.best_level_name,
                'best_detected': r.best_detected,
                'best_mae': r.best_mae,
                'all_levels': {
                    lr.level: {
                        'detected': lr.detected_count,
                        'mae': lr.mae,
                        'error_percent': lr.error_percent
                    }
                    for lr in r.all_levels
                }
            }
        
        json_path = os.path.join(self.output_dir, 'optimization_results.json')
        with open(json_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Results saved: {json_path}")


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize beach crowd detection pipeline with ground truth comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python visualize_with_ground_truth.py -i images -a coordinates.csv
  python visualize_with_ground_truth.py -i images -a gt.csv -o results
        '''
    )
    
    parser.add_argument('-i', '--images-dir', default='images',
                       help='Directory containing input images (default: images)')
    
    parser.add_argument('-a', '--annotations', default='coordinates.csv',
                       help='Path to ground truth CSV (default: coordinates.csv)')
    
    parser.add_argument('-o', '--output-dir', default='pipeline_gt_outputs',
                       help='Directory for output files (default: pipeline_gt_outputs)')
    
    parser.add_argument('--all-steps', action='store_true',
                       help='Show all intermediate steps (default: True)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    visualizer = PipelineVisualizerWithGroundTruth(
        images_dir=args.images_dir,
        annotations_path=args.annotations,
        output_dir=args.output_dir
    )
    
    results = visualizer.process_all_images(show_all_steps=True)
    
    print(f"\n✅ All visualizations saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
