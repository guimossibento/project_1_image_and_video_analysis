import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import sys
from pathlib import Path
import argparse


def adjust_gamma(image, gamma=0.4):
    img = image.astype(np.float32) / 255.0
    img = np.power(img, gamma)
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def get_blob_params_level(level):
    """
    Get blob detection parameters by accuracy level.

    Level 1:  Ultra loose  - catches everything, many false positives
    Level 5:  Balanced     - good trade-off between recall and precision
    Level 10: Ultra strict - high precision, may miss some people

    PARAMETER EXPLANATIONS:
    =======================

    min_area (pixels²):
        Minimum blob size to consider. People in beach images typically occupy
        300-5000 pixels depending on distance from camera. Lower values catch
        distant people but also noise; higher values miss distant/small people.

    max_area (pixels²):
        Maximum blob size. Prevents detecting large regions like umbrellas,
        beach towels, or merged groups as single entities. People rarely
        exceed 15000-20000 pixels unless very close to camera.

    min_circularity (0-1):
        Measures how close to a perfect circle: 4π × Area / Perimeter²
        - Circle = 1.0
        - Square ≈ 0.785
        - Human silhouette ≈ 0.2-0.5
        Low values accept elongated shapes (standing people, shadows).
        High values reject irregular shapes but may miss seated/lying people.

    min_convexity (0-1):
        Ratio of blob area to its convex hull area.
        - Solid shape = 1.0
        - Shape with indentations < 1.0
        People with arms out or partial occlusion have lower convexity (~0.4-0.7).
        Low values accept fragmented detections; high values require solid blobs.

    min_inertia (0-1):
        Measures elongation (ratio of minor to major axis).
        - Circle = 1.0
        - Line = 0.0
        Standing people are elongated (~0.1-0.3), sitting people rounder (~0.4-0.6).
        Low values accept standing silhouettes; high values prefer compact shapes.
    """

    blob_params = {
        # Level 1 - Ultra Loose
        # Use case: Initial exploration, ensure nothing is missed
        # Trade-off: Many false positives (rocks, debris, shadows)
        1: {
            'min_area': 30,
            'max_area': 60000,
            'min_circularity': 0.02,
            'min_convexity': 0.15,
            'min_inertia': 0.005
        },

        # Level 2 - Very Loose
        # Use case: Crowded scenes where people overlap significantly
        # Trade-off: Still catches noise, but filters extreme outliers
        2: {
            'min_area': 80,
            'max_area': 50000,
            'min_circularity': 0.05,
            'min_convexity': 0.20,
            'min_inertia': 0.01
        },

        # Level 3 - Loose
        # Use case: Distant crowds, small people in frame
        # Trade-off: Good recall for distant people, some false positives
        3: {
            'min_area': 150,
            'max_area': 40000,
            'min_circularity': 0.08,
            'min_convexity': 0.28,
            'min_inertia': 0.02
        },

        # Level 4 - Moderately Loose
        # Use case: Mixed distances, some occlusion expected
        # Trade-off: Balances distant detection with noise reduction
        4: {
            'min_area': 220,
            'max_area': 32000,
            'min_circularity': 0.12,
            'min_convexity': 0.35,
            'min_inertia': 0.035
        },

        # Level 5 - Balanced (DEFAULT)
        # Use case: General purpose, typical beach scenes
        # Trade-off: Best overall accuracy for most scenarios
        5: {
            'min_area': 300,
            'max_area': 25000,
            'min_circularity': 0.15,
            'min_convexity': 0.42,
            'min_inertia': 0.05
        },

        # Level 6 - Moderately Strict
        # Use case: Cleaner images, less noise in background
        # Trade-off: May miss some distant or partially occluded people
        6: {
            'min_area': 380,
            'max_area': 20000,
            'min_circularity': 0.20,
            'min_convexity': 0.48,
            'min_inertia': 0.07
        },

        # Level 7 - Strict
        # Use case: Close-up scenes, clear visibility
        # Trade-off: Good precision, misses distant/small people
        7: {
            'min_area': 450,
            'max_area': 17000,
            'min_circularity': 0.25,
            'min_convexity': 0.55,
            'min_inertia': 0.09
        },

        # Level 8 - Very Strict
        # Use case: High-quality images, minimal occlusion
        # Trade-off: High precision, lower recall
        8: {
            'min_area': 520,
            'max_area': 14000,
            'min_circularity': 0.30,
            'min_convexity': 0.62,
            'min_inertia': 0.11
        },

        # Level 9 - Extra Strict
        # Use case: When false positives are costly
        # Trade-off: Very few false positives, misses many true positives
        9: {
            'min_area': 600,
            'max_area': 12000,
            'min_circularity': 0.38,
            'min_convexity': 0.70,
            'min_inertia': 0.14
        },

        # Level 10 - Ultra Strict
        # Use case: Only detect very clear, well-defined people
        # Trade-off: Minimal false positives, significant missed detections
        10: {
            'min_area': 700,
            'max_area': 10000,
            'min_circularity': 0.45,
            'min_convexity': 0.78,
            'min_inertia': 0.18
        }
    }

    return blob_params.get(level, blob_params[5])


def get_level_name(level):
    """Get human-readable name for each level."""
    names = {
        1: 'ultra loose',
        2: 'very loose',
        3: 'loose',
        4: 'moderately loose',
        5: 'balanced',
        6: 'moderately strict',
        7: 'strict',
        8: 'very strict',
        9: 'extra strict',
        10: 'ultra strict'
    }
    return names.get(level, 'balanced')


def print_blob_params_table():
    """Print a formatted table of all blob parameters."""
    print("\n" + "=" * 85)
    print("BLOB DETECTION PARAMETERS BY LEVEL")
    print("=" * 85)
    print(f"{'Level':<6} {'Name':<18} {'min_area':<10} {'max_area':<10} "
          f"{'circ':<8} {'conv':<8} {'inertia':<8}")
    print("-" * 85)

    for level in range(1, 11):
        params = get_blob_params_level(level)
        name = get_level_name(level)
        print(f"{level:<6} {name:<18} {params['min_area']:<10} {params['max_area']:<10} "
              f"{params['min_circularity']:<8.3f} {params['min_convexity']:<8.2f} "
              f"{params['min_inertia']:<8.3f}")

    print("=" * 85 + "\n")


class BeachCrowdCounter:

    def __init__(self, images_dir='images', annotations_path='coordinates.csv',
                 output_dir='outputs', params=None, level='balanced'):
        self.images_dir = images_dir
        self.annotations_path = annotations_path
        self.output_dir = output_dir

        # Load parameters
        if params is None:
            self.params = self.default_params(level)
        else:
            self.params = params

        os.makedirs(output_dir, exist_ok=True)
        self.annotations = self._load_annotations()

    def default_params(self, accuracy_level=3):
        """Default parameters with configurable blob accuracy level"""

        blob = get_blob_params_level(accuracy_level)

        return {
            'clahe_clip': 2.0,
            'gamma': 0.4,
            'gaussian_size': 5,
            'top_mask_percent': 0.45,
            'hsv_s_max': 50,
            'hsv_v_min': 100,
            'morph_size': 5,
            'adaptive_block_size': 11,
            'adaptive_c': 2,

            # Blob detection (from selected level)
            **blob
        }

    def _load_annotations(self):
        if not os.path.exists(self.annotations_path):
            print(f"Warning: Annotations file not found", file=sys.stderr)
            return pd.DataFrame()

        df = pd.read_csv(self.annotations_path, sep=';')
        return df

    def get_image_annotations(self, image_name):
        if self.annotations.empty:
            return np.array([])

        img_annotations = self.annotations[self.annotations['file'] == image_name]
        return img_annotations[['x', 'y']].values if len(img_annotations) > 0 else np.array([])

    def create_spatial_mask(self, image_shape):
        h, w = image_shape[:2]
        mask = np.ones((h, w), dtype=np.uint8) * 255
        exclude_h = int(h * self.params['top_mask_percent'])
        mask[:exclude_h, :] = 0
        return mask

    def preprocess_image(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=float(self.params['clahe_clip']),
                                tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)

        lab_clahe = cv2.merge([l_clahe, a, b])
        enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        enhanced = adjust_gamma(enhanced, gamma=float(self.params['gamma']))

        blur_size = int(self.params['gaussian_size'])
        if blur_size % 2 == 0:
            blur_size += 1
        blurred = cv2.GaussianBlur(enhanced, (blur_size, blur_size), 0)

        mask = self.create_spatial_mask(image.shape)
        masked = blurred.copy()
        masked[mask == 0] = 0

        return masked, mask

    def blob_detection(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_sand = np.array([0, 0, int(self.params['hsv_v_min'])])
        upper_sand = np.array([180, int(self.params['hsv_s_max']), 255])
        sand_mask = cv2.inRange(hsv, lower_sand, upper_sand)
        non_sand_mask = cv2.bitwise_not(sand_mask)

        morph_size = int(self.params['morph_size'])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size, morph_size))
        non_sand_mask = cv2.morphologyEx(non_sand_mask, cv2.MORPH_OPEN, kernel)
        non_sand_mask = cv2.morphologyEx(non_sand_mask, cv2.MORPH_CLOSE, kernel)

        filtered = cv2.bitwise_and(image, image, mask=non_sand_mask)
        gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)

        block_size = int(self.params['adaptive_block_size'])
        if block_size % 2 == 0:
            block_size += 1
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, block_size,
                                       int(self.params['adaptive_c']))

        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = int(self.params['min_area'])
        params.maxArea = int(self.params['max_area'])
        params.filterByCircularity = True
        params.minCircularity = float(self.params['min_circularity'])
        params.filterByConvexity = True
        params.minConvexity = float(self.params['min_convexity'])
        params.filterByInertia = True
        params.minInertiaRatio = float(self.params['min_inertia'])
        params.filterByColor = False

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(thresh)
        points = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])

        return points, len(keypoints)

    def visualize_results(self, image, preprocessed, mask, ground_truth,
                          detected_points, output_path, image_name, gt_count, detected_count):

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('Preprocessed (CLAHE + Gamma + Blur)', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(mask, cmap='gray')
        axes[0, 2].set_title(f'Spatial Mask (Top {int(self.params["top_mask_percent"] * 100)}% Excluded)',
                             fontsize=14, fontweight='bold')
        axes[0, 2].axis('off')

        img_gt = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
        if len(ground_truth) > 0:
            for x, y in ground_truth:
                cv2.circle(img_gt, (int(x), int(y)), 8, (255, 0, 0), 2)
        axes[1, 0].imshow(img_gt)
        axes[1, 0].set_title(f'Ground Truth: {gt_count} people', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')

        img_detected = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
        if len(detected_points) > 0:
            for x, y in detected_points:
                cv2.circle(img_detected, (int(x), int(y)), 8, (0, 255, 0), 2)
        axes[1, 1].imshow(img_detected)
        axes[1, 1].set_title(f'Blob Detection: {detected_count} people', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')

        img_compare = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
        if len(ground_truth) > 0:
            for x, y in ground_truth:
                cv2.circle(img_compare, (int(x), int(y)), 8, (255, 0, 0), 2)
        if len(detected_points) > 0:
            for x, y in detected_points:
                cv2.circle(img_compare, (int(x), int(y)), 6, (0, 255, 0), -1)
        axes[1, 2].imshow(img_compare)

        mae = abs(detected_count - gt_count)
        error_pct = (mae / gt_count * 100) if gt_count > 0 else 0

        title = f'Overlay: Red=Ground Truth ({gt_count}), Green=Detected ({detected_count})\n'
        title += f'MAE: {mae} | Error: {error_pct:.1f}%'
        axes[1, 2].set_title(title, fontsize=14, fontweight='bold')
        axes[1, 2].axis('off')

        plt.suptitle(f'Beach Crowd Counting: {image_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def process_single_image(self, image_name, visualize=False):
        image_path = os.path.join(self.images_dir, image_name)
        if not os.path.exists(image_path):
            return None

        image = cv2.imread(image_path)
        if image is None:
            return None

        ground_truth = self.get_image_annotations(image_name)
        gt_count = len(ground_truth)

        preprocessed, mask = self.preprocess_image(image)
        detected_points, detected_count = self.blob_detection(preprocessed)

        if visualize:
            output_path = os.path.join(self.output_dir,
                                       f"{Path(image_name).stem}_result.png")
            self.visualize_results(image, preprocessed, mask, ground_truth,
                                   detected_points, output_path, image_name,
                                   gt_count, detected_count)

        mae = abs(detected_count - gt_count)
        return {
            'image': image_name,
            'ground_truth': gt_count,
            'detected': detected_count,
            'mae': mae,
            'error_percent': (mae / gt_count * 100) if gt_count > 0 else 0,
            'image_size': f"{image.shape[1]}x{image.shape[0]}"
        }

    def process_all_images(self, visualize_all=True, verbose=True):
        image_files = [f for f in os.listdir(self.images_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if not image_files:
            if verbose:
                print(f"No images found in {self.images_dir}", file=sys.stderr)
            return None

        image_files.sort()
        if verbose:
            print(f"\nProcessing {len(image_files)} images...")

        all_results = []
        for i, image_name in enumerate(image_files):
            if verbose:
                print(f"[{i + 1}/{len(image_files)}] {image_name}")
            result = self.process_single_image(image_name, visualize=visualize_all)
            if result:
                all_results.append(result)
                if verbose:
                    print(f"  GT: {result['ground_truth']} | Detected: {result['detected']} | MAE: {result['mae']}")

        results_df = pd.DataFrame(all_results)

        results_csv_path = os.path.join(self.output_dir, 'results.csv')
        results_df.to_csv(results_csv_path, index=False)

        mean_mae = results_df['mae'].mean()
        std_mae = results_df['mae'].std()
        mean_error_pct = results_df['error_percent'].mean()

        summary_stats = {
            'total_images': len(results_df),
            'mean_mae': float(mean_mae),
            'std_mae': float(std_mae),
            'mean_error_percent': float(mean_error_pct),
            'total_ground_truth': int(results_df['ground_truth'].sum()),
            'total_detected': int(results_df['detected'].sum()),
            'parameters': self.params
        }

        summary_path = os.path.join(self.output_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary_stats, f, indent=2)

        if verbose:
            print("\n" + "=" * 70)
            print("RESULTS SUMMARY")
            print("=" * 70)
            print(f"Mean Absolute Error: {mean_mae:.2f} ± {std_mae:.2f}")
            print(f"Mean Error Percentage: {mean_error_pct:.1f}%")
            print(f"Total People (GT): {summary_stats['total_ground_truth']}")
            print(f"Total Detected: {summary_stats['total_detected']}")
            print("=" * 70)

        if visualize_all:
            self.plot_results(results_df)

        return results_df

    def plot_results(self, results_df):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        axes[0].scatter(results_df['ground_truth'], results_df['detected'],
                        alpha=0.7, s=120, color='#2196F3')

        max_val = max(results_df['ground_truth'].max(), results_df['detected'].max())
        axes[0].plot([0, max_val], [0, max_val], 'r--', alpha=0.5, linewidth=2, label='Perfect Prediction')

        axes[0].set_xlabel('Ground Truth Count', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Detected Count', fontsize=12, fontweight='bold')
        axes[0].set_title('Detection Accuracy', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].bar(['Mean Absolute Error'], [results_df['mae'].mean()],
                    alpha=0.8, color='#FF5722')
        axes[1].set_ylabel('MAE (people)', fontsize=12, fontweight='bold')
        axes[1].set_title('Overall Performance', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].text(0, results_df['mae'].mean(),
                     f"{results_df['mae'].mean():.2f}",
                     ha='center', va='bottom', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'performance.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\nPlot saved: {plot_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Beach Crowd Counter - Detect and count people in beach images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python main.py                              # Run with balanced (default) params
  python main.py --level 8                   # Run with very strict detection
  python main.py --level 2 -v                # Very loose detection + visualization
  python main.py params.json                 # Load params from JSON file
  python main.py params.json --level 7       # JSON params override level

Levels:
  1  = Ultra loose       (max recall, many false positives)
  2  = Very loose
  3  = Loose
  4  = Moderately loose
  5  = Balanced (default)
  6  = Moderately strict
  7  = Strict
  8  = Very strict
  9  = Extra strict
  10 = Ultra strict      (max precision, fewer detections)
        '''
    )

    parser.add_argument(
        'params_file',
        nargs='?',
        default=None,
        help='JSON file with optimized parameters (optional)'
    )

    parser.add_argument(
        '-l', '--level',
        type=int,
        choices=range(1, 11),
        default=5,
        metavar='1-10',
        help='Detection accuracy level 1-10 (default: 5)'
    )

    parser.add_argument(
        '-v', '--visualize',
        action='store_true',
        help='Show visualization for each processed image'
    )

    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )

    parser.add_argument(
        '-i', '--images-dir',
        default='images',
        help='Directory containing input images (default: images)'
    )

    parser.add_argument(
        '-a', '--annotations',
        default='coordinates.csv',
        help='Path to annotations CSV file (default: coordinates.csv)'
    )

    parser.add_argument(
        '-o', '--output-dir',
        default='outputs',
        help='Directory for output files (default: outputs)'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Determine parameters source
    params = None

    if args.params_file and args.params_file.endswith('.json'):
        # Load parameters from JSON file
        with open(args.params_file, 'r') as f:
            params = json.load(f)['best_params']
        print(f"Loaded parameters from {args.params_file}")
        print(f"Parameters: {params}")
    else:
        # Use level-based parameters
        level_names = {
            1: 'ultra loose',
            2: 'very loose',
            3: 'loose',
            4: 'moderately loose',
            5: 'balanced',
            6: 'moderately strict',
            7: 'strict',
            8: 'very strict',
            9: 'extra strict',
            10: 'ultra strict'
        }
        print(f"Using detection level: {args.level} ({level_names[args.level]})")

    # Create counter instance
    counter = BeachCrowdCounter(
        images_dir=args.images_dir,
        annotations_path=args.annotations,
        output_dir=args.output_dir,
        params=params,
        level=args.level
    )

    # Process images
    counter.process_all_images(
        visualize_all=args.visualize,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
