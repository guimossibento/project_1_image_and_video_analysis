"""
Improved Brute Force Parameter Optimizer
- Uses multiprocessing for parallel execution
- Saves checkpoints during optimization
- Can resume from previous run
- Shows real-time progress with best parameters
"""

import os
import json
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import time
from itertools import product
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
import signal
import sys


class BruteForceOptimizer:

    def __init__(self, detector_script='detector_final.py',
                 images_dir='images',
                 annotations_path='coordinates.csv',
                 n_workers=None):
        self.detector_script = detector_script
        self.images_dir = images_dir
        self.annotations_path = annotations_path
        self.results_dir = 'optimization_results'
        os.makedirs(self.results_dir, exist_ok=True)

        # Set number of workers (leave 1 core free for system)
        self.n_workers = n_workers or max(1, cpu_count() - 1)

        # Parameter explanations
        self.param_explanations = self._get_parameter_explanations()

        # Checkpoint file
        self.checkpoint_file = None
        self.best_params_file = None

    def _get_parameter_explanations(self):
        """Complete explanation of each parameter"""
        return {
            # PREPROCESSING PARAMETERS
            'clahe_clip': {
                'name': 'CLAHE Clip Limit',
                'min': 1.5,
                'max': 3.5,
                'step': 0.5,
                'default': 2.0,
                'why': 'Controls contrast enhancement. Higher values = more contrast but risk amplifying noise. '
                       'Beach scenes need moderate enhancement (2.0-3.0) to reveal people in shadows.',
                'effect': 'Too low (1.5): People in shadows remain dark. '
                          'Too high (4.0): Sand texture becomes noisy.',
                'optimal_range': '2.0-3.0',
                'priority': 'MEDIUM'
            },

            'gamma': {
                'name': 'Gamma Correction',
                'min': 0.3,
                'max': 0.6,
                'step': 0.05,
                'default': 0.4,
                'why': 'Brightens shadows using power-law transformation. Lower gamma = stronger brightening. '
                       'Essential for revealing people walking in shadowed areas.',
                'effect': 'gamma=0.3: Very strong brightening (shadows at 20% → 52%). '
                          'gamma=0.5: Moderate brightening (shadows at 20% → 36%). '
                          'gamma=0.6: Weak brightening (shadows at 20% → 28%).',
                'optimal_range': '0.35-0.45',
                'priority': 'HIGH - Critical for shadow regions'
            },

            'gaussian_size': {
                'name': 'Gaussian Blur Kernel Size',
                'values': [3, 5, 7],
                'default': 5,
                'why': 'Removes high-frequency noise (sand texture, sensor noise). '
                       'Larger kernels = more smoothing but risk losing fine details.',
                'effect': 'Size 3: Minimal smoothing, preserves edges. '
                          'Size 5: Good balance for beach scenes. '
                          'Size 7: Strong smoothing, may blur small people.',
                'optimal_range': '5 (usually best)',
                'priority': 'LOW'
            },

            'top_mask_percent': {
                'name': 'Top Masking Percentage',
                'min': 0.30,
                'max': 0.50,
                'step': 0.05,
                'default': 0.40,
                'why': 'Excludes top portion of image (sky, buildings, horizon). '
                       'Depends on camera angle - higher angle = more masking needed.',
                'effect': '30%: Keeps more image, may include false positives from clouds. '
                          '40%: Standard for elevated cameras. '
                          '50%: Aggressive masking, may cut off far people.',
                'optimal_range': '0.35-0.45',
                'priority': 'MEDIUM'
            },

            'hsv_s_max': {
                'name': 'HSV Saturation Maximum (Sand Removal)',
                'min': 40,
                'max': 60,
                'step': 5,
                'default': 50,
                'why': 'Sand has LOW saturation (desaturated beige). This threshold filters sand pixels. '
                       'Lower = more aggressive sand removal.',
                'effect': 'S_max=40: Aggressive, may remove skin tones. '
                          'S_max=50: Balanced. '
                          'S_max=60: Conservative, may keep too much sand.',
                'optimal_range': '45-55',
                'priority': 'MEDIUM'
            },

            'hsv_v_min': {
                'name': 'HSV Value Minimum (Sand Removal)',
                'min': 80,
                'max': 120,
                'step': 10,
                'default': 100,
                'why': 'Sand is BRIGHT. This threshold removes bright pixels. '
                       'Higher = only remove very bright pixels.',
                'effect': 'V_min=80: Remove more pixels (including some people). '
                          'V_min=100: Balanced. '
                          'V_min=120: Only remove extremely bright sand.',
                'optimal_range': '90-110',
                'priority': 'MEDIUM'
            },

            'morph_size': {
                'name': 'Morphology Kernel Size',
                'values': [3, 5, 7],
                'default': 5,
                'why': 'Opening removes small noise dots. Closing fills small gaps. '
                       'Larger kernels = more aggressive but may erode person boundaries.',
                'effect': 'Size 3: Gentle, preserves small details. '
                          'Size 5: Standard, good for beach noise. '
                          'Size 7: Aggressive, may merge nearby people.',
                'optimal_range': '5 (usually best)',
                'priority': 'LOW'
            },

            'adaptive_block_size': {
                'name': 'Adaptive Threshold Block Size',
                'values': [9, 11, 13, 15],
                'default': 11,
                'why': 'Size of local neighborhood for adaptive thresholding. '
                       'Larger = smoother transitions, smaller = more local adaptation.',
                'effect': 'Block=9: Very local, adapts to small changes. '
                          'Block=11: Good balance. '
                          'Block=15: Global-ish, less adaptation.',
                'optimal_range': '11 (usually best)',
                'priority': 'LOW'
            },

            'adaptive_c': {
                'name': 'Adaptive Threshold Constant',
                'values': [1, 2, 3],
                'default': 2,
                'why': 'Offset subtracted from mean to set threshold. '
                       'Higher = more conservative (fewer foreground pixels).',
                'effect': 'C=1: Liberal thresholding. '
                          'C=2: Balanced. '
                          'C=3: Conservative.',
                'optimal_range': '2 (usually best)',
                'priority': 'LOW'
            },

            # BLOB DETECTION PARAMETERS
            'min_area': {
                'name': 'Minimum Blob Area',
                'min': 60,
                'max': 150,
                'step': 10,
                'default': 100,
                'why': 'Smallest person (far away) has head diameter ~11 pixels → area ~95 px². '
                       'Lower = detect smaller/farther people. Too low = detect noise.',
                'effect': 'minArea=60: Detects very small blobs (noise risk). '
                          'minArea=100: Standard for head size. '
                          'minArea=150: Misses far people.',
                'optimal_range': '80-120',
                'priority': 'HIGH - Affects recall (missing far people)'
            },

            'max_area': {
                'name': 'Maximum Blob Area',
                'min': 2000,
                'max': 4000,
                'step': 500,
                'default': 3000,
                'why': 'Largest person (close) ~50×60 pixels = 3000 px². '
                       'Higher = detect larger objects. Too high = detect umbrellas, merged crowds.',
                'effect': 'maxArea=2000: May miss close people. '
                          'maxArea=3000: Standard for full body. '
                          'maxArea=4000: Risk detecting umbrellas (~5000 px²).',
                'optimal_range': '2500-3500',
                'priority': 'HIGH - Affects precision (false positives from large objects)'
            },

            'min_circularity': {
                'name': 'Minimum Circularity',
                'min': 0.30,
                'max': 0.50,
                'step': 0.05,
                'default': 0.40,
                'why': 'Circularity = 4π×Area/Perimeter². Circle=1.0, standing person≈0.5, towel≈0.25. '
                       'Lower = allow more elongated shapes.',
                'effect': 'Circ=0.30: Accepts very elongated (may include towels). '
                          'Circ=0.40: Rejects towels, keeps people. '
                          'Circ=0.50: May reject standing people.',
                'optimal_range': '0.35-0.45',
                'priority': 'HIGH - Critical for rejecting elongated non-people'
            },

            'min_convexity': {
                'name': 'Minimum Convexity',
                'min': 0.60,
                'max': 0.80,
                'step': 0.05,
                'default': 0.70,
                'why': 'Convexity = BlobArea/ConvexHullArea. Complete shape≈0.9, fragmented<0.6. '
                       'Lower = allow more fragmented shapes.',
                'effect': 'Conv=0.60: Accepts fragmented detections (noise risk). '
                          'Conv=0.70: Rejects fragments, keeps complete people. '
                          'Conv=0.80: Very strict, may reject partially occluded.',
                'optimal_range': '0.65-0.75',
                'priority': 'MEDIUM - Rejects noise clusters'
            },

            'min_inertia': {
                'name': 'Minimum Inertia Ratio',
                'min': 0.15,
                'max': 0.25,
                'step': 0.02,
                'default': 0.20,
                'why': 'Inertia = (width/height)². Standing person (1:2) = 0.25. Surfboard (1:5) = 0.04. '
                       'CRITICAL: 0.20 allows standing people (I=0.25). 0.40 would miss them!',
                'effect': 'Inertia=0.15: Accepts very stretched objects. '
                          'Inertia=0.20: Perfect for standing people. '
                          'Inertia=0.25: Borderline - might miss some standing people.',
                'optimal_range': '0.18-0.22',
                'priority': 'CRITICAL - Most important blob parameter!'
            }
        }

    def print_parameter_guide(self):
        """Print comprehensive parameter guide"""
        print("=" * 80)
        print("PARAMETER OPTIMIZATION GUIDE")
        print("=" * 80)
        print()

        categories = {
            'PREPROCESSING': ['clahe_clip', 'gamma', 'gaussian_size', 'top_mask_percent',
                             'hsv_s_max', 'hsv_v_min', 'morph_size', 'adaptive_block_size', 'adaptive_c'],
            'BLOB DETECTION': ['min_area', 'max_area', 'min_circularity', 'min_convexity', 'min_inertia']
        }

        for category, params in categories.items():
            print(f"\n{'='*80}")
            print(f"{category}")
            print(f"{'='*80}\n")

            for param in params:
                info = self.param_explanations[param]
                print(f"📊 {info['name']} ({param})")
                print(f"   Priority: {info['priority']}")

                if 'values' in info:
                    print(f"   Options: {info['values']} (default: {info['default']})")
                else:
                    print(f"   Range: {info['min']} to {info['max']} step {info['step']} (default: {info['default']})")

                print(f"   Why: {info['why']}")
                print(f"   Effect: {info['effect']}")
                print(f"   Optimal: {info['optimal_range']}")
                print()

    def generate_parameter_combinations(self, mode='standard'):
        """Generate parameter combinations based on mode"""

        if mode == 'minimal':
            # Test only critical parameters
            return {
                'gamma': [0.35, 0.40, 0.45],
                'min_area': [80, 100, 120],
                'max_area': [2500, 3000, 3500],
                'min_circularity': [0.35, 0.40, 0.45],
                'min_convexity': [0.65, 0.70, 0.75],
                'min_inertia': [0.18, 0.20, 0.22],
                # Keep rest as default
                'clahe_clip': [2.0],
                'gaussian_size': [5],
                'top_mask_percent': [0.40],
                'hsv_s_max': [50],
                'hsv_v_min': [100],
                'morph_size': [5],
                'adaptive_block_size': [11],
                'adaptive_c': [2]
            }

        elif mode == 'standard':
            # Test high/medium priority parameters
            return {
                'clahe_clip': [2.0, 2.5, 3.0],
                'gamma': [0.35, 0.40, 0.45],
                'gaussian_size': [5, 7],
                'top_mask_percent': [0.35, 0.40, 0.45],
                'hsv_s_max': [45, 50, 55],
                'hsv_v_min': [90, 100, 110],
                'morph_size': [5],
                'adaptive_block_size': [11],
                'adaptive_c': [2],
                'min_area': [80, 100, 120],
                'max_area': [2500, 3000, 3500],
                'min_circularity': [0.35, 0.40, 0.45],
                'min_convexity': [0.65, 0.70, 0.75],
                'min_inertia': [0.18, 0.20, 0.22]
            }

        elif mode == 'quick':
            # Quick test with fewer combinations (~100)
            return {
                'gamma': [0.35, 0.40, 0.45],
                'min_area': [80, 100],
                'max_area': [2500, 3000],
                'min_circularity': [0.35, 0.40],
                'min_convexity': [0.65, 0.70],
                'min_inertia': [0.18, 0.20, 0.22],
                'clahe_clip': [2.0],
                'gaussian_size': [5],
                'top_mask_percent': [0.40],
                'hsv_s_max': [50],
                'hsv_v_min': [100],
                'morph_size': [5],
                'adaptive_block_size': [11],
                'adaptive_c': [2]
            }

        else:
            raise ValueError(f"Unknown mode: {mode}")

    @staticmethod
    def run_single_combination(args):
        """Run detector for a single parameter combination (static method for multiprocessing)"""
        params, param_names, detector_script, result_idx, results_dir = args

        # Create parameter dict
        param_dict = dict(zip(param_names, params))

        # Save parameters to worker-specific file
        worker_id = os.getpid()
        temp_params_file = os.path.join(results_dir, f'temp_params_worker_{worker_id}.json')

        try:
            with open(temp_params_file, 'w') as f:
                json.dump(param_dict, f, indent=2)

            # Run detector script with timeout
            result = subprocess.run(
                ['python', detector_script, temp_params_file, '--quiet'],
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout per combination
            )

            # Read results
            summary_file = f'outputs/summary_worker_{worker_id}.json'
            if os.path.exists(summary_file):
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                mae = summary['mean_mae']
            else:
                # Fallback to default output location
                summary_file = 'outputs/summary.json'
                if os.path.exists(summary_file):
                    with open(summary_file, 'r') as f:
                        summary = json.load(f)
                    mae = summary['mean_mae']
                else:
                    mae = float('inf')

            # Clean up temp file
            if os.path.exists(temp_params_file):
                os.remove(temp_params_file)

            return (result_idx, param_dict, mae)

        except subprocess.TimeoutExpired:
            print(f"\n⚠️  Worker {worker_id} timed out on combination {result_idx}")
            return (result_idx, param_dict, float('inf'))

        except Exception as e:
            print(f"\n⚠️  Worker {worker_id} error on combination {result_idx}: {e}")
            return (result_idx, param_dict, float('inf'))

    def load_checkpoint(self, mode):
        """Load checkpoint from previous run"""
        checkpoint_file = os.path.join(self.results_dir, f'checkpoint_{mode}.json')

        if os.path.exists(checkpoint_file):
            print(f"\n📂 Found checkpoint: {checkpoint_file}")
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)

            print(f"   Completed: {checkpoint['completed']}/{checkpoint['total']} combinations")
            print(f"   Best MAE so far: {checkpoint['best_mae']:.2f}")

            response = input("   Resume from checkpoint? (y/n): ")
            if response.lower() == 'y':
                return checkpoint

        return None

    def save_checkpoint(self, mode, completed, total, best_mae, best_params, all_results):
        """Save checkpoint during optimization"""
        checkpoint = {
            'mode': mode,
            'completed': completed,
            'total': total,
            'best_mae': best_mae,
            'best_params': best_params,
            'all_results': all_results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)

    def save_best_params_partial(self, best_params, best_mae, completed, total):
        """Save best parameters found so far"""
        best_data = {
            'best_params': best_params,
            'best_mae': best_mae,
            'progress': f"{completed}/{total}",
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(self.best_params_file, 'w') as f:
            json.dump(best_data, f, indent=2)

    def optimize(self, mode='standard', max_combinations=None, resume=True):
        """Run brute force optimization with multiprocessing"""

        print("=" * 80)
        print("PARALLEL BRUTE FORCE PARAMETER OPTIMIZATION")
        print("=" * 80)
        print(f"\nMode: {mode}")
        print(f"Workers: {self.n_workers} CPU cores")
        print()

        # Setup checkpoint files
        self.checkpoint_file = os.path.join(self.results_dir, f'checkpoint_{mode}.json')
        self.best_params_file = os.path.join(self.results_dir, f'best_params_live_{mode}.json')

        # Try to resume from checkpoint
        checkpoint = None
        if resume:
            checkpoint = self.load_checkpoint(mode)

        # Generate parameter combinations
        param_ranges = self.generate_parameter_combinations(mode)
        param_names = list(param_ranges.keys())
        param_values = [param_ranges[name] for name in param_names]

        # Generate all combinations
        all_combinations = list(product(*param_values))
        if max_combinations:
            all_combinations = all_combinations[:max_combinations]

        total_combinations = len(all_combinations)

        # Resume from checkpoint if available
        if checkpoint:
            completed_indices = set(r['index'] for r in checkpoint['all_results'])
            remaining_combinations = [(i, combo) for i, combo in enumerate(all_combinations)
                                     if i not in completed_indices]
            best_mae = checkpoint['best_mae']
            best_params = checkpoint['best_params']
            all_results = checkpoint['all_results']
            start_idx = len(all_results)

            print(f"Resuming: {len(remaining_combinations)} combinations remaining")
        else:
            remaining_combinations = list(enumerate(all_combinations))
            best_mae = float('inf')
            best_params = None
            all_results = []
            start_idx = 0

        # Print optimization info
        print("Parameters to optimize:")
        for param, values in param_ranges.items():
            info = self.param_explanations[param]
            if len(values) > 1:
                print(f"  {param}: {len(values)} values - Priority: {info['priority']}")

        print(f"\nTotal combinations: {total_combinations}")
        print(f"Estimated time (sequential): {total_combinations * 2 / 60:.1f} minutes")
        print(f"Estimated time (parallel): {total_combinations * 2 / 60 / self.n_workers:.1f} minutes")
        print()

        if not checkpoint:
            response = input("Continue? (y/n): ")
            if response.lower() != 'y':
                print("Optimization cancelled.")
                return None

        # Prepare arguments for multiprocessing
        tasks = [
            (combo, param_names, self.detector_script, idx, self.results_dir)
            for idx, combo in remaining_combinations
        ]

        print(f"\n{'='*80}")
        print("OPTIMIZATION STARTED")
        print(f"{'='*80}\n")

        start_time = time.time()

        # Use multiprocessing pool
        try:
            with Pool(processes=self.n_workers) as pool:
                # Process results as they complete
                for i, (result_idx, param_dict, mae) in enumerate(pool.imap_unordered(
                    self.run_single_combination, tasks)):

                    # Store result
                    result = param_dict.copy()
                    result['mae'] = mae
                    result['index'] = result_idx
                    all_results.append(result)

                    completed = start_idx + i + 1

                    # Check if best
                    if mae < best_mae:
                        best_mae = mae
                        best_params = param_dict.copy()

                        print(f"[{completed}/{total_combinations}] ⭐ NEW BEST! MAE: {mae:.2f}")
                        print(f"    gamma={param_dict['gamma']:.2f}, "
                              f"minArea={param_dict['min_area']}, "
                              f"maxArea={param_dict['max_area']}, "
                              f"circ={param_dict['min_circularity']:.2f}, "
                              f"inertia={param_dict['min_inertia']:.2f}")

                        # Save best parameters immediately
                        self.save_best_params_partial(best_params, best_mae, completed, total_combinations)

                    else:
                        # Periodic progress update (every 5%)
                        if completed % max(1, total_combinations // 20) == 0:
                            elapsed = time.time() - start_time
                            eta = (elapsed / completed) * (total_combinations - completed)
                            progress = completed / total_combinations * 100

                            print(f"[{completed}/{total_combinations}] "
                                  f"Progress: {progress:.1f}% | "
                                  f"Elapsed: {elapsed/60:.1f}min | "
                                  f"ETA: {eta/60:.1f}min | "
                                  f"Best MAE: {best_mae:.2f}")

                    # Save checkpoint every 10% or every 50 combinations
                    if completed % max(10, total_combinations // 10) == 0:
                        self.save_checkpoint(mode, completed, total_combinations,
                                           best_mae, best_params, all_results)
                        print(f"    💾 Checkpoint saved")

        except KeyboardInterrupt:
            print("\n\n⚠️  Optimization interrupted by user")
            print("Saving checkpoint...")
            self.save_checkpoint(mode, len(all_results), total_combinations,
                               best_mae, best_params, all_results)
            print("✓ Checkpoint saved. You can resume later.")
            return None

        elapsed = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"OPTIMIZATION COMPLETED in {elapsed/60:.1f} minutes")
        print(f"{'='*80}\n")

        # Save final results
        self._save_results(best_params, best_mae, all_results, mode)

        # Clean up checkpoint file
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)

        return best_params, all_results

    def _save_results(self, best_params, best_mae, all_results, mode):
        """Save optimization results"""

        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)
        if 'index' in results_df.columns:
            results_df = results_df.drop('index', axis=1)
        results_df = results_df.sort_values('mae')

        # Save CSV
        csv_file = os.path.join(self.results_dir, f'bruteforce_{mode}_results.csv')
        results_df.to_csv(csv_file, index=False)

        # Save best parameters
        params_file = os.path.join(self.results_dir, f'best_params_bruteforce_{mode}.json')
        best_data = {
            'best_params': best_params,
            'best_mae': best_mae,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(params_file, 'w') as f:
            json.dump(best_data, f, indent=2)

        # Print summary
        print("\n" + "=" * 80)
        print("OPTIMIZATION RESULTS")
        print("=" * 80)
        print(f"Best MAE: {best_mae:.2f}\n")

        # Print by priority
        print("CRITICAL PARAMETERS:")
        for param, info in self.param_explanations.items():
            if 'CRITICAL' in str(info['priority']):
                print(f"  {param}: {best_params[param]} (optimal: {info['optimal_range']})")

        print("\nHIGH PRIORITY:")
        for param, info in self.param_explanations.items():
            if info['priority'] == 'HIGH':
                print(f"  {param}: {best_params[param]} (optimal: {info['optimal_range']})")

        print("\nMEDIUM PRIORITY:")
        for param, info in self.param_explanations.items():
            if info['priority'] == 'MEDIUM':
                print(f"  {param}: {best_params[param]}")

        print(f"\n{'='*80}")
        print("Files saved:")
        print(f"  - {params_file}")
        print(f"  - {csv_file}")

        # Plot results
        self._plot_results(results_df, mode)

        print(f"  - {os.path.join(self.results_dir, f'analysis_{mode}.png')}")
        print(f"{'='*80}")

        # Show top 10
        print("\n" + "=" * 80)
        print("TOP 10 PARAMETER COMBINATIONS")
        print("=" * 80)
        top_cols = ['mae', 'gamma', 'min_area', 'max_area', 'min_circularity',
                    'min_convexity', 'min_inertia']
        print(results_df[top_cols].head(10).to_string(index=False))
        print()

        return results_df

    def _plot_results(self, results_df, mode):
        """Plot optimization results"""

        # Top 20 parameters
        top_results = results_df.head(20)

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        critical_params = [
            ('gamma', 'Gamma'),
            ('min_area', 'Min Area'),
            ('max_area', 'Max Area'),
            ('min_circularity', 'Min Circularity'),
            ('min_convexity', 'Min Convexity'),
            ('min_inertia', 'Min Inertia ⭐')
        ]

        for idx, (param, label) in enumerate(critical_params):
            ax = axes[idx // 3, idx % 3]

            x = range(len(top_results))
            y = top_results[param].values
            colors = top_results['mae'].values

            scatter = ax.scatter(x, y, c=colors, cmap='RdYlGn_r', s=120, alpha=0.7)
            ax.set_xlabel('Rank (0=best)', fontsize=11, fontweight='bold')
            ax.set_ylabel(label, fontsize=11, fontweight='bold')
            ax.set_title(f'Top 20: {label}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Highlight optimal range
            info = self.param_explanations[param]
            if 'optimal_range' in info and '-' in str(info['optimal_range']):
                try:
                    opt_min, opt_max = map(float, info['optimal_range'].split('-'))
                    ax.axhspan(opt_min, opt_max, alpha=0.1, color='green', label='Optimal Range')
                    ax.legend(fontsize=8)
                except:
                    pass

            plt.colorbar(scatter, ax=ax, label='MAE')

        plt.suptitle(f'Parameter Analysis - {mode.title()} Mode\n'
                     f'Best MAE: {top_results["mae"].iloc[0]:.2f}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        plot_file = os.path.join(self.results_dir, f'analysis_{mode}.png')
        plt.savefig(plot_file, dpi=150)
        plt.close()


def main():
    print("=" * 80)
    print("PARALLEL BRUTE FORCE PARAMETER OPTIMIZER")
    print("=" * 80)
    print(f"\nAvailable CPU cores: {cpu_count()}")
    print("Systematic exploration with multiprocessing")
    print()

    # Allow custom worker count
    n_workers = input(f"Number of workers (default={max(1, cpu_count()-1)}): ").strip()
    n_workers = int(n_workers) if n_workers else None

    optimizer = BruteForceOptimizer(n_workers=n_workers)

    print("\nChoose optimization mode:")
    print("  1. Quick - Fast test (~144 combinations, ~5 min with 4 cores)")
    print("  2. Minimal - Critical parameters (~729 combinations, ~20 min with 4 cores)")
    print("  3. Standard - High/Medium priority (~3888 combinations, ~110 min with 4 cores)")
    print("  4. Show parameter guide")

    choice = input("\nEnter choice (1-4): ").strip()

    if choice == '1':
        mode = 'quick'
        best_params, results = optimizer.optimize(mode=mode)

    elif choice == '2':
        mode = 'minimal'
        best_params, results = optimizer.optimize(mode=mode)

    elif choice == '3':
        mode = 'standard'
        best_params, results = optimizer.optimize(mode=mode)

    elif choice == '4':
        optimizer.print_parameter_guide()
        return

    else:
        print("Invalid choice")
        return


if __name__ == "__main__":
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print('\n\n⚠️  Caught interrupt signal. Exiting gracefully...')
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    main()