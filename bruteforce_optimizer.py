"""
Brute Force Parameter Optimizer
Systematically explores parameter space by running detector_final.py
Starts from minimum values and incrementally increases each parameter
"""

import os
import json
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import time
from itertools import product
import numpy as np


class BruteForceOptimizer:
    
    def __init__(self, detector_script='detector_final.py', 
                 images_dir='images', 
                 annotations_path='coordinates.csv'):
        self.detector_script = detector_script
        self.images_dir = images_dir
        self.annotations_path = annotations_path
        self.results_dir = 'optimization_results'
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Parameter explanations
        self.param_explanations = self._get_parameter_explanations()
    
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
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def run_detector_with_params(self, params):
        """Run detector_final.py with given parameters"""
        
        # Save parameters to temporary JSON file
        temp_params_file = os.path.join(self.results_dir, 'temp_params.json')
        with open(temp_params_file, 'w') as f:
            json.dump(params, f, indent=2)
        
        # Run detector script
        try:
            result = subprocess.run(
                ['python', self.detector_script, temp_params_file, '--quiet'],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # Read results from summary.json
            summary_file = 'outputs/summary.json'
            if os.path.exists(summary_file):
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                return summary['mean_mae']
            else:
                return float('inf')
        
        except Exception as e:
            print(f"Error running detector: {e}")
            return float('inf')
    
    def optimize(self, mode='standard', max_combinations=None):
        """Run brute force optimization"""
        
        print("=" * 80)
        print("BRUTE FORCE PARAMETER OPTIMIZATION")
        print("=" * 80)
        print(f"\nMode: {mode}")
        print()
        
        # Generate parameter combinations
        param_ranges = self.generate_parameter_combinations(mode)
        
        # Print what will be tested
        print("Parameters to optimize:")
        for param, values in param_ranges.items():
            info = self.param_explanations[param]
            if len(values) > 1:
                print(f"  {param}: {len(values)} values - Priority: {info['priority']}")
        
        # Calculate total combinations
        total_combinations = np.prod([len(v) for v in param_ranges.values()])
        print(f"\nTotal combinations: {total_combinations}")
        
        if max_combinations and total_combinations > max_combinations:
            print(f"WARNING: Limited to {max_combinations} combinations")
            total_combinations = max_combinations
        
        print(f"Estimated time: {total_combinations * 2 / 60:.1f} minutes")
        print()
        
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Optimization cancelled.")
            return None
        
        # Generate all combinations
        param_names = list(param_ranges.keys())
        param_values = [param_ranges[name] for name in param_names]
        
        best_mae = float('inf')
        best_params = None
        all_results = []
        
        start_time = time.time()
        
        combinations = list(product(*param_values))
        if max_combinations:
            combinations = combinations[:max_combinations]
        
        for i, combination in enumerate(combinations):
            # Create parameter dict
            params = dict(zip(param_names, combination))
            
            # Test this combination
            print(f"[{i+1}/{len(combinations)}] Testing...", end=' ')
            mae = self.run_detector_with_params(params)
            
            # Store result
            result = params.copy()
            result['mae'] = mae
            all_results.append(result)
            
            # Check if best
            if mae < best_mae:
                best_mae = mae
                best_params = params.copy()
                print(f"✓ NEW BEST! MAE: {mae:.2f}")
                print(f"    Key params: gamma={params['gamma']:.2f}, "
                      f"minArea={params['min_area']}, inertia={params['min_inertia']:.2f}")
            else:
                print(f"MAE: {mae:.2f}")
            
            # Progress update
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                eta = (elapsed / (i + 1)) * (len(combinations) - i - 1)
                print(f"    Progress: {(i+1)/len(combinations)*100:.1f}% | "
                      f"Elapsed: {elapsed/60:.1f}min | ETA: {eta/60:.1f}min | "
                      f"Best: {best_mae:.2f}")
        
        elapsed = time.time() - start_time
        print(f"\nCompleted in {elapsed/60:.1f} minutes")
        
        # Save results
        self._save_results(best_params, all_results, mode)
        
        return best_params, all_results
    
    def _save_results(self, best_params, all_results, mode):
        """Save optimization results"""
        
        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values('mae')
        
        # Save CSV
        csv_file = os.path.join(self.results_dir, f'bruteforce_{mode}_results.csv')
        results_df.to_csv(csv_file, index=False)
        
        # Save best parameters
        params_file = os.path.join(self.results_dir, f'best_params_bruteforce_{mode}.json')
        with open(params_file, 'w') as f:
            json.dump(best_params, f, indent=2)
        
        # Print summary
        print("\n" + "=" * 80)
        print("OPTIMIZATION RESULTS")
        print("=" * 80)
        print(f"Best MAE: {best_params.get('mae', results_df['mae'].min()):.2f}\n")
        
        # Print by priority
        print("CRITICAL PARAMETERS:")
        for param, info in self.param_explanations.items():
            if info['priority'] == 'CRITICAL' or 'CRITICAL' in info['priority']:
                print(f"  {param}: {best_params[param]} (optimal: {info['optimal_range']})")
        
        print("\nHIGH PRIORITY:")
        for param, info in self.param_explanations.items():
            if info['priority'] == 'HIGH' and param not in ['min_inertia']:
                print(f"  {param}: {best_params[param]} (optimal: {info['optimal_range']})")
        
        print("\nMEDIUM PRIORITY:")
        for param, info in self.param_explanations.items():
            if info['priority'] == 'MEDIUM':
                print(f"  {param}: {best_params[param]}")
        
        print(f"\n{'='*80}")
        print("Files saved:")
        print(f"  - {params_file}")
        print(f"  - {csv_file}")
        print(f"{'='*80}")
        
        # Plot results
        self._plot_results(results_df, mode)
        
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
        
        plt.suptitle(f'Parameter Analysis - {mode.title()} Mode', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plot_file = os.path.join(self.results_dir, f'analysis_{mode}.png')
        plt.savefig(plot_file, dpi=150)
        plt.close()
        
        print(f"  - {plot_file}")


def main():
    print("=" * 80)
    print("BRUTE FORCE PARAMETER OPTIMIZER")
    print("=" * 80)
    print("\nSystematic exploration of parameter space")
    print("Runs detector_final.py with different parameter combinations")
    print()
    
    optimizer = BruteForceOptimizer()
    
    print("Choose optimization mode:")
    print("  1. Minimal - Critical parameters only (~729 combinations, ~25 min)")
    print("  2. Standard - High/Medium priority (~3888 combinations, ~130 min)")
    print("  4. Show parameter guide")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        mode = 'minimal'
        best_params, results = optimizer.optimize(mode=mode)
        
    elif choice == '2':
        mode = 'standard'
        best_params, results = optimizer.optimize(mode=mode)
        
    elif choice == '4':
        optimizer.print_parameter_guide()
        return
    
    else:
        print("Invalid choice")
        return
    
    # Show top 10 results
    if results is not None:
        results_df = pd.DataFrame(results).sort_values('mae')
        print("\n" + "=" * 80)
        print("TOP 10 PARAMETER COMBINATIONS")
        print("=" * 80)
        print(results_df[['mae', 'gamma', 'min_area', 'max_area', 'min_circularity', 
                          'min_convexity', 'min_inertia']].head(10).to_string(index=False))
        print()


if __name__ == "__main__":
    main()
