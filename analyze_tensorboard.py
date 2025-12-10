#!/usr/bin/env python3
"""
Analyze TensorBoard logs for Ambient Diffusion experiment on Matcha TTS.
Extract training curves and generate comparison plots.
"""

import os
import glob
from collections import defaultdict
import json

# Try to import tensorboard and create plots, fallback to basic extraction
try:
    from tensorboard.backend.event_processing import event_accumulator
    HAS_TB = True
except ImportError:
    HAS_TB = False
    print("TensorBoard not available, attempting alternative parsing...")

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("Matplotlib not available, will output data as text/JSON...")


def find_event_files(base_dir):
    """Find all TensorBoard event files and their corresponding t_max values."""
    runs = {}
    
    # Search for event files
    pattern = os.path.join(base_dir, "logs/train/*/runs/*/tensorboard/*/events.out.tfevents.*")
    event_files = glob.glob(pattern)
    
    for ef in event_files:
        # Extract t_max from path
        parts = ef.split(os.sep)
        for part in parts:
            if part.startswith("combined_tmax_"):
                t_max_str = part.replace("combined_tmax_", "")
                try:
                    t_max = float(t_max_str)
                except ValueError:
                    continue
                runs[t_max] = ef
                break
    
    return runs


def extract_scalars_tb(event_file):
    """Extract scalar data from TensorBoard event file using tensorboard library."""
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    
    scalars = {}
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        scalars[tag] = {
            'steps': [e.step for e in events],
            'values': [e.value for e in events],
            'wall_times': [e.wall_time for e in events]
        }
    
    return scalars


def extract_scalars_binary(event_file):
    """Fallback: Parse TensorBoard event file binary directly."""
    import struct
    
    scalars = defaultdict(lambda: {'steps': [], 'values': [], 'wall_times': []})
    
    try:
        with open(event_file, 'rb') as f:
            while True:
                # Read the header (length of the data)
                header = f.read(8)
                if len(header) < 8:
                    break
                
                length = struct.unpack('Q', header)[0]
                
                # Read masked CRC of length
                f.read(4)
                
                # Read the data
                data = f.read(length)
                if len(data) < length:
                    break
                
                # Read masked CRC of data  
                f.read(4)
                
                # Try to parse as protobuf (basic extraction)
                # This is a simplified parser - may not get all data
                try:
                    from tensorflow.core.util import event_pb2
                    event = event_pb2.Event()
                    event.ParseFromString(data)
                    
                    if event.HasField('summary'):
                        for value in event.summary.value:
                            if value.HasField('simple_value'):
                                scalars[value.tag]['steps'].append(event.step)
                                scalars[value.tag]['values'].append(value.simple_value)
                                scalars[value.tag]['wall_times'].append(event.wall_time)
                except:
                    pass
                    
    except Exception as e:
        print(f"Error parsing {event_file}: {e}")
    
    return dict(scalars)


def extract_all_data(base_dir):
    """Extract data from all runs."""
    runs = find_event_files(base_dir)
    print(f"Found {len(runs)} runs: {sorted(runs.keys())}")
    
    all_data = {}
    
    for t_max, event_file in sorted(runs.items()):
        print(f"\nProcessing t_max = {t_max}...")
        print(f"  File: {event_file}")
        
        if HAS_TB:
            try:
                scalars = extract_scalars_tb(os.path.dirname(event_file))
            except Exception as e:
                print(f"  TensorBoard extraction failed: {e}")
                scalars = extract_scalars_binary(event_file)
        else:
            scalars = extract_scalars_binary(event_file)
        
        all_data[t_max] = scalars
        
        # Print available tags
        if scalars:
            print(f"  Available tags: {list(scalars.keys())}")
            for tag, data in scalars.items():
                if data['values']:
                    print(f"    {tag}: {len(data['values'])} points, "
                          f"range [{min(data['values']):.4f}, {max(data['values']):.4f}]")
    
    return all_data


def create_comparison_plots(all_data, output_dir):
    """Create comparison plots for all runs."""
    if not HAS_PLOT:
        print("Cannot create plots - matplotlib not available")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Define colors for different t_max values
    colors = {
        0.0: '#1f77b4',   # Blue (baseline)
        0.25: '#ff7f0e',  # Orange
        0.5: '#2ca02c',   # Green
        0.75: '#d62728',  # Red
        1.0: '#9467bd'    # Purple
    }
    
    # Tags to plot
    tags_to_plot = [
        ('loss/train_epoch', 'Training Loss (per epoch)'),
        ('loss/val_epoch', 'Validation Loss (per epoch)'),
        ('loss/train_step', 'Training Loss (per step)'),
        ('loss/val_step', 'Validation Loss (per step)'),
    ]
    
    # Create individual plots
    for tag, title in tags_to_plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        has_data = False
        for t_max in sorted(all_data.keys()):
            data = all_data[t_max]
            if tag in data and data[tag]['values']:
                steps = data[tag]['steps']
                values = data[tag]['values']
                label = f'$t_{{max}}$ = {t_max}' + (' (baseline)' if t_max == 0 else '')
                ax.plot(steps, values, label=label, color=colors.get(t_max, 'gray'), 
                       linewidth=1.5 if t_max == 0 else 1.0,
                       linestyle='-' if t_max == 0 else '--')
                has_data = True
        
        if has_data:
            ax.set_xlabel('Step')
            ax.set_ylabel('Loss')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            filename = tag.replace('/', '_') + '.png'
            fig.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
            print(f"Saved: {filename}")
        
        plt.close(fig)
    
    # Create combined validation loss comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training loss
    ax = axes[0]
    tag = 'loss/train_epoch'
    for t_max in sorted(all_data.keys()):
        data = all_data[t_max]
        if tag in data and data[tag]['values']:
            steps = data[tag]['steps']
            values = data[tag]['values']
            label = f'$t_{{max}}$ = {t_max}'
            ax.plot(steps, values, label=label, color=colors.get(t_max, 'gray'),
                   linewidth=2 if t_max == 0 else 1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Validation loss
    ax = axes[1]
    tag = 'loss/val_epoch'
    for t_max in sorted(all_data.keys()):
        data = all_data[t_max]
        if tag in data and data[tag]['values']:
            steps = data[tag]['steps']
            values = data[tag]['values']
            label = f'$t_{{max}}$ = {t_max}'
            ax.plot(steps, values, label=label, color=colors.get(t_max, 'gray'),
                   linewidth=2 if t_max == 0 else 1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.suptitle('Ambient Diffusion for Matcha TTS: $t_{max}$ Ablation', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'combined_comparison.png'), dpi=150, bbox_inches='tight')
    print("Saved: combined_comparison.png")
    plt.close(fig)


def compute_statistics(all_data):
    """Compute summary statistics for each run."""
    stats = {}
    
    for t_max, data in all_data.items():
        run_stats = {}
        
        # Final losses
        for tag in ['loss/train_epoch', 'loss/val_epoch']:
            if tag in data and data[tag]['values']:
                values = data[tag]['values']
                run_stats[f'{tag}_final'] = values[-1]
                run_stats[f'{tag}_min'] = min(values)
                run_stats[f'{tag}_mean'] = sum(values) / len(values)
                
                # Compute variance over last 20 epochs
                if len(values) >= 20:
                    last_20 = values[-20:]
                    mean = sum(last_20) / len(last_20)
                    variance = sum((x - mean) ** 2 for x in last_20) / len(last_20)
                    run_stats[f'{tag}_final_std'] = variance ** 0.5
        
        stats[t_max] = run_stats
    
    return stats


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'analysis_output')
    
    print("=" * 60)
    print("Ambient Diffusion for Matcha TTS - Experiment Analysis")
    print("=" * 60)
    
    # Extract data
    all_data = extract_all_data(base_dir)
    
    if not all_data:
        print("\nNo data extracted! Check if TensorBoard event files exist.")
        return
    
    # Save raw data
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to JSON-serializable format
    json_data = {}
    for t_max, data in all_data.items():
        json_data[str(t_max)] = data
    
    with open(os.path.join(output_dir, 'raw_data.json'), 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"\nRaw data saved to: {os.path.join(output_dir, 'raw_data.json')}")
    
    # Compute statistics
    stats = compute_statistics(all_data)
    
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    print("\nFinal Validation Loss (per t_max):")
    print("-" * 40)
    for t_max in sorted(stats.keys()):
        s = stats[t_max]
        val_loss = s.get('loss/val_epoch_final', 'N/A')
        train_loss = s.get('loss/train_epoch_final', 'N/A')
        if isinstance(val_loss, float):
            print(f"  t_max = {t_max:4.2f}: Val = {val_loss:.4f}, Train = {train_loss:.4f}")
        else:
            print(f"  t_max = {t_max:4.2f}: {val_loss}")
    
    # Save statistics
    stats_json = {str(k): v for k, v in stats.items()}
    with open(os.path.join(output_dir, 'statistics.json'), 'w') as f:
        json.dump(stats_json, f, indent=2)
    print(f"\nStatistics saved to: {os.path.join(output_dir, 'statistics.json')}")
    
    # Create plots
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)
    create_comparison_plots(all_data, output_dir)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()

