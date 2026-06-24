import pretty_midi
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import matplotlib.colors as mcolors

def plot_piano_roll(midi_path, output_png_path=None):
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"Error loading {midi_path}: {e}")
        return

    plt.figure(figsize=(14, 6))
    
    out_path = output_png_path or midi_path.replace('.mid', '.png')
    
    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
            
        # Extract features for color mapping
        velocities = [n.velocity for n in instrument.notes]
        if not velocities:
            continue
            
        # Plot notes
        for note in instrument.notes:
            # 颜色透明度反映了力度 (Velocity)，越重的音颜色越深
            alpha = max(0.2, note.velocity / 127.0)
            plt.plot([note.start, note.end], [note.pitch, note.pitch], 
                     color='b', linewidth=5, alpha=alpha, solid_capstyle='round')

    plt.title(f"Piano Roll Analysis: {os.path.basename(midi_path)}\nDarker/thicker lines = heavier velocity/accents")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Pitch (MIDI Note Number)")
    
    # 绘制背景网格，模拟五线谱/钢琴卷帘
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.grid(axis='x', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"-> Visualization saved to: {out_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        plot_piano_roll(sys.argv[1])
    else:
        print("Usage: python visualize_midi.py <path_to_midi_file>")
