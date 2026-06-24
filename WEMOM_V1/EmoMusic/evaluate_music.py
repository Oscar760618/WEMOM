import pretty_midi
import numpy as np
import argparse
import os

def evaluate_midi(midi_path, target_scale=None):
    if not os.path.exists(midi_path):
        print(f"File not found: {midi_path}")
        return None

    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"Could not load MIDI {midi_path}: {e}")
        return None
        
    total_notes = 0
    total_duration = 0.0
    velocities = []
    pitches = []
    
    # Calculate scale consistency
    # C Major: 0, 2, 4, 5, 7, 9, 11
    # C Minor: 0, 2, 3, 5, 7, 8, 10
    c_major_intervals = {0, 2, 4, 5, 7, 9, 11}
    c_minor_intervals = {0, 2, 3, 5, 7, 8, 10}
    
    in_c_major = 0
    in_c_minor = 0
    
    end_time = 0.0
    start_time = float('inf')

    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
            
        for note in instrument.notes:
            total_notes += 1
            dur = note.end - note.start
            total_duration += dur
            velocities.append(note.velocity)
            pitches.append(note.pitch)
            
            pc = note.pitch % 12
            if pc in c_major_intervals:
                in_c_major += 1
            if pc in c_minor_intervals:
                in_c_minor += 1
                
            if note.start < start_time: start_time = note.start
            if note.end > end_time: end_time = note.end

    if total_notes == 0:
        print("No valid notes found in the MIDI.")
        return None

    total_length_sec = end_time - start_time
    avg_velocity = np.mean(velocities)
    notes_per_sec = total_notes / total_length_sec if total_length_sec > 0 else 0
    avg_duration = total_duration / total_notes
    pitch_min = np.min(pitches)
    pitch_max = np.max(pitches)
    
    c_major_ratio = in_c_major / total_notes
    c_minor_ratio = in_c_minor / total_notes

    print(f"=== MIDI Playback Evaluation ===")
    print(f"File: {os.path.basename(midi_path)}")
    print(f"Total Length (sec)   : {total_length_sec:.2f}")
    print(f"Total Notes          : {total_notes}")
    print(f"Density (notes/sec)  : {notes_per_sec:.2f} notes/s (Reflects Rhythm/Density constraints)")
    print(f"Avg Note Duration    : {avg_duration:.3f} s (Reflects Grids/Quantization)")
    print(f"Avg Velocity         : {avg_velocity:.1f}  (Reflects Velocity constraints)")
    print(f"Pitch Min-Max        : {pitch_min} - {pitch_max}")
    print(f"C Major Strictness   : {c_major_ratio:.1%} in key (Reflects Scale force)")
    print(f"C Minor Strictness   : {c_minor_ratio:.1%} in key (Reflects Scale force)")
    print(f"================================\n")
    
    return {
        "notes": total_notes,
        "density": notes_per_sec,
        "duration": avg_duration,
        "velocity": avg_velocity,
        "c_maj_ratio": c_major_ratio,
        "c_min_ratio": c_minor_ratio
    }

def evaluate_diary_macro(mid_files, merged_mid_file):
    """
    Evaluates a full sequence of diary MIDI files to understand emotional flow,
    transition smoothness, and global structure.
    """
    densities, pitches_avg, cmaj_ratios = [], [], []
    start_pitches, end_pitches = [], []
    
    for mid_file in mid_files:
        try:
            eval_res = evaluate_midi(mid_file)
            if eval_res:
                densities.append(eval_res['density'])
                cmaj_ratios.append(eval_res['c_maj_ratio'])
                
            pm = pretty_midi.PrettyMIDI(mid_file)
            if pm.instruments and pm.instruments[0].notes:
                sorted_notes = sorted(pm.instruments[0].notes, key=lambda x: x.start)
                pitches_avg.append(np.mean([n.pitch for n in sorted_notes]))
                start_pitches.append(np.mean([n.pitch for n in sorted_notes[:3]]))
                end_pitches.append(np.mean([n.pitch for n in sorted_notes[-3:]]))
        except Exception as e:
            print(f"Error reading mid file {mid_file} for macro evaluation: {e}")
            
    # Macro Stats
    traj_density_var = np.std(densities) if len(densities) > 1 else 0
    traj_pitch_var = np.std(pitches_avg) if len(pitches_avg) > 1 else 0
    
    pitch_leaps = []
    for i in range(len(end_pitches)-1):
        leap = abs(end_pitches[i] - start_pitches[i+1])
        pitch_leaps.append(leap)
    smoothness_avg_leap = np.mean(pitch_leaps) if pitch_leaps else 0
    tonal_cohesion_var = np.std(cmaj_ratios) if len(cmaj_ratios) > 1 else 0
    
    # Evaluate the global merged file
    merged_eval = evaluate_midi(merged_mid_file)
    
    return {
        "traj_density_var": traj_density_var,
        "traj_pitch_var": traj_pitch_var,
        "smoothness_avg_leap": smoothness_avg_leap,
        "tonal_cohesion_var": tonal_cohesion_var,
        "merged_eval": merged_eval
    }

if __name__ == "__main__":
    import sys
    # For testing, you can pass original and constrained MIDI files.
    if len(sys.argv) > 1:
        for f in sys.argv[1:]:
            evaluate_midi(f)
    else:
        # Default evaluate the generated ones
        base_dir = "D:/Projects/URIS/URIS/WEMOM_V1/generated_midis"
        print("Evaluating Original Output (Before Constraints)...")
        evaluate_midi(os.path.join(base_dir, "temp_raw.mid"))
        
        print("Evaluating Constrained Output (After Constraints)...")
        evaluate_midi(os.path.join(base_dir, "generated_music.mid"))
