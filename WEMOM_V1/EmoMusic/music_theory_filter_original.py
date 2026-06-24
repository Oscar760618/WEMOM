import pretty_midi
import numpy as np

class MusicTheoryFilterOriginal:
    def __init__(self, quantization_step=0.25):
        """
        Original vanilla filter, strictly utilizing Valence and Arousal.
        No user override parameters (force_scale, force_grid, etc).
        """
        self.quantization_step = quantization_step

    def _snap_to_scale(self, note_number, scale_intervals, root=0):
        pitch_class = note_number % 12
        octave = note_number // 12
        rel_pitch = (pitch_class - root) % 12
        
        if rel_pitch in scale_intervals:
            return note_number
        
        min_dist = 100
        best_pitch = rel_pitch
        for p in scale_intervals:
            dist = abs(p - rel_pitch)
            if dist < min_dist:
                min_dist = dist
                best_pitch = p
                
        new_pitch_class = (best_pitch + root) % 12
        return octave * 12 + new_pitch_class

    def apply_constraints(self, input_midi_path, output_midi_path, valence=0.0, arousal=0.0):
        try:
            pm = pretty_midi.PrettyMIDI(input_midi_path)
        except Exception as e:
            print(f"Error loading MIDI for filtering: {e}")
            return

        print(f"--- Applying ORIGINAL Pure Rule-based Filter (V={valence:.2f}, A={arousal:.2f}) ---")

        # ---------------------------
        # 1. Emotion (V/A) -> Rules Map
        # ---------------------------
        
        # A. Valence -> Scale Major/Minor
        if valence >= 0:
            scale_intervals = [0, 2, 4, 5, 7, 9, 11] # C Major for positive emotion
        else:
            scale_intervals = [0, 2, 3, 5, 7, 8, 10] # C Minor for negative emotion

        # B. Arousal -> Dynamics & Rhythm
        # 基础力度计算：高唤醒度 -> 大音量；低唤醒度 -> 小音量
        target_velocity_base = 80 + (arousal * 30)
        target_velocity_base = max(40, min(110, target_velocity_base))
        
        # 音符长度过滤阈值：低唤醒度会滤除所有碎碎的/急促的短音，保留长音；高唤醒度保留微小急促音
        min_duration = 0.05 if arousal > 0 else 0.15 

        # ---------------------------
        # 2. Modify MIdI
        # ---------------------------
        for instrument in pm.instruments:
            if instrument.is_drum: continue
            
            cleaned_notes = []
            
            for note in instrument.notes:
                duration = note.end - note.start
                
                # 过滤掉不符合当前情绪长度要求的杂音
                if duration < min_duration:
                    continue

                # 音高的调性修正（对齐大小调）
                note.pitch = self._snap_to_scale(note.pitch, scale_intervals, root=0)

                # 力度的自动赋予（加入微小的人性化自然波动）
                noise = np.random.randint(-8, 8) 
                note.velocity = int(max(20, min(127, target_velocity_base + noise)))

                cleaned_notes.append(note)

            instrument.notes = cleaned_notes

        # Save
        pm.write(output_midi_path)
        print(f"-> Original Constrained MIDI saved to {output_midi_path}")

    # 对于之前某些老代码直接调用的兼容
    def apply(self, input_path, output_path, valence, arousal):
        self.apply_constraints(input_path, output_path, valence, arousal)
