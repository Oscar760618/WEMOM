import pretty_midi
import numpy as np

class MusicTheoryFilterMonophonic:
    def __init__(self):
        pass

    def _snap_to_scale(self, note_number, scale_intervals, root=0):
        pitch_class = note_number % 12
        octave = note_number // 12
        rel_pitch = (pitch_class - root) % 12
        if rel_pitch in scale_intervals: return note_number
        min_dist = 100
        best_pitch = rel_pitch
        for p in scale_intervals:
            dist = abs(p - rel_pitch)
            if dist < min_dist:
                min_dist = dist
                best_pitch = p
        new_pitch_class = (best_pitch + root) % 12
        return octave * 12 + new_pitch_class

    def apply_constraints(self, input_midi_path, output_midi_path, valence=0.0, arousal=0.0, force_scale="Auto", force_velocity=None, force_grid=None):
        try:
            pm = pretty_midi.PrettyMIDI(input_midi_path)
        except: return

        print(f"--- Applying V5 Monophonic Simple Melody Filter ---")

        # 0. 基于情绪生成基础曲速
        computed_bpm = 100 + arousal * 30 
        bpm = max(80, min(120, computed_bpm))

        if force_scale == "C Major" or (force_scale == "Auto" and valence >= 0): scale_intervals = [0, 2, 4, 5, 7, 9, 11]
        elif force_scale == "C Minor" or (force_scale == "Auto" and valence < 0): scale_intervals = [0, 2, 3, 5, 7, 8, 10]
        else: scale_intervals = list(range(12))

        # 固定网格：我们以 1/8 音符为最小跳动单位（非常工整的滴答声）
        beat_duration = 60.0 / bpm
        eighth_duration = beat_duration / 2.0

        cleaned_pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        ts = pretty_midi.TimeSignature(4, 4, 0.0)
        cleaned_pm.time_signature_changes.append(ts)
        
        for instrument in pm.instruments:
            if instrument.is_drum: continue
            
            instrument.notes.sort(key=lambda x: x.start)
            
            # 消前缀空白
            if len(instrument.notes) > 0:
                first_start = instrument.notes[0].start
                for n in instrument.notes:
                    n.start -= first_start
                    n.end -= first_start

            # 【核心1：网格化归组】
            grid_to_notes = {}
            for note in instrument.notes:
                if note.end - note.start < 0.05: continue 
                # 强行把所有音压到最近的 1/8 拍格子上
                grid_idx = round(note.start / eighth_duration)
                if grid_idx not in grid_to_notes:
                    grid_to_notes[grid_idx] = []
                grid_to_notes[grid_idx].append(note)

            final_notes = []
            grid_indices = sorted(grid_to_notes.keys())
            
            # 【核心2：提纯单旋律（Monophonic削和弦）】
            for grid_idx in grid_indices:
                notes_at_grid = grid_to_notes[grid_idx]
                
                # 同一个时刻如果有多个音（和弦），我们只保留音高最高的那一个（通常是主旋律）！
                # 彻底去除你截图里那些叠在一起的柱状和弦
                highest_note = max(notes_at_grid, key=lambda x: x.pitch)
                
                # 吸附音阶
                new_pitch = self._snap_to_scale(highest_note.pitch, scale_intervals, root=0)
                
                # 重新计算物理时间
                start_time = grid_idx * eighth_duration
                # 长度固定为：占满该网格的 90%（留 10% 空隙产生颗粒感节奏）
                end_time = start_time + (eighth_duration * 0.9)
                
                # 极其简单的力度：强拍 90，弱拍 70
                is_downbeat = (grid_idx % 2 == 0)
                vel = 90 if is_downbeat else 70

                final_notes.append(pretty_midi.Note(
                    velocity=vel, pitch=new_pitch, start=start_time, end=end_time
                ))

            # 写入单通道纯净乐谱
            cleaned_inst = pretty_midi.Instrument(program=0)
            cleaned_inst.notes = final_notes
            cleaned_pm.instruments.append(cleaned_inst)

        cleaned_pm.write(output_midi_path)
        print(f"-> V5 Monophonic MIDI saved to {output_midi_path}")

