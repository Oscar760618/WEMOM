import pretty_midi
import numpy as np
from collections import defaultdict

class MusicTheoryFilterCrispClean:
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

        print(f"--- Applying V6 Crisp Clean (2-Part) Filter ---")

        # 0. 基于情绪生成基础曲速
        computed_bpm = 100 + arousal * 30 
        bpm = max(80, min(120, computed_bpm))

        if force_scale == "C Major" or (force_scale == "Auto" and valence >= 0): scale_intervals = [0, 2, 4, 5, 7, 9, 11]
        elif force_scale == "C Minor" or (force_scale == "Auto" and valence < 0): scale_intervals = [0, 2, 3, 5, 7, 8, 10]
        else: scale_intervals = list(range(12))

        # 采用16分音符网格，既能表现一点节奏变化，又非常工整
        beat_duration = 60.0 / bpm
        sixteenth_duration = beat_duration / 4.0

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

            # 【核心1：网格化与和弦按组折叠】
            grid_to_notes = defaultdict(list)
            for note in instrument.notes:
                if note.end - note.start < 0.05: continue 
                
                # 强行吸附到 16 分音符网格上
                grid_idx = round(note.start / sixteenth_duration)
                grid_to_notes[grid_idx].append(note)

            final_notes = []
            grid_indices = sorted(grid_to_notes.keys())
            
            # 【核心2：抽离“双声部”（最高音主旋律 + 最低音打底），去泥泞】
            for i, grid_idx in enumerate(grid_indices):
                notes_at_grid = grid_to_notes[grid_idx]
                notes_at_grid.sort(key=lambda x: x.pitch)
                
                # 无论这个时刻模型生成了多少个音符叠加，我们最多只保留2个
                # (1) 最高音：担当旋律
                # (2) 最低音：担当和弦结构/底盘 (如果和最高音不是一回事的话)
                top_note = notes_at_grid[-1]
                selected_notes = [top_note]
                
                if len(notes_at_grid) > 1:
                    bottom_note = notes_at_grid[0]
                    # 确保底音不要离最高音太近（至少隔开 5 个半音），不然依然拥挤
                    if top_note.pitch - bottom_note.pitch > 5:
                        selected_notes.append(bottom_note)
                
                start_time = grid_idx * sixteenth_duration
                
                # 【核心3：干脆利落的断奏发音 (Crisp Articulation)】
                # 寻找下一个音符的发出时间，确保这个音符在下一个音符前收手，绝不粘滞拖音
                next_start_time = start_time + beat_duration # 默认给一拍长
                if i + 1 < len(grid_indices):
                    next_start_time = grid_indices[i+1] * sixteenth_duration
                
                # 音符长度只占空隙的 85%（制造你在电子琴上听到的那种清晰断奏感）
                # 设定硬上限不超过1秒，防止极其冗长的拖音
                max_duration = min((next_start_time - start_time) * 0.85, 1.0)
                end_time = start_time + max(0.1, max_duration)

                # 处理力度的情绪化：底音弱，主音强
                for sn in selected_notes:
                    new_pitch = self._snap_to_scale(sn.pitch, scale_intervals, root=0)
                    
                    if sn == top_note:
                        vel = int(80 + arousal * 15)  # 主旋律稍微突出致敬情感
                    else:
                        vel = int(60 + arousal * 10)  # 底音非常克制
                        
                    is_downbeat = (grid_idx % 4 == 0)
                    if is_downbeat: vel += 10 # 仅在重拍稍微加强一点点
                    
                    vel = max(40, min(100, vel))

                    final_notes.append(pretty_midi.Note(
                        velocity=vel, pitch=new_pitch, start=start_time, end=end_time
                    ))

            # 写入乐谱
            final_notes.sort(key=lambda x: x.start)
            cleaned_inst = pretty_midi.Instrument(program=0)
            cleaned_inst.notes = final_notes
            cleaned_pm.instruments.append(cleaned_inst)

        cleaned_pm.write(output_midi_path)
        print(f"-> V6 Crisp Clean (2-Part) MIDI saved to {output_midi_path}")

