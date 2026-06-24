import pretty_midi
import numpy as np

class MusicTheoryFilterSmartRhythm:
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

        # 0. 根据 Arousal 智能匹配曲速
        computed_bpm = 100 + arousal * 30 
        bpm = max(75, min(130, computed_bpm))

        if force_scale == "C Major" or (force_scale == "Auto" and valence >= 0): scale_intervals = [0, 2, 4, 5, 7, 9, 11]
        elif force_scale == "C Minor" or (force_scale == "Auto" and valence < 0): scale_intervals = [0, 2, 3, 5, 7, 8, 10]
        else: scale_intervals = list(range(12))

        time_sig = (4, 4)
        beat_duration = 60.0 / bpm
        sixteenth_duration = beat_duration / 4.0
        measure_duration = beat_duration * time_sig[0]

        # 基础力度调整：舒缓的音乐力度低，欢快的力度高
        base_vel = 75 + (arousal * 15) 
        base_vel = max(50, min(95, base_vel))

        cleaned_pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        ts = pretty_midi.TimeSignature(4, 4, 0.0)
        cleaned_pm.time_signature_changes.append(ts)
        
        for instrument in pm.instruments:
            if instrument.is_drum: continue
            
            cleaned_notes = []
            instrument.notes.sort(key=lambda x: x.start)
            
            if len(instrument.notes) > 0:
                first_start = instrument.notes[0].start
                for n in instrument.notes:
                    n.start -= first_start
                    n.end -= first_start

            # 分组处理同时落下的和弦音符，为了避免和弦太重
            from collections import defaultdict
            chord_groups = defaultdict(list)

            for note in instrument.notes:
                if note.end - note.start < 0.05: continue 
                
                # 【革新1：弹性吸附 (Humanized Grid)】 
                # 不再死板地 100% 对齐，而是 80% 靠近网格，保留 20% 的人性微小偏差 (避免呆板的机器感)
                closest_16th_step = round(note.start / sixteenth_duration)
                quantized_start = closest_16th_step * sixteenth_duration
                humanized_start = (quantized_start * 0.8) + (note.start * 0.2)
                note.start = humanized_start
                
                # 吸附到调性
                note.pitch = self._snap_to_scale(note.pitch, scale_intervals, root=0)
                
                # 将差不多同一时间（<50ms）响起的音符归类为一个和弦组
                chord_time_key = round(note.start, 1)
                chord_groups[chord_time_key].append(note)

            for t_key, chord_notes in chord_groups.items():
                chord_notes.sort(key=lambda x: x.pitch)
                
                pos_in_measure = chord_notes[0].start % measure_duration
                beat_index = pos_in_measure / beat_duration
                
                # 【革新2：轻柔的 4/4 律动 (Soft Groove)】 减轻重音力度
                if abs(beat_index - 0.0) < 0.1: dyn_offset = 12       # 第1拍：轻微重音
                elif abs(beat_index - 2.0) < 0.1: dyn_offset = 5      # 第3拍：次重音
                elif abs(beat_index - 1.0) < 0.1 or abs(beat_index - 3.0) < 0.1: dyn_offset = -5 # 2、4拍：偏弱
                else: dyn_offset = -15 # 反拍、碎音：极弱
                
                # 【革新3：智能和弦稀释 (Chord Voicing Polish)】 
                # 解决 S3, S4 听起来“非常乱、重音互相交杂”的问题。
                # 当弹下一大把音符时，只有最高音（主旋律）和最低音（根音）保持原力度，中间的填充音力度大幅降低
                top_note_idx = len(chord_notes) - 1
                bottom_note_idx = 0
                
                for idx, note in enumerate(chord_notes):
                    inner_voice_penalty = 0
                    if len(chord_notes) > 2 and (idx != top_note_idx and idx != bottom_note_idx):
                        inner_voice_penalty = -20 # 内部和弦音变得虚无飘渺，不再轰炸耳朵
                    
                    target_vel = base_vel + dyn_offset + inner_voice_penalty + np.random.normal(0, 3)
                    note.velocity = int(max(30, min(110, target_vel))) # 强制上限 110，防止砸钢琴
                    cleaned_notes.append(note)
                    
                    # 【革新4：克制的低音下潜】 只有落在第1拍的正拍上，且是和弦根音，我们才给它复制低音八度
                    if idx == bottom_note_idx and note.pitch > 45 and abs(beat_index - 0.0) < 0.1:
                        bass_note = pretty_midi.Note(
                            velocity=int(note.velocity * 0.7), 
                            pitch=max(21, note.pitch - 12), 
                            start=note.start, end=note.end 
                        )
                        cleaned_notes.append(bass_note)

            cleaned_notes.sort(key=lambda x: x.start)
            
            # 【革新5：无缝连奏 (Legato)】
            final_notes = []
            for i in range(len(cleaned_notes)):
                current_note = cleaned_notes[i]
                next_start = current_note.end
                for j in range(i+1, len(cleaned_notes)):
                    if cleaned_notes[j].start > current_note.start + 0.01:
                        next_start = cleaned_notes[j].start
                        break
                max_extend = current_note.start + (beat_duration * 2.0)
                current_note.end = min(max_extend, max(current_note.end, next_start))
                final_notes.append(current_note)

            if final_notes:
                final_notes.sort(key=lambda x: x.start)
                last_note = final_notes[-1]
                last_note.pitch = (last_note.pitch // 12) * 12 + 0 
                last_note.end += beat_duration * 1.5 

            cleaned_inst = pretty_midi.Instrument(program=0)
            cleaned_inst.notes = final_notes
            cleaned_pm.instruments.append(cleaned_inst)

        cleaned_pm.write(output_midi_path)
        print(f"-> Smart Rhythm MIDI saved to {output_midi_path}")

