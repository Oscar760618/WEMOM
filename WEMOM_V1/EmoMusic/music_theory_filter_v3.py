import pretty_midi
import numpy as np

class MusicTheoryFilterRhythm:
    def __init__(self):
        pass

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

    def apply_constraints(self, input_midi_path, output_midi_path, valence=0.0, arousal=0.0, force_scale="Auto", force_velocity=None, force_grid=None):
        try:
            pm = pretty_midi.PrettyMIDI(input_midi_path)
        except Exception as e:
            print(f"Error loading MIDI for filtering: {e}")
            return

        print(f"--- Applying Rhythm & Groove Filter (V={valence:.2f}, A={arousal:.2f}) ---")
        
        # 0. 节奏参数预设 (Rhythm Parameters)
        # BPM 决定速度。越兴奋 (arousal > 0) 速度越快 (120-130)；越伤心速度越慢 (80-90)。
        if force_grid == "Auto" or force_grid is None:
            computed_bpm = 100 + arousal * 30  # arousal: -1 to 1 => BPM 70 to 130
            bpm = max(70, min(140, computed_bpm))
        else:
            bpm = 110 # default fallback
            
        print(f"--- Decided Groove BPM: {bpm:.1f} ---")
        
        # 1. 音阶约束 (Scale)
        if force_scale == "C Major" or (force_scale == "Auto" and valence >= 0):
            scale_intervals = [0, 2, 4, 5, 7, 9, 11] # C Major
        elif force_scale == "C Minor" or (force_scale == "Auto" and valence < 0):
            scale_intervals = [0, 2, 3, 5, 7, 8, 10] # C Minor
        else:
            scale_intervals = list(range(12)) # Original

        time_sig = (4, 4) # 默认强制设定为 4/4 拍
        beat_duration = 60.0 / bpm          # 1/4 音符时长 (Quarter note)
        sixteenth_duration = beat_duration / 4.0 # 最小网格：1/16音符
        measure_duration = beat_duration * time_sig[0] # 4拍的总时长

        # 基础力度 (Velocity)
        base_vel = 80 + (arousal * 20) 
        base_vel = max(50, min(100, base_vel))

        # 重新构建带有官方拍号的 MIDI
        cleaned_pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        # 注入 4/4 拍的 Time Signature (可以方便后期导入 DAW 等宿主软件观看)
        ts = pretty_midi.TimeSignature(4, 4, 0.0)
        cleaned_pm.time_signature_changes.append(ts)
        
        for instrument in pm.instruments:
            if instrument.is_drum: continue
            
            cleaned_notes = []
            instrument.notes.sort(key=lambda x: x.start)
            
            # 【革新0：消除空白，对齐原点】
            if len(instrument.notes) > 0:
                first_start = instrument.notes[0].start
                for n in instrument.notes:
                    n.start -= first_start
                    n.end -= first_start

            for note in instrument.notes:
                if note.end - note.start < 0.05: continue 
                
                # 【革新1：纯粹的节拍器量化 (Strict Quantization)】
                # 我们不再做软吸附，而是直接把音符强制吸附到最近的 16 分音符网格上
                closest_16th_step = round(note.start / sixteenth_duration)
                quantized_start = closest_16th_step * sixteenth_duration
                note.start = quantized_start
                
                # 计算这个音符在当前小节（Measure）内的拍子位置（0拍、1拍、2拍等）
                position_in_measure = note.start % measure_duration
                beat_index = position_in_measure / beat_duration
                
                # 【革新2：灌入 4/4 拍律动灵魂 (Groove Injection)】
                # 4/4 拍的强弱次序：强、弱、次强、弱。
                if abs(beat_index - 0.0) < 0.1:     # 第一拍 (强) Downbeat
                    dyn_offset = 20
                elif abs(beat_index - 2.0) < 0.1:   # 第三拍 (次强) Backbeat
                    dyn_offset = 12
                elif abs(beat_index - 1.0) < 0.1 or abs(beat_index - 3.0) < 0.1: # 二四拍 (弱)
                    dyn_offset = -10
                else: # 反拍/切分音 (Off-beats, 16分音符等细碎音)
                    dyn_offset = -20
                    
                note.velocity = int(max(30, min(127, base_vel + dyn_offset + np.random.normal(0, 3))))

                # 合谐调性
                note.pitch = self._snap_to_scale(note.pitch, scale_intervals, root=0)

                cleaned_notes.append(note)
                
            cleaned_notes.sort(key=lambda x: x.start)
            
            # 【革新3：无缝连奏/延音 (Seamless Legato)】
            # 解决以往的“断奏、不连贯”现象。自动拉长前一个音，直到碰上后一个音的开头。
            final_notes = []
            for i in range(len(cleaned_notes)):
                current_note = cleaned_notes[i]
                
                # 向后寻找到下一个发生的动作的时间点
                next_start = current_note.end
                for j in range(i+1, len(cleaned_notes)):
                    if cleaned_notes[j].start > current_note.start + 0.01:
                        next_start = cleaned_notes[j].start
                        break
                
                # 如果这个音很短并且离下一个音很近，就延长它（踩下延音踏板）
                # 限制最大延音长度为两拍，以免糊成一片
                max_extend = current_note.start + (beat_duration * 2.0)
                # 将音符向右延长到补齐所有的空缺
                current_note.end = min(max_extend, max(current_note.end, next_start))
                
                # 【革新4：重拍结构重塑（左手低音八度补齐）】
                # 在强弱拍上增补低音，让音乐在声学上具有厚重感和方向感
                is_chord_root = True
                for prev in final_notes:
                    if abs(prev.start - current_note.start) < 0.05 and prev.pitch < current_note.pitch:
                        is_chord_root = False
                        break
                
                if is_chord_root and current_note.pitch > 45:
                    # 我们只在整数拍（0拍, 1拍, 2拍, 3拍）上自动为它增加低音厚度
                    pos_in_meas = current_note.start % measure_duration
                    b_idx = pos_in_meas / beat_duration
                    if abs(b_idx - round(b_idx)) < 0.1:
                        bass_pitch = max(21, current_note.pitch - 12)
                        bass_note = pretty_midi.Note(
                            velocity=max(40, int(current_note.velocity * 0.85)), 
                            pitch=bass_pitch, 
                            start=current_note.start, 
                            end=current_note.end 
                        )
                        final_notes.append(bass_note)
                        
                final_notes.append(current_note)

            # 主音收束
            if final_notes:
                final_notes.sort(key=lambda x: x.start)
                last_note = final_notes[-1]
                octave = last_note.pitch // 12
                last_note.pitch = octave * 12 + 0 
                last_note.end += beat_duration * 2 # 最后一拍长一点

            cleaned_inst = pretty_midi.Instrument(program=0) # 0 is Acoustic Grand Piano
            cleaned_inst.notes = final_notes
            cleaned_pm.instruments.append(cleaned_inst)

        cleaned_pm.write(output_midi_path)
        print(f"-> Rhythm Pro Constrained MIDI saved to {output_midi_path}")

