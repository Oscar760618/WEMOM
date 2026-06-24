import pretty_midi
import numpy as np

class MusicTheoryFilterSmooth:
    def __init__(self):
        # 不再强制量子化，而是采用柔性清理
        pass

    def _snap_to_scale(self, note_number, scale_intervals, root=0):
        """
        柔和地将音符吸附到目标音阶，防止出现不和谐的离调音（Dissonance）。
        """
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

        print(f"--- Applying Advanced Pro Filter (V={valence:.2f}, A={arousal:.2f}) ---")

        # 1. 音阶约束 (Scale)
        if force_scale == "C Major" or (force_scale == "Auto" and valence >= 0):
            scale_intervals = [0, 2, 4, 5, 7, 9, 11] # C Major
        elif force_scale == "C Minor" or (force_scale == "Auto" and valence < 0):
            scale_intervals = [0, 2, 3, 5, 7, 8, 10] # C Minor
        else:
            scale_intervals = list(range(12)) # Original

        # 2. 基础力度 (Velocity)
        if force_grid is not None and force_grid != "Auto":
            try: grid_sec = float(force_grid.split()[0]) * 0.5
            except: grid_sec = 0.25
        else:
            grid_sec = 0.5 if arousal < -0.2 else 0.25

        if force_velocity is not None and force_velocity != "Auto":
            try: base_vel = float(force_velocity.split()[0])
            except: base_vel = 80
        else:
            base_vel = 80 + (arousal * 20) 
        base_vel = max(50, min(100, base_vel))

        # 3. 开始清理与修正 MIDI
        for instrument in pm.instruments:
            if instrument.is_drum: continue
            
            cleaned_notes = []
            
            for note in instrument.notes:
                if note.end - note.start < 0.05: continue 

                # 【革新1：磁性节奏吸附 (Magnetic Groove)】
                # 不再死板地对齐网格，而是将原本零散的按键，拉扯近节拍点（70%拉扯力），保留30%的人性化微小错位
                nearest_grid = round(note.start / grid_sec) * grid_sec
                note.start = note.start * 0.3 + nearest_grid * 0.7
                
                # 【革新2：延音与连奏 (Legato)】
                # 解决“断断续续胡乱敲键”的感觉，人工稍微延长音符长度，使得声音互相衔接包裹
                note.end = max(note.end, note.start + grid_sec * 1.5)

                # 【革新3：和谐调性】
                note.pitch = self._snap_to_scale(note.pitch, scale_intervals, root=0)

                # 【革新4：重拍律动 (Groove)】
                # 正拍（强拍）音量大，反拍音量小，彻底改变机器人的死板力度
                is_strong_beat = abs(note.start % 0.5) < 0.05
                dyn_offset = 12 if is_strong_beat else -8
                note.velocity = int(max(30, min(115, base_vel + dyn_offset + np.random.normal(0, 5))))

                cleaned_notes.append(note)
                
            # 重新排序确保时序正确
            cleaned_notes.sort(key=lambda x: x.start)
            
            # 【革新5：智能左手低音补偿 (Bass Octave Doubling)】
            # AI通常只生成右手的主律动，导致声音非常单薄。这里自动捕捉和弦根音，向下复制一个八度。
            final_notes = []
            current_onset = -1.0
            chord_notes = []
            
            def process_chord(cn):
                if not cn: return []
                enhanced = list(cn)
                lowest = min(cn, key=lambda n: n.pitch)
                if lowest.pitch > 45: 
                    bass_pitch = max(21, lowest.pitch - 12)
                    bass_note = pretty_midi.Note(
                        velocity=max(40, int(lowest.velocity * 0.8)), 
                        pitch=bass_pitch, 
                        start=lowest.start, 
                        end=lowest.start + (grid_sec * 2.0) 
                    )
                    enhanced.append(bass_note)
                return enhanced

            for n in cleaned_notes:
                if abs(n.start - current_onset) < 0.05:
                    chord_notes.append(n)
                else:
                    final_notes.extend(process_chord(chord_notes))
                    chord_notes = [n]
                    current_onset = n.start
                    
            final_notes.extend(process_chord(chord_notes))
            instrument.notes = final_notes
            
            # 【革新7：渐弱收尾 (Fade-out) 与 主音收束 (Root Completion)】
            # 解决短音乐“戛然而止”的断裂感
            if instrument.notes:
                instrument.notes.sort(key=lambda x: x.start)
                max_time = max([n.end for n in instrument.notes])
                fade_duration = 2.0  # 最后 2 秒渐弱
                fade_start = max_time - fade_duration
                
                for n in instrument.notes:
                    if n.start > fade_start:
                        # 计算衰减比例，越靠后越小
                        ratio = 1.0 - ((n.start - fade_start) / fade_duration)
                        ratio = max(0.2, min(1.0, ratio)) # 保证最小声还能听到一点
                        n.velocity = int(n.velocity * ratio)
                
                # 主音收束：找最后一个音符，强制设为主音（C），并轻微延长，营造“完成感”
                last_note = instrument.notes[-1]
                octave = last_note.pitch // 12
                # 根据我们的 scale_intervals，如果强制大调小调，主音目前都是 0 (对齐C)
                # 为了听感稳定，我们把它强制归入当前八度的 C (root=0) 
                # (如果你传入了不是 C 的调，需要修改这里，但我们目前默认全转 C大调/小调)
                last_note.pitch = octave * 12 + 0 
                last_note.end = max_time + 1.5 # 拉长尾音叹息
                
            # 【革新6：注入真实的钢琴延音踏板 (Sustain Pedal CC64)】
# 【革新8：时间轴紧凑裁剪 (Silence Trimming)】
            # 解决单独句子前方或后方有大量空白导致生成的 wav 文件冗长的问题
            if instrument.notes:
                instrument.notes.sort(key=lambda x: x.start)
                first_note_start = instrument.notes[0].start
                
                # 平移所有音符和 CC (消除前置空白)
                for n in instrument.notes:
                    n.start = n.start - first_note_start
                    n.end = n.end - first_note_start
                
                for cc in instrument.control_changes:
                    cc.time = max(0.0, cc.time - first_note_start)
                    
                # 重新计算 max_time 以清理废弃的尾部 CC
                max_time = max([n.end for n in instrument.notes])
                instrument.control_changes = [cc for cc in instrument.control_changes if cc.time <= max_time + 1.0]
                
        # 【最终优化】：新建一个 MIDI 对象只保存有效部分，避免 magenta 隐藏的 EndOfTrack meta event!
        cleaned_pm = pretty_midi.PrettyMIDI()
        cleaned_inst = pretty_midi.Instrument(program=pm.instruments[0].program if pm.instruments else 0)
        if pm.instruments:
            cleaned_inst.notes = pm.instruments[0].notes
            cleaned_inst.control_changes = pm.instruments[0].control_changes
        cleaned_pm.instruments.append(cleaned_inst)

        # Save
        cleaned_pm.write(output_midi_path)
        print(f"-> Smooth Pro Constrained MIDI (Trimmed) saved to {output_midi_path}")
