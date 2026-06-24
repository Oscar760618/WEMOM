import sys
import os
import torch
import numpy as np
import json
import re
import shutil
import subprocess
import datetime
import pretty_midi

# --- Path Setup ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
EMOMUSIC_DIR = os.path.join(PARENT_DIR, 'EmoMusic')
TEXT2MUSIC_DIR = os.path.join(PARENT_DIR, 'Text2Music')
TEXT_PRED_DIR = os.path.join(PARENT_DIR, 'Text_Prediction')
TEXT_CHECKPOINT_PATH = os.path.join(TEXT_PRED_DIR, 'params', 'roberta_regression_hf_best')

sys.path.append(PARENT_DIR)
sys.path.append(EMOMUSIC_DIR)
sys.path.append(TEXT_PRED_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from EmoMusic.MusicVAE import MusicAttrRegGMVAE
from EmoMusic.MidiPerformanceEncoder import MidiPerformanceEncoder
from EmoMusic.music_theory_filter_v6 import MusicTheoryFilterCrispClean
from EmoMusic.evaluate_music import evaluate_midi, evaluate_diary_macro
from transformers import AutoTokenizer, RobertaConfig
from Text_Prediction.Roberta import RobertaForSequenceClassificationSig
from EmoText.TextVAE import TextVAE
from Text2Music.model import CLloss
import Text2Music.config as cl_config

# --- Config & Constants ---
MUSIC_CHECKPOINT_PATH = os.path.join(EMOMUSIC_DIR, 'params', 'MusicVAE_50.pt')
MUSIC_CONFIG_PATH = os.path.join(EMOMUSIC_DIR, 'MusicVAE.json')
TEXTVAE_CHECKPOINT_PATH = os.path.join(PARENT_DIR, 'EmoText', 'params', 'TextVAE_latest.pt')
TEXTVAE_CONFIG_PATH = os.path.join(PARENT_DIR, 'EmoText', 'TextVAE_config.json')
LANG_DICT_PATH = os.path.join(PARENT_DIR, 'Data', 'Saves', 'id2word_text.npy')
CL_CHECKPOINT_PATH = os.path.join(TEXT2MUSIC_DIR, 'params', 'params_1000_6.167.pt')

FLUIDSYNTH_BIN = os.path.join(PARENT_DIR, 'fluidsynth-2.4.7-winXP-x86', 'bin', 'fluidsynth.exe')
SF2_PATH = os.path.join(PARENT_DIR, 'FluidR3_GM', 'FluidR3_GM.sf2')

class DiaryEngine:
    def __init__(self):
        print(">>> [Init] Loading Models into Memory...")
        # 1. Text Pred
        self.tokenizer = AutoTokenizer.from_pretrained(TEXT_CHECKPOINT_PATH)
        config = RobertaConfig.from_pretrained(TEXT_CHECKPOINT_PATH)
        self.text_pred_model = RobertaForSequenceClassificationSig.from_pretrained(TEXT_CHECKPOINT_PATH, config=config).to(device).eval()

        # 2. TextVAE
        with open(TEXTVAE_CONFIG_PATH) as f: t_args = json.load(f)
        self.text_vae = TextVAE(vocab_size=t_args['vocab_size'], embed_size=t_args['embedding_size'], hidden_size=t_args['hidden_size'], num_layers=t_args['num_layers'], dropout=t_args['dropout']).to(device)
        self.text_vae.load_state_dict(torch.load(TEXTVAE_CHECKPOINT_PATH, map_location=device)['model_state_dict'])
        self.text_vae.eval()

        # Dict
        id2word = np.load(LANG_DICT_PATH, allow_pickle=True).item()
        self.word2id = {v: k for k, v in id2word.items()}

        # 3. CL Net
        self.cl_net = CLloss(txt_dim=cl_config.TXT_DIM, mus_dim=cl_config.MUS_DIM).to(device)
        cl_checkpoint = torch.load(CL_CHECKPOINT_PATH, map_location=device)
        self.cl_net.load_state_dict(cl_checkpoint['model_state_dict'] if 'model_state_dict' in cl_checkpoint else cl_checkpoint)
        self.cl_net.eval()

        # 4. MusicVAE
        with open(MUSIC_CONFIG_PATH) as f: m_args = json.load(f)
        self.music_vae = MusicAttrRegGMVAE(
            roll_dims=342, rhythm_dims=3, note_dims=16, chroma_dims=24, dynamic_dims=5, chord_dims=24,
            hidden_dims=m_args['hidden_dim'], z_dims=m_args['z_dim'], n_step=m_args['time_step'], n_component=m_args['num_clusters']
        ).to(device)
        m_ckpt = torch.load(MUSIC_CHECKPOINT_PATH, map_location=device)
        self.music_vae.load_state_dict(m_ckpt['model_state_dict'] if 'model_state_dict' in m_ckpt else m_ckpt)
        self.music_vae.eval()

        self.mpe = MidiPerformanceEncoder(steps_per_second=100, num_velocity_bins=64, min_pitch=21, max_pitch=108, add_eos=True)
        # 换用 V6 折中极简滤镜（双声部 + 电子琴干脆断奏）
        self.mtf = MusicTheoryFilterCrispClean()
        self.filter_version_info = (
            "Filter Version: V6 (Crisp Clean 2-Part / 双声部电子琴清脆风格)\n"
            "特点: 针对乱敲和弦进行折中优化。\n"
            "1. 取消了死板的纯单音旋律，最多允许保留【最高音(主旋律)】和【最低音(和弦根音)】，剔除中间导致听觉粘稠的填充浑浊音。\n"
            "2. 取消了延音踏板连奏，音轨长度限制在下一个音符出现前的 85%，形成干脆利落（Staccato/Crisp）的“电子琴跳动版”听感。\n"
            "3. 力度情绪化：依然维持文字的情感（Arousal控制力度节奏），但不会有过分夸张的爆音砸琴。\n"
        )
        print(">>> [Init] All Models Loaded!\n")

        # State memory for smooth continuation
        self.previous_latent = None
        self.current_diary_dir = None
        self.current_diary_id = 1
        self.sentence_count = 0
        self.generated_midis = []
        self.last_midi_path = None
        self.last_eval = None
        self.last_macro = None

    def start_new_diary(self, log_path):
        """Prepare folder and log for a new diary entry containing multiple sentences."""
        dir_id = 1
        diary_base_dir = os.path.join(PARENT_DIR, 'generated_midis', 'diary')
        os.makedirs(diary_base_dir, exist_ok=True)
        
        while os.path.exists(os.path.join(diary_base_dir, f'diary_{dir_id}')):
            dir_id += 1
            
        self.current_diary_dir = os.path.join(diary_base_dir, f'diary_{dir_id}')
        os.makedirs(self.current_diary_dir, exist_ok=True)
        self.current_diary_id = dir_id
        self.sentence_count = 0
        self.previous_latent = None
        self.generated_midis = []
        
        # 写入本次使用的 Filter Version Info 供用户记录
        if hasattr(self, 'filter_version_info'):
            info_path = os.path.join(self.current_diary_dir, 'filter_version_info.txt')
            with open(info_path, 'w', encoding='utf-8') as f:
                f.write(self.filter_version_info)
        
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"馃摉 DIARY SESSION STARTED: Diary {dir_id}\n")
            f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*60}\n")

    def process_text_to_latent(self, text, max_len=30):
        token_list = re.findall(r"[\w']+|[.,!?;]", text.lower())
        input_ids = [self.word2id.get(token, self.word2id.get('<unk>', 1)) for token in token_list]
        input_ids += [0] * max(0, max_len - len(input_ids))
        return torch.tensor([input_ids[:max_len]], dtype=torch.long).to(device)

    def midi_to_wav(self, midi_path, output_wav_path):
        cmd = [FLUIDSYNTH_BIN, '-n', '-i', '-T', 'wav', '-F', output_wav_path, '-r', '44100', SF2_PATH, midi_path]
        subprocess.run(cmd, check=True, timeout=60, capture_output=True)

    def generate_clip(self, text_input, user_config, log_path):
        self.sentence_count += 1
        sentence_idx = self.sentence_count
        print(f"\n Diary {self.current_diary_id} - Sentence {sentence_idx} ---")
        print(f"User: \"{text_input}\"")

        # 1. Emotion Pred
        inputs = self.tokenizer([text_input], return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
        with torch.no_grad():
            logits = self.text_pred_model(**inputs).logits.cpu().numpy()[0]
            val_raw = float(logits[0]) * 14.0 - 2.0
            aro_raw = float(logits[1]) * 14.0 - 2.0
            predicted_val = (max(0.0, min(10.0, val_raw)) / 5.0) - 1.0
            predicted_aro = (max(2.5, min(7.5, aro_raw)) - 5.0) / 2.5
        print(f"Emotion -> V: {predicted_val:.2f}, A: {predicted_aro:.2f}")

        # Dynamic Steps length based on text length
        # 我们重新精细化时间映射：强迫锁定在 5 到 15 秒的最佳击球区内
        word_count = len(text_input.split())
        
        # 即使字数很多，我们也把硬性规定上限卡在 11 秒左右，留下一点收尾空间刚好能达到 12-15 秒
        target_time = max(5.0, min(11.0, word_count / 2.5)) 
        
        # 每秒所需Token：对于这套模型，25 个 Token/秒 是比较稳妥的均值
        min_music_tokens = int(target_time * 25)
        
        # 将自由发挥的 buffer 从 300 砍到 150，逼迫模型在达到目标长度后，尽快且自然地画上句号
        steps = min(1600, min_music_tokens + 150)

        # 2. Text -> Mus Latent
        seq_tensor = self.process_text_to_latent(text_input, 256)
        with torch.no_grad():
            z_text, _, _ = self.text_vae.encode(seq_tensor)
            z_music_pred = self.cl_net(z_text, None, None, None, training=False)

            va_tensor = torch.tensor([[predicted_val, predicted_aro]], dtype=torch.float32).to(device)
            y_emb_full = self.music_vae.y_proj(va_tensor).repeat(1, 5)
            z_music_modified = z_music_pred + y_emb_full

            if self.previous_latent is not None:
                z_music_modified = z_music_modified * 0.8 + self.previous_latent * 0.2
            self.previous_latent = z_music_modified.clone()

            # Latent Mod 
            try:
                if any(v != 0.0 for v in [user_config['density'], user_config['pitch'], user_config['velocity']]):
                    dir_den = torch.tensor(np.load(os.path.join(EMOMUSIC_DIR, 'latent_directions', 'dir_density.npy')), dtype=torch.float32).to(device)
                    dir_pit = torch.tensor(np.load(os.path.join(EMOMUSIC_DIR, 'latent_directions', 'dir_pitch.npy')), dtype=torch.float32).to(device)
                    dir_vel = torch.tensor(np.load(os.path.join(EMOMUSIC_DIR, 'latent_directions', 'dir_velocity.npy')), dtype=torch.float32).to(device)
                    z_music_modified += (user_config['density']*dir_den + user_config['pitch']*dir_pit + user_config['velocity']*dir_vel)
            except: pass

            logits = self.music_vae.global_decoder(z_music_modified, steps=int(steps))
            
            # 銆愭牳蹇冧慨琛ワ細寮哄埗寤朵几闊充箰闀垮害銆?
            # 妯″瀷鍦?Temp=0.6 鏃惰繃浜庘€滆嚜淇♀€濓紝甯稿父鍦ㄧ敓鎴?00-200涓猅oken锛堢害2-3绉掞級鍚庡氨棰勬祴浼戞绗︽垨缁撴潫绗︺€?
            # 鎴戜滑閫氳繃鎶婂墠 800 涓?Token 鐨?PAD(0) 鍜?EOS(1) 姒傜巼寮哄埗鎷夐粦锛岄€艰揩妯″瀷缁х画鍒涗綔鍑鸿冻澶熼暱鐨勬棆寰嬶紒
            # min_music_tokens dynamically calculated above
            logits[:, :min_music_tokens, 0] = -1e9  # Mask PAD token
            logits[:, :min_music_tokens, 1] = -1e9  # Mask EOS token

            probs = torch.softmax(logits / 0.6, dim=-1)
            predicted_ids = torch.multinomial(probs.view(-1, 342), 1).view(1, int(steps)).cpu().numpy()

        clean_ids = [t for t in predicted_ids[0] if t >= 2]
        
        # Save nested directly into Diary Folder
        temp_mid = os.path.join(self.current_diary_dir, f'temp_sentence_{sentence_idx}.mid')
        out_mid = os.path.join(self.current_diary_dir, f'sentence_{sentence_idx}.mid')
        out_wav = os.path.join(self.current_diary_dir, f'sentence_{sentence_idx}.wav')

        shutil.copy2(self.mpe.decode(clean_ids, strip_extraneous=True), temp_mid)

        # Apply Theory Filter
        self.mtf.apply_constraints(temp_mid, out_mid, valence=predicted_val, arousal=predicted_aro,
                                   force_scale=user_config['scale'], force_velocity=user_config['rule_vel'], force_grid=user_config['grid'])
        
        self.midi_to_wav(out_mid, out_wav)
        self.generated_midis.append(out_mid)
        
        # Evaluate snippet
        eval_res = evaluate_midi(out_mid)
        self.last_midi_path = out_mid
        self.last_eval = eval_res
        
        # Log Sentence Eval
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"\n[Diary {self.current_diary_id} - Sentence {sentence_idx}]\n")
            f.write(f"Text: {text_input}\n")
            f.write(f"Emotion Predicted: Valence={predicted_val:.2f}, Arousal={predicted_aro:.2f}\n")
            f.write(f"UI Controls: Density={user_config['density']}, Pitch={user_config['pitch']}, Vel={user_config['velocity']}, Scale={user_config['scale']}, Grid={user_config['grid']}, VelRule={user_config['rule_vel']}\n")
            f.write(f"Evaluation: Notes={eval_res['notes']}, Dens={eval_res['density']:.2f}, Dur={eval_res['duration']:.2f}s, Vel={eval_res['velocity']:.1f}, CMaj={eval_res['c_maj_ratio']*100:.1f}%, CMin={eval_res['c_min_ratio']*100:.1f}%\n")
            
        print(f"-> Clip {sentence_idx} generated: {out_wav}")
        return out_wav

    def merge_diary(self, log_path):
        """Concatenates all sentence midis into one full diary midi and performs Macro-Evaluation."""
        if not self.generated_midis:
            return None
            
        merged_pm = pretty_midi.PrettyMIDI()
        piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
        merged_inst = pretty_midi.Instrument(program=piano_program)
        
        current_time = 0.0
        
        # Lists for Macro-Evaluation Tracking
        densities, pitches_avg, cmaj_ratios = [], [], []
        start_pitches, end_pitches = [], []
        
        for mid_file in self.generated_midis:
            try:
                # Store Eval Data
                eval_res = evaluate_midi(mid_file)
                densities.append(eval_res['density'])
                cmaj_ratios.append(eval_res['c_maj_ratio'])
                
                pm = pretty_midi.PrettyMIDI(mid_file)
                if not pm.instruments or not pm.instruments[0].notes:
                    continue
                    
                # Fix blank starting gaps
                first_note_start = min(n.start for n in pm.instruments[0].notes)
                
                # Evaluation tracking: pitch ranges
                sorted_notes = sorted(pm.instruments[0].notes, key=lambda x: x.start)
                pitches_avg.append(np.mean([n.pitch for n in sorted_notes]))
                start_pitches.append(np.mean([n.pitch for n in sorted_notes[:3]]))
                end_pitches.append(np.mean([n.pitch for n in sorted_notes[-3:]]))
                
                clip_max_relative_time = 0.0
                for inst in pm.instruments:
                    for note in inst.notes:
                        # Shift by first_note_start to remove front silence
                        shifted_start = note.start - first_note_start
                        shifted_end = note.end - first_note_start
                        
                        new_note = pretty_midi.Note(
                            velocity=note.velocity,
                            pitch=note.pitch,
                            start=shifted_start + current_time,
                            end=shifted_end + current_time
                        )
                        merged_inst.notes.append(new_note)
                        
                        # Find the relative duration of this specific clip
                        if shifted_end > clip_max_relative_time:
                            clip_max_relative_time = shifted_end
                            
                    # Preserve CC64 Sustain Pedal across merge
                    for cc in inst.control_changes:
                        shifted_cc = cc.time - first_note_start
                        if shifted_cc >= 0:
                            merged_inst.control_changes.append(pretty_midi.ControlChange(cc.number, cc.value, shifted_cc + current_time))
                        
                # Update current time: Previous time + length of this clip + 0.8s gap
                current_time = current_time + clip_max_relative_time + 0.8
            except Exception as e:
                print(f"Error merging {mid_file}: {e}")
                
        merged_pm.instruments.append(merged_inst)
        
        merged_out_mid = os.path.join(self.current_diary_dir, 'full_diary_merged.mid')
        merged_out_wav = os.path.join(self.current_diary_dir, 'full_diary_merged.wav')
        
        merged_pm.write(merged_out_mid)
        self.midi_to_wav(merged_out_mid, merged_out_wav)
        
        # --- Call Centralized Macro Evaluation ---
        macro_stats = evaluate_diary_macro(self.generated_midis, merged_out_mid)
        self.last_macro = macro_stats

        # Log total macro sequence
        merged_eval = macro_stats["merged_eval"]
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"\n[{'='*15} Diary {self.current_diary_id} MACRO SUMMARY {'='*15}]\n")
            f.write(f"Total Sentences: {self.sentence_count}\n")
            f.write(f"Total Length: {current_time:.2f} seconds\n")
            f.write(f"Total Notes: {merged_eval['notes']}\n")
            f.write(f"- Global Eval: Dens={merged_eval['density']:.2f}, Vel={merged_eval['velocity']:.1f}, CMaj={merged_eval['c_maj_ratio']*100:.1f}%, CMin={merged_eval['c_min_ratio']*100:.1f}%\n")
            f.write(f"--- High-level Macro Evaluation ---\n")
            f.write(f">> 1. Emotional Trajectory (Variance): Pitch Var = {macro_stats['traj_pitch_var']:.2f}, Density Var = {macro_stats['traj_density_var']:.2f} (Higher means more emotional fluctuation)\n")
            f.write(f">> 2. Transition Smoothness: Avg Pitch Leap = {macro_stats['smoothness_avg_leap']:.2f} semitones (Lower means smoother transitions)\n")
            f.write(f">> 3. Tonal Cohesion: Key Mode StdDev = {macro_stats['tonal_cohesion_var']:.3f} (Lower means more harmonically consistent)\n")
            f.write(f"{'='*60}\n")
            
        print(f"Merged full diary saved to: {merged_out_wav}")
        return merged_out_wav

if __name__ == "__main__":
    # --- Simulated Diary Session ---
    diary_entries = [
        "Today was a really exhausting day. Everything went wrong.",
        "I just wanted to lay in bed and do nothing.",
        "But then, my friends surprised me with a cake at my door!",
        "I feel so much better and hopeful now."
    ]

    ui_state = {
        'density': 0.0, 'pitch': 0.0, 'velocity': 0.0,
        'scale': 'Auto', 'grid': 'Auto', 'rule_vel': 'Auto'
    }

    engine = DiaryEngine()
    log_file_path = os.path.join(CURRENT_DIR, 'diary_log.txt')
    
    # 1. Initialize Diary
    engine.start_new_diary(log_file_path)

    print("="*60)
    print(f"WEMOM Diary Workflow Demo Started (Diary {engine.current_diary_id})")
    print("="*60)

    # 2. Iterate through sentences
    for idx, sentence in enumerate(diary_entries):
        # UX Concept: User adjusts sliders sentence-by-sentence
        if idx == 2:
            ui_state['density'] = 0.5  # 适度增加活力，避免 2.0 甚至 3.0 造成的音符爆炸拥挤
        elif idx == 3:
            ui_state['density'] = 1.0  # 同样降低这个极端的参数，让旋律保持清晰不砸琴

        engine.generate_clip(sentence, ui_state, log_file_path)

    # 3. Merge entire state at the end
    print("\n Merging individual segments into a final master track...")
    engine.merge_diary(log_file_path)

    print("\n Diary Simulation Complete! Check generated_midis/diary folder and diary_log.txt.")

