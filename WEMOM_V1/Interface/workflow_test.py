import sys
import os
import torch
import numpy as np
import json
import re
import shutil
import subprocess

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

print(">>> [Environment] Using device:", device)

from WEMOM_V1.EmoMusic.MusicVAE import MusicAttrRegGMVAE
from EmoMusic.MidiPerformanceEncoder import MidiPerformanceEncoder
from EmoMusic.music_theory_filter import MusicTheoryFilterSmooth
from EmoMusic.music_theory_filter_original import MusicTheoryFilterOriginal
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

# --- Output Paths (for debugging) ---
TEXT_LATENT_OUT = os.path.join(PARENT_DIR, 'generated_midis', 'debug_text_latent.npy')
MUSIC_LATENT_OUT = os.path.join(PARENT_DIR, 'generated_midis', 'debug_music_latent.npy')
TEMP_RAW_MIDI = os.path.join(PARENT_DIR, 'generated_midis', 'temp_raw.mid')
OUTPUT_MIDI_PATH = os.path.join(PARENT_DIR, 'generated_midis', 'generated_music.mid')
OUTPUT_WAV_PATH = os.path.join(PARENT_DIR, 'generated_midis', 'generated_music.wav')

SF2_PATH = os.path.join(PARENT_DIR, 'FluidR3_GM', 'FluidR3_GM.sf2')
FLUIDSYNTH_BIN = os.path.join(PARENT_DIR, 'fluidsynth-2.4.7-winXP-x86', 'bin', 'fluidsynth.exe')

# --------------------------------------------------------------------------------
# Loading Functions
# --------------------------------------------------------------------------------

def load_text_prediction_model():
    print(">>> [1/4] Loading Text VA Prediction Model (Roberta)...")
    tokenizer = AutoTokenizer.from_pretrained(TEXT_CHECKPOINT_PATH)
    config = RobertaConfig.from_pretrained(TEXT_CHECKPOINT_PATH)
    model = RobertaForSequenceClassificationSig.from_pretrained(TEXT_CHECKPOINT_PATH, config=config)
    model.to(device)
    model.eval()
    print("    - Text VA Prediction Model loaded successfully!")
    return tokenizer, model

def load_generating_models():
    models = {}
    print(">>> [2/4] Loading Text Feature Extraction Model (TextVAE)...")
    with open(TEXTVAE_CONFIG_PATH) as f:
        t_args = json.load(f)
    text_vae = TextVAE(
        vocab_size=t_args['vocab_size'],
        embed_size=t_args['embedding_size'],
        hidden_size=t_args['hidden_size'],
        num_layers=t_args['num_layers'],
        dropout=t_args['dropout']
    ).to(device)
    t_checkpoint = torch.load(TEXTVAE_CHECKPOINT_PATH, map_location=device)
    text_vae.load_state_dict(t_checkpoint['model_state_dict'])
    text_vae.eval()
    models['text_vae'] = text_vae
    
    id2word = np.load(LANG_DICT_PATH, allow_pickle=True).item()
    word2id = {v: k for k, v in id2word.items()}
    models['word2id'] = word2id
    print("    - TextVAE and Dictionary loaded successfully!")

    print(">>> [3/4] Loading Contrastive Learning Alignment Model (CL Net)...")
    cl_net = CLloss(txt_dim=cl_config.TXT_DIM, mus_dim=cl_config.MUS_DIM).to(device)
    cl_checkpoint = torch.load(CL_CHECKPOINT_PATH, map_location=device)
    if 'model_state_dict' in cl_checkpoint:
        cl_net.load_state_dict(cl_checkpoint['model_state_dict'])
    else:
        cl_net.load_state_dict(cl_checkpoint)
    cl_net.eval()
    models['cl_net'] = cl_net
    print("    - CL Net loaded successfully!")

    print(">>> [4/4] Loading Music Generation Model (MusicVAE)...")
    with open(MUSIC_CONFIG_PATH) as f:
        m_args = json.load(f)
    EVENT_DIMS = 342
    RHYTHM_DIMS, NOTE_DIMS, CHROMA_DIMS, DYNAMIC_DIMS, CHORD_DIMS = 3, 16, 24, 5, 24
    music_vae = MusicAttrRegGMVAE(
        roll_dims=EVENT_DIMS, rhythm_dims=RHYTHM_DIMS, note_dims=NOTE_DIMS,
        chroma_dims=CHROMA_DIMS, dynamic_dims=DYNAMIC_DIMS, chord_dims=CHORD_DIMS,
        hidden_dims=m_args['hidden_dim'], z_dims=m_args['z_dim'],
        n_step=m_args['time_step'], n_component=m_args['num_clusters']
    ).to(device)
    m_checkpoint = torch.load(MUSIC_CHECKPOINT_PATH, map_location=device)
    if 'model_state_dict' in m_checkpoint:
        music_vae.load_state_dict(m_checkpoint['model_state_dict'])
    else:
        music_vae.load_state_dict(m_checkpoint)
    music_vae.eval()
    models['music_vae'] = music_vae
    
    mpe = MidiPerformanceEncoder(steps_per_second=100, num_velocity_bins=64, min_pitch=21, max_pitch=108, add_eos=True)
    models['mpe'] = mpe
    print("    - MusicVAE and MPE loaded successfully!")
    return models

# --------------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------------

def process_text_to_latent(text, output_dim, word2id, max_len=30):
    token_list = re.findall(r"[\w']+|[.,!?;]", text.lower())
    input_ids = [word2id.get(token, word2id.get('<unk>', 1)) for token in token_list]
    if len(input_ids) < max_len:
        input_ids += [0] * (max_len - len(input_ids))
    else:
        input_ids = input_ids[:max_len]
    return torch.tensor([input_ids], dtype=torch.long).to(device)

def midi_to_wav(midi_path, output_wav_path):
    print(f"\n>>> [Audio Conversion] Converting MIDI to WAV via FluidSynth...\n")
    if not os.path.exists(FLUIDSYNTH_BIN):
        print(f"    [Error] FluidSynth not found: {FLUIDSYNTH_BIN}\n")
        return False
    # Modify fluidsynth fast render arguments to correctly point to input file and sf2 last.
    cmd = [
        FLUIDSYNTH_BIN, 
        '-n', '-i', 
        '-T', 'wav', 
        '-F', output_wav_path, 
        '-r', '44100',
        SF2_PATH, 
        midi_path
    ]
    try:
        subprocess.run(cmd, check=True, timeout=60, capture_output=True, text=True)
        print(f"    [Success] WAV saved to: {output_wav_path}")
        return True
    except Exception as e:
        print(f"    [Failed] WAV conversion error: {e}")
        return False

# --------------------------------------------------------------------------------
# Main Generation Workflow
# --------------------------------------------------------------------------------

def run_workflow_test(text_input):
    print("\n" + "="*50)
    
    # --- Determine Run ID ---
    run_id = 1
    while os.path.exists(os.path.join(PARENT_DIR, 'generated_midis', f'generated_music_{run_id}.mid')):
        run_id += 1
        
    OUTPUT_MIDI_PATH = os.path.join(PARENT_DIR, 'generated_midis', f'generated_music_{run_id}.mid')
    OUTPUT_WAV_PATH = os.path.join(PARENT_DIR, 'generated_midis', f'generated_music_{run_id}.wav')
    TEMP_RAW_MIDI = os.path.join(PARENT_DIR, 'generated_midis', f'temp_raw_{run_id}.mid')

    print(f"🎶 WEMOM End-to-End Generation Test Started [Run ID: {run_id}]")
    
    # =================================================================
    # 🎛️ U`SER CONTROLS (Simulate Streamlit UI sliders & selects here)
    # =================================================================
    # 1. Latent Space Macro-Controls (-5.0 to 5.0)
    # 想要重现最纯净的基础模型生成，请全设为 0.0
    user_density_shift  = 0.0    
    user_pitch_shift    = 0.0   
    user_velocity_shift = 0.0 
    
    # 2. Music Theory Constraints overrides (Micro-Controls)
    # 在 Original Filter 下，这三个强制参数不会被读取，系统只认 V 和 A
    user_force_scale    = "Auto"                         
    user_force_grid     = "Auto"     
    user_force_velocity = "Auto"              
    # =================================================================

    print(f"📝 Test Text Input: '{text_input}'")
    print("="*50 + "\n")

    # Step 1: Predict Emotion (Valence/Arousal)
    tk, text_pred_model = load_text_prediction_model()
    inputs = tk([text_input], return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    # Inference
    with torch.no_grad():
        outputs = text_pred_model(**inputs)
        logits = outputs.logits.cpu().numpy()[0]
        
        # Consistent with refer.py Reverse scaling
        min_safe = -2.0
        max_range = 14.0
        
        p_val_raw = float(logits[0]) * max_range + min_safe
        p_aro_raw = float(logits[1]) * max_range + min_safe
        
        # Clip to valid range [0, 10]
        p_val_raw = max(0.0, min(10.0, p_val_raw))
        # Map the concentrated mid data [2.5, 7.5] to [-1, 1] for distinct slider control
        clipped_aro_raw = max(2.5, min(7.5, p_aro_raw))

        # Map to [-1, 1] for constraints
        predicted_val = (p_val_raw / 5.0) - 1.0
        predicted_aro = (clipped_aro_raw - 5.0) / 2.5
    print(f"    - Valence : {p_val_raw:.2f} (Mapped to {predicted_val:.2f})")
    print(f"    - Arousal : {p_aro_raw:.2f} (Mapped to {predicted_aro:.2f})")

    # Step 2: Load Generation Models
    print("\n" + "-"*50)
    models = load_generating_models()
    
    # Step 3: Text -> TextVAE
    print(f"\n>>> [Stage 2] Text -> TextVAE Latent (z_text)")
    seq_tensor = process_text_to_latent(text_input, 256, models['word2id'])
    with torch.no_grad():
        z_text, _, _ = models['text_vae'].encode(seq_tensor)
    
    np.save(TEXT_LATENT_OUT, z_text.cpu().numpy())
    print(f"    - z_text extraction complete, dimension: {z_text.shape}. Saved to: {TEXT_LATENT_OUT}")

    # Step 4: Z_text -> CL Net -> Z_music
    print(f"\n>>> [Stage 3] Z_text -> CL Net -> Z_music (z_music_pred)")
    with torch.no_grad():
        z_music_pred = models['cl_net'](z_text, None, None, None, training=False)
        
    np.save(MUSIC_LATENT_OUT, z_music_pred.cpu().numpy())
    print(f"    - z_music_pred inference complete, dimension: {z_music_pred.shape}. Saved to: {MUSIC_LATENT_OUT}")

    # Step 5: Z_music -> MusicVAE -> MIDI
    print(f"\n>>> [Stage 4] Z_music -> MusicVAE -> MIDI Generation")
    STEPS = 1600
    TEMPERATURE = 0.6  # 降低温度：避免“胡乱按键”，让生成的音乐结构更稳定
    EVENT_DIMS = 342
    
    with torch.no_grad():
        # Inject continuous emotion condition (Valence/Arousal) into latent matching C-VAE training
        va_tensor = torch.tensor([[predicted_val, predicted_aro]], dtype=torch.float32).to(device)
        y_emb = models['music_vae'].y_proj(va_tensor) # [1, 128]
        # Repeat for the 5 concatenated latent chunks and add to z_music_pred
        y_emb_full = y_emb.repeat(1, 5) # [1, 640]
        z_music_modified = z_music_pred + y_emb_full
        
        # Load and apply Latent Directions (Simulating UI sliders: density_shift, pitch_shift, velocity_shift)
        density_shift, pitch_shift, velocity_shift = user_density_shift, user_pitch_shift, user_velocity_shift # Applied from top config
        try:
            dir_density = torch.tensor(np.load(os.path.join(EMOMUSIC_DIR, 'latent_directions', 'dir_density.npy')), dtype=torch.float32).to(device)
            dir_pitch = torch.tensor(np.load(os.path.join(EMOMUSIC_DIR, 'latent_directions', 'dir_pitch.npy')), dtype=torch.float32).to(device)
            dir_vel = torch.tensor(np.load(os.path.join(EMOMUSIC_DIR, 'latent_directions', 'dir_velocity.npy')), dtype=torch.float32).to(device)
            z_music_modified = z_music_modified + (density_shift * dir_density) + (pitch_shift * dir_pitch) + (velocity_shift * dir_vel)
        except Exception as e:
            print(f"    - Warning: Latent direction sliders skipped ({e})")
        
        logits = models['music_vae'].global_decoder(z_music_modified, steps=STEPS)
        logits = logits / TEMPERATURE
        probs = torch.softmax(logits, dim=-1)
        predicted_ids = torch.multinomial(probs.view(-1, EVENT_DIMS), 1).view(1, STEPS).cpu().numpy()
    
    clean_ids = [t for t in predicted_ids[0] if t >= 2]
    midi_obj = models['mpe'].decode(clean_ids, strip_extraneous=True)
    
    print(f"    - MusicVAE decoding complete, sequence length: {len(clean_ids)}")
    shutil.copy2(midi_obj, TEMP_RAW_MIDI)
    print(f"    - Raw MIDI saved to: {TEMP_RAW_MIDI}")

    # Step 6
    print(f"\n>>> [Stage 5] Music Theory & Emotion Constraints (MusicTheoryFilterSmooth)")
    print(f"    - Applying constraints for Valence={predicted_val:.2f}, Arousal={predicted_aro:.2f}...")
    mtf = MusicTheoryFilterSmooth()
    
    # We can simulate user overrides here
    force_scale = user_force_scale
    force_grid = user_force_grid
    force_velocity = user_force_velocity
    grid_val = "Auto" if force_grid == "Auto" else float(force_grid.split(' ')[0])
    vel_val = "Auto" if force_velocity == "Auto" else float(force_velocity.split(' ')[0])

    mtf.apply_constraints(TEMP_RAW_MIDI, OUTPUT_MIDI_PATH, valence=predicted_val, arousal=predicted_aro, force_scale=force_scale, force_velocity=vel_val, force_grid=grid_val)
    print(f"    - Constrained MIDI saved to: {OUTPUT_MIDI_PATH}")

    # Step 7: 转换音频
    midi_to_wav(OUTPUT_MIDI_PATH, OUTPUT_WAV_PATH)

    # Step 8: Evaluate Generated Music
    print(f"\n>>> [Stage 6] Evaluating MIDI Properties")
    eval_raw = None
    eval_final = None
    try:
        sys.path.append(os.path.join(PARENT_DIR, 'EmoMusic'))
        from EmoMusic.evaluate_music import evaluate_midi
        print("\n--- Evaluate Raw (Pre-constraints) ---")
        eval_raw = evaluate_midi(TEMP_RAW_MIDI)
        print("\n--- Evaluate Final (Post-constraints) ---")
        eval_final = evaluate_midi(OUTPUT_MIDI_PATH)
    except Exception as e:
        print(f"    [Error] Could not run evaluation script: {e}")

    # Step 9: Save Log
    log_path = os.path.join(CURRENT_DIR, 'generation_log.txt')
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] WEMOM Generation Run (Run ID: {run_id})\n")
        f.write(f"Output File: generated_music_{run_id}.mid\n")
        f.write(f"Text Input : {text_input}\n")
        f.write(f"Emotion    : Valence={predicted_val:.2f} (raw:{p_val_raw:.2f}), Arousal={predicted_aro:.2f} (raw:{p_aro_raw:.2f})\n")
        f.write(f"Latent Ctl : Density={user_density_shift}, Pitch={user_pitch_shift}, Velocity={user_velocity_shift}\n")
        f.write(f"Rule Ctl   : Scale={user_force_scale}, Grid={user_force_grid}, Velocity={user_force_velocity}\n")
        if eval_raw:
            f.write(f"Eval (Raw) : Notes={eval_raw['notes']}, Dens={eval_raw['density']:.2f}, Dur={eval_raw['duration']:.3f}, Vel={eval_raw['velocity']:.1f}, CMaj={eval_raw['c_maj_ratio']*100:.1f}%, CMin={eval_raw['c_min_ratio']*100:.1f}%\n")
        if eval_final:
            f.write(f"Eval (Fin) : Notes={eval_final['notes']}, Dens={eval_final['density']:.2f}, Dur={eval_final['duration']:.3f}, Vel={eval_final['velocity']:.1f}, CMaj={eval_final['c_maj_ratio']*100:.1f}%, CMin={eval_final['c_min_ratio']*100:.1f}%\n")
        f.write("-" * 60 + "\n\n")
    print(f"\n    - 📝 Execution log appended and saved to: {log_path}")

    print("\n" + "="*50)
    print("🎉 Workflow test fully completed!")
    print("="*50)

if __name__ == "__main__":
    # Test Prompt
    sample_text = "I feel very sad and depressed today."
    run_workflow_test(sample_text)
