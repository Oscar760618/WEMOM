import streamlit as st
import sys
import os
import torch
import numpy as np
import json
import pretty_midi
import subprocess
import time
import matplotlib.pyplot as plt

# --- Path Setup ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
EMOMUSIC_DIR = os.path.join(PARENT_DIR, 'EmoMusic')
TEXT2MUSIC_DIR = os.path.join(PARENT_DIR, 'Text2Music')
TEXT_PRED_DIR = os.path.join(PARENT_DIR, 'Text_Prediction')
# Update path to point to the HuggingFace format folder inside Text_Prediction/params
TEXT_CHECKPOINT_PATH = os.path.join(TEXT_PRED_DIR, 'params', 'roberta_regression_hf_best')

sys.path.append(PARENT_DIR)
sys.path.append(EMOMUSIC_DIR)
sys.path.append(TEXT_PRED_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    from EmoMusic.MusicVAE import MusicAttrRegGMVAE
    from EmoMusic.MidiPerformanceEncoder import MidiPerformanceEncoder
    from EmoMusic.music_theory_filter import MusicTheoryFilterSmooth
    from transformers import AutoTokenizer, RobertaConfig
    from Text_Prediction.Roberta import RobertaForSequenceClassificationSig
    
    # TextVAE Imports
    from EmoText.TextVAE import TextVAE
    
    # Text2Music Imports
    from Text2Music.model import CLloss
    import Text2Music.config as cl_config
    
except ImportError as e:
    st.error(f"Import Error: {e}. Please ensure you are running this from the correct directory.")

# --- Config & Constants ---
MUSIC_CHECKPOINT_PATH = os.path.join(EMOMUSIC_DIR, 'params', 'MusicVAE_50.pt')
MUSIC_CONFIG_PATH = os.path.join(EMOMUSIC_DIR, 'MusicVAE.json')

TEXTVAE_CHECKPOINT_PATH = os.path.join(PARENT_DIR, 'EmoText', 'params', 'TextVAE_latest.pt')
TEXTVAE_CONFIG_PATH = os.path.join(PARENT_DIR, 'EmoText', 'TextVAE_config.json')
LANG_DICT_PATH = os.path.join(PARENT_DIR, 'Data', 'Saves', 'id2word_text.npy')

CL_CHECKPOINT_PATH = os.path.join(TEXT2MUSIC_DIR, 'params', 'params_1000_6.167.pt')

OUTPUT_MIDI_PATH = os.path.join(PARENT_DIR, 'generated_midis', 'generated_music.mid')
TEMP_WAV_PATH = os.path.join(PARENT_DIR, 'generated_midis', 'generated_music.wav')
SF2_PATH = os.path.join(PARENT_DIR, 'FluidR3_GM', 'FluidR3_GM.sf2')
FLUIDSYNTH_BIN = os.path.join(PARENT_DIR, 'fluidsynth-2.4.7-winXP-x86', 'bin', 'fluidsynth.exe')

# --- Header ---
st.set_page_config(page_title="WEMOM Music Interface", layout="wide")
st.title("Affective Music Co-Creation")
st.markdown("Use AI to generate music from text, then refine the emotion using interactive controls.")

# --------------------------------------------------------------------------------
# Load Text Prediction Model

@st.cache_resource
def load_text_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(TEXT_CHECKPOINT_PATH)
        config = RobertaConfig.from_pretrained(TEXT_CHECKPOINT_PATH)
        model = RobertaForSequenceClassificationSig.from_pretrained(TEXT_CHECKPOINT_PATH, config=config)
        model.to(device)
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"Failed to load Text Prediction Model from {TEXT_CHECKPOINT_PATH}: {e}")
        return None, None
# --------------------------------------------------------------------------------

st.subheader("1. Describe Your Emotion")
text_input = st.text_area("Enter a description (e.g., 'A happy morning with sunshine', 'Sad rain falling down')", height=100)

# Predict Emotion from Text
predicted_val, predicted_aro = 0.0, 0.0

if text_input:
    tk, text_model = load_text_model()
    
    if tk and text_model:
        try:
            # Tokenize
            inputs = tk([text_input], return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
            
            with torch.no_grad():
                outputs = text_model(**inputs)
                logits = outputs.logits.cpu().numpy()[0]
                
                # Consistent with refer.py Reverse scaling
                min_safe = -2.0
                max_range = 14.0
                
                p_val_raw = float(logits[0]) * max_range + min_safe
                p_aro_raw = float(logits[1]) * max_range + min_safe

                # 1. Clip to valid range [0, 10] just in case
                p_val_raw = max(0.0, min(10.0, p_val_raw))
                # For arousal, map the concentrated middle portion [2.5, 7.5] to [-1, 1] 
                clipped_aro_raw = max(2.5, min(7.5, p_aro_raw))

                # 2. Map for the Interface Sliders
                # Valence: [0, 10] -> [-1, 1]
                predicted_val = (p_val_raw / 5.0) - 1.0
                # Arousal: [2.5, 7.5] -> [-1, 1]
                predicted_aro = (clipped_aro_raw - 5.0) / 2.5
        except Exception as e:
            st.error(f"AI Model Inference Failed: {e}")
            st.stop()
    else:
        st.error("Text Prediction Model could not be loaded. Please check the checkpoint path.")
        st.stop()

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("2. Affective Control")
    st.markdown("Refine the emotional parameters:")
    
    # Sliders initialized with predicted values
    user_valence = st.slider("Valence (Negative -> Positive)", min_value=-1.0, max_value=1.0, value=predicted_val, step=0.1)
    user_arousal = st.slider("Arousal (Calm -> Excited)", min_value=-1.0, max_value=1.0, value=predicted_aro, step=0.1)

with col2:
    st.subheader("3. Explorable Latent Space")
    st.caption("Macro-adjustments: Shift latent directions")
    density_shift = st.slider("Note Density (Rhythm/Pace)", -5.0, 5.0, 0.0, 0.5)
    pitch_shift = st.slider("Average Pitch (Brightness)", -5.0, 5.0, 0.0, 0.5)
    velocity_shift = st.slider("Average Velocity (Energy)", -5.0, 5.0, 0.0, 0.5)

with col3:
    st.subheader("4. Music Theory Constraints")
    st.caption("Micro-adjustments: Hard-rule overrides")
    force_scale = st.selectbox("Force Scale", ["Auto", "C Major", "C Minor", "Original"])
    force_grid = st.selectbox("Quantization Grid", ["Auto", "1.0 (Quarter)", "0.5 (Eighth)", "0.25 (Sixteenth)"])
    force_velocity = st.selectbox("Velocity Base", ["Auto", "60 (Soft)", "80 (Medium)", "100 (Hard)"])

@st.cache_resource
def load_generating_models():
    """Load TextVAE, Text2Music(CL), and MusicVAE models"""
    models = {}
    
    # 1. Load TextVAE
    try:
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
        
        # Load Dictionary
        id2word = np.load(LANG_DICT_PATH, allow_pickle=True).item()
        word2id = {v: k for k, v in id2word.items()}
        models['word2id'] = word2id
        
    except Exception as e:
        st.error(f"Failed to load TextVAE: {e}")
        return None

    # 2. Load Text2Music (CL Model)
    try:
        # Assuming CL model uses config dims
        cl_net = CLloss(
            txt_dim=cl_config.TXT_DIM, 
            mus_dim=cl_config.MUS_DIM
        ).to(device)
        
        cl_checkpoint = torch.load(CL_CHECKPOINT_PATH, map_location=device)
        if 'model_state_dict' in cl_checkpoint:
            cl_net.load_state_dict(cl_checkpoint['model_state_dict'])
        else:
            cl_net.load_state_dict(cl_checkpoint)
        cl_net.eval()
        models['cl_net'] = cl_net
        
    except Exception as e:
        st.error(f"Failed to load Text2Music Model: {e}")
        return None

    # 3. Load MusicVAE
    try:
        with open(MUSIC_CONFIG_PATH) as f:
            m_args = json.load(f)
        
        # Hardcoded dims from reconstruct.py
        EVENT_DIMS = 342
        RHYTHM_DIMS, NOTE_DIMS, CHROMA_DIMS, DYNAMIC_DIMS, CHORD_DIMS = 3, 16, 24, 5, 24
        
        music_vae = MusicAttrRegGMVAE(
            roll_dims=EVENT_DIMS,
            rhythm_dims=RHYTHM_DIMS,
            note_dims=NOTE_DIMS,
            chroma_dims=CHROMA_DIMS,
            dynamic_dims=DYNAMIC_DIMS,
            chord_dims=CHORD_DIMS,
            hidden_dims=m_args['hidden_dim'],
            z_dims=m_args['z_dim'],
            n_step=m_args['time_step'],
            n_component=m_args['num_clusters']
        ).to(device)
        
        m_checkpoint = torch.load(MUSIC_CHECKPOINT_PATH, map_location=device)
        if 'model_state_dict' in m_checkpoint:
            music_vae.load_state_dict(m_checkpoint['model_state_dict'])
        else:
            music_vae.load_state_dict(m_checkpoint)
        music_vae.eval()
        models['music_vae'] = music_vae
        
        # Encoder
        mpe = MidiPerformanceEncoder(
            steps_per_second=100,
            num_velocity_bins=64,
            min_pitch=21,
            max_pitch=108,
            add_eos=True
        )
        models['mpe'] = mpe
        
    except Exception as e:
        st.error(f"Failed to load MusicVAE: {e}")
        return None
        
    return models

def process_text_to_latent(text, output_dim, word2id, max_len=30):
    """Convert raw text to TextVAE latent z"""
    # Simple tokenization matching utils.py logic
    import re
    token_list = re.findall(r"[\w']+|[.,!?;]", text.lower())
    
    # Convert to IDs
    input_ids = [word2id.get(token, word2id.get('<unk>', 1)) for token in token_list]
    
    # Pad/Truncate
    if len(input_ids) < max_len:
        input_ids += [word2id.get('<pad>', 0)] * (max_len - len(input_ids))
    else:
        input_ids = input_ids[:max_len]
        
    # Add SOS/EOS if model expects it (TextDataset often adds SOS)
    # Checking text_dataset.py: it adds <sos> at start and <eos> at end usually
    # But let's stick to what TextVAE.encode expects.
    # TextVAE forward expects [Batch, Seq]
    
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    return input_tensor

# --- Logic: Generation ---
def generate_music(models, text_input, valence, arousal, density_shift, pitch_shift, velocity_shift, force_scale, force_grid, force_velocity):
    """
    Full Pipeline: Text -> TextVAE -> Z_text -> CL_Net -> Z_music -> MusicVAE -> MIDI
    """
    # 1. Text -> TextVAE Latent
    seq_tensor = process_text_to_latent(text_input, 256, models['word2id'])
    
    with torch.no_grad():
        # TextVAE encode returns: z, mean, std
        z_text, _, _ = models['text_vae'].encode(seq_tensor)
        
    # 2. Z_text -> Z_music (via CL Net)
    with torch.no_grad():
        z_music_pred = models['cl_net'](z_text, None, None, None, training=False)
        
    # 3. Z_music -> MusicVAE -> Logits
    STEPS = 1600
    TEMPERATURE = 0.8
    EVENT_DIMS = 342
    
    with torch.no_grad():
        # Inject continuous emotion condition (Valence/Arousal) into latent matching C-VAE training
        va_tensor = torch.tensor([[valence, arousal]], dtype=torch.float32).to(device)
        y_emb = models['music_vae'].y_proj(va_tensor) # [1, 128]
        # Repeat for the 5 concatenated latent chunks and add to z_music_pred
        y_emb_full = y_emb.repeat(1, 5) # [1, 640]
        z_music_modified = z_music_pred + y_emb_full
        
        # Apply Latent Directions (Sliders)
        try:
            dir_density = torch.tensor(np.load(os.path.join(EMOMUSIC_DIR, 'latent_directions', 'dir_density.npy')), dtype=torch.float32).to(device)
            dir_pitch = torch.tensor(np.load(os.path.join(EMOMUSIC_DIR, 'latent_directions', 'dir_pitch.npy')), dtype=torch.float32).to(device)
            dir_vel = torch.tensor(np.load(os.path.join(EMOMUSIC_DIR, 'latent_directions', 'dir_velocity.npy')), dtype=torch.float32).to(device)
            z_music_modified = z_music_modified + (density_shift * dir_density) + (pitch_shift * dir_pitch) + (velocity_shift * dir_vel)
        except Exception as e:
            st.warning(f"Could not load latent directions: {e}")
        
        logits = models['music_vae'].global_decoder(z_music_modified, steps=STEPS)
        logits = logits / TEMPERATURE
        probs = torch.softmax(logits, dim=-1)
        predicted_ids = torch.multinomial(probs.view(-1, EVENT_DIMS), 1).view(1, STEPS).cpu().numpy()
    
    ids = predicted_ids[0]
    clean_ids = [t for t in ids if t >= 2]
    
    # 4. Decode to MIDI
    midi_obj = models['mpe'].decode(clean_ids, strip_extraneous=True)
    
    # 5. Apply Rules & V/A Constraints
    # We use the user-provided (or predicted) valence/arousal
    mtf = MusicTheoryFilterSmooth()
    
    temp_raw = os.path.join(PARENT_DIR, 'generated_midis', 'temp_raw.mid')
    import shutil
    shutil.copy2(midi_obj, temp_raw) 
    
    # Parse UI values for constraints
    grid_val = "Auto" if force_grid == "Auto" else float(force_grid.split(' ')[0])
    vel_val = "Auto" if force_velocity == "Auto" else float(force_velocity.split(' ')[0])
    scale_val = force_scale
    
    mtf.apply_constraints(temp_raw, OUTPUT_MIDI_PATH, valence=valence, arousal=arousal, force_scale=scale_val, force_velocity=vel_val, force_grid=grid_val)
    
    return OUTPUT_MIDI_PATH

# --- Logic: Audio Conversion ---
def midi_to_wav(midi_path, output_wav_path):
    # Use fluidasynth to convert MIDI to WAV
    try:
        if not os.path.exists(FLUIDSYNTH_BIN):
            st.error(f"FluidSynth not found at: {FLUIDSYNTH_BIN}")
            return False

        # Command: fluidsynth -ni soundfont.sf2 input.mid -F output.wav -r 44100
        # Ensure arguments are separate properly
        cmd = [
            FLUIDSYNTH_BIN, 
            '-n', '-i', 
            '-T', 'wav', 
            '-F', output_wav_path, 
            '-r', '44100',
            SF2_PATH, 
            midi_path
        ]
        
        # Run command with timeout
        # Add timeout to avoid hanging forever if FluidSynth fails silently
        # Capture output to help debugging
        result = subprocess.run(cmd, check=True, timeout=60, capture_output=True, text=True) 
        # st.info(f"FluidSynth Output: {result.stdout}") # Uncomment for debugging
        return True
    except subprocess.TimeoutExpired:
        st.warning("Audio conversion timed out. FluidSynth took too long.")
        return False
    except subprocess.CalledProcessError as e:
        st.warning(f"Audio Conversion Failed: {e}. Output: {e.output}")
        return False
    except Exception as e:
        st.warning(f"Audio Conversion Failed: {e}. You can still download the MIDI file.")
        return False

# --- Logic: Visualization ---
def plot_piano_roll(midi_path, valence=0.0):
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
        
        # Color based on Valence
        # Happy (V>0) -> Orange/Warm
        # Sad (V<0) -> Blue/Cool
        color = '#FFA500' if valence >= 0 else '#4682B4'
        
        # Extract notes
        pitch_list = []
        start_list = []
        duration_list = []
        
        for instrument in pm.instruments:
            if instrument.is_drum: continue
            for note in instrument.notes:
                pitch_list.append(note.pitch)
                start_list.append(note.start)
                duration_list.append(note.end - note.start)
                
        if not pitch_list:
            return None
            
        # Plot
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh(pitch_list, duration_list, left=start_list, height=0.8, color=color, alpha=0.7)
        
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Pitch", fontsize=8)
        ax.set_title("Generated Piano Roll", fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Limit Y-axis to relevant range
        min_p = min(pitch_list)
        max_p = max(pitch_list)
        ax.set_ylim(max(0, min_p - 5), min(127, max_p + 5))
        
        return fig
    except Exception as e:
        st.error(f"Visualization Error: {e}")
        return None

# --- 3. Action Button ---
st.divider()

if st.button("🎵 Generate Music", type="primary", use_container_width=True):
    if not text_input:
        st.warning("Please enter a text description first.")
    else:
        status = st.status("Generating...")
        try:
            status.write("Loading AI Models (TextVAE + CLNet + MusicVAE)...")
            models = load_generating_models()
            
            if models:
                status.write("Synthesizing: Text -> Latent -> Music...")
                midi_file = generate_music(
                    models, text_input, user_valence, user_arousal,
                    density_shift, pitch_shift, velocity_shift,
                    force_scale, force_grid, force_velocity
                )
                
                status.write("Rendering Audio...")
                audio_success = midi_to_wav(midi_file, TEMP_WAV_PATH)
                
                status.update(label="Generation Complete!", state="complete", expanded=False)
                
                # --- Result Display ---
                st.subheader("🎵 Generation Result")
                st.markdown("**Audio Preview**")
                
                if audio_success and os.path.exists(TEMP_WAV_PATH):
                    st.audio(TEMP_WAV_PATH, format='audio/wav')
                else:
                    st.info("Audio preview unavailable (Conversion failed).")
                
                with open(midi_file, "rb") as f:
                    st.download_button("Download MIDI File", f, "generated.mid", use_container_width=True)
            else:
                status.update(label="Model Loading Failed", state="error", expanded=True)
                
        except Exception as e:
            status.update(label="Generation Error", state="error", expanded=True)
            st.error(f"Generation Error: {e}")
            import traceback
            st.code(traceback.format_exc())

            
# --- Footer ---
st.markdown("---")
st.caption("WEMOM Project V1 | HCI Interface Prototype")
