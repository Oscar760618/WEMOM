import os
import sys
import torch
import numpy as np
import json
import pretty_midi
import warnings
from tqdm import tqdm

# Setup paths to import project modules
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)
sys.path.append(CURRENT_DIR)

from WEMOM_V1.EmoMusic.MusicVAE import MusicAttrRegGMVAE
from MidiPerformanceEncoder import MidiPerformanceEncoder

warnings.filterwarnings("ignore")

# --- Constants & Config ---
MUSIC_CHECKPOINT_PATH = os.path.join(CURRENT_DIR, 'params', 'MusicVAE_50.pt')
MUSIC_CONFIG_PATH = os.path.join(CURRENT_DIR, 'MusicVAE.json')
OUTPUT_DIR = os.path.join(CURRENT_DIR, 'latent_directions')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Z_DIM = 640
NUM_SAMPLES = 500  # Number of random samples to generate
STEPS = 1000 
EVENT_DIMS = 342
TEMPERATURE = 0.8

def load_music_vae():
    print(">>> Loading MusicVAE...")
    with open(MUSIC_CONFIG_PATH) as f:
        m_args = json.load(f)
    
    RHYTHM_DIMS, NOTE_DIMS, CHROMA_DIMS, DYNAMIC_DIMS, CHORD_DIMS = 3, 16, 24, 5, 24
    music_vae = MusicAttrRegGMVAE(
        roll_dims=EVENT_DIMS, rhythm_dims=RHYTHM_DIMS, note_dims=NOTE_DIMS,
        chroma_dims=CHROMA_DIMS, dynamic_dims=DYNAMIC_DIMS, chord_dims=CHORD_DIMS,
        hidden_dims=m_args['hidden_dim'], z_dims=m_args['z_dim'],
        n_step=m_args['time_step'], n_component=m_args['num_clusters']
    ).to(DEVICE)
    
    m_checkpoint = torch.load(MUSIC_CHECKPOINT_PATH, map_location=DEVICE)
    if 'model_state_dict' in m_checkpoint:
        music_vae.load_state_dict(m_checkpoint['model_state_dict'])
    else:
        music_vae.load_state_dict(m_checkpoint)
        
    music_vae.eval()
    
    mpe = MidiPerformanceEncoder(steps_per_second=100, num_velocity_bins=64, min_pitch=21, max_pitch=108, add_eos=True)
    return music_vae, mpe

def extract_heuristic_features(midi_path):
    """
    Extracts simple physical features from a generated MIDI file:
    1. Note Density (Notes per second -> rhythm/tempo proxy)
    2. Average Pitch (High/low -> brightness proxy)
    3. Average Velocity (Loudness -> energy proxy)
    """
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        return None
        
    if len(pm.instruments) == 0:
        return None
        
    instrument = pm.instruments[0]
    notes = instrument.notes
    if len(notes) < 10: # Skip empty or extremely sparse outputs
        return None
        
    total_duration = pm.get_end_time()
    if total_duration == 0:
        return None
        
    # Feature 1: Note Density (notes per second)
    note_density = len(notes) / total_duration
    
    # Feature 2: Average Pitch (0-127)
    avg_pitch = np.mean([note.pitch for note in notes])
    
    # Feature 3: Average Velocity (0-127)
    avg_velocity = np.mean([note.velocity for note in notes])
    
    return {
        "density": note_density,
        "pitch": avg_pitch,
        "velocity": avg_velocity
    }

def ridge_regression_direction(X, y, alpha=100.0):
    """
    Pure Numpy implementation of Ridge Regression to find the mapping vector.
    """
    # 1. Standardize X
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + 1e-8
    X_scaled = (X - X_mean) / X_std
    
    # 2. Standardize Y
    y_mean = np.mean(y)
    y_std = np.std(y) + 1e-8
    y_scaled = (y - y_mean) / y_std
    
    # 3. Fit Ridge Regression: w = (X^T X + alpha*I)^-1 X^T Y
    I = np.eye(X_scaled.shape[1])
    w = np.linalg.inv(X_scaled.T @ X_scaled + alpha * I) @ X_scaled.T @ y_scaled
    
    # 4. R^2 Score Validation
    y_pred = X_scaled @ w
    ss_res = np.sum((y_scaled - y_pred)**2)
    ss_tot = np.sum((y_scaled - np.mean(y_scaled))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    return w, r2

def main():
    print("="*60)
    print("🎵 WEMOM: Explorable Latent Space Direction Discovery")
    print("="*60)
    
    music_vae, mpe = load_music_vae()
    
    print(f"\n>>> Step 1: Sampling {NUM_SAMPLES} random latent vectors (z) and decoding...")
    z_samples = []
    densities = []
    pitches = []
    velocities = []
    
    successful_samples = 0
    with torch.no_grad():
        for i in tqdm(range(NUM_SAMPLES), desc="Generating & Analyzing"):
            # Sample random z vector from standard normal distribution
            z = torch.randn(1, Z_DIM).to(DEVICE)
            
            # Decode to MIDI sequence
            logits = music_vae.global_decoder(z, steps=STEPS)
            logits = logits / TEMPERATURE
            probs = torch.softmax(logits, dim=-1)
            predicted_ids = torch.multinomial(probs.view(-1, EVENT_DIMS), 1).view(1, STEPS).cpu().numpy()
            
            clean_ids = [t for t in predicted_ids[0] if t >= 2]
            midi_path = mpe.decode(clean_ids, strip_extraneous=True)
            
            # Extract features
            features = extract_heuristic_features(midi_path)
            
            # Clean up temp file
            if os.path.exists(midi_path):
                os.remove(midi_path)
                
            if features is not None:
                z_samples.append(z.squeeze().cpu().numpy())
                densities.append(features["density"])
                pitches.append(features["pitch"])
                velocities.append(features["velocity"])
                successful_samples += 1

    print(f"\n    -> Extracted {successful_samples}/{NUM_SAMPLES} valid sub-clips.")
    if successful_samples < 50:
        print("[Error] Not enough valid samples to compute directions reliably!")
        return

    X = np.array(z_samples)  # Shape: (samples, Z_DIM)
    y_target = {
        "density": np.array(densities),
        "pitch": np.array(pitches),
        "velocity": np.array(velocities)
    }

    print("\n>>> Step 2: Training linear models to find interpretable Direction Vectors...")
    directions = {}
    
    for feature_name, y in y_target.items():
        # Using Ridge Regression to find the vector
        direction_vector, r2_score = ridge_regression_direction(X, y, alpha=50.0)
        
        # L2 Normalize the vector so it has length 1 (pure direction vector)
        direction_norm = direction_vector / np.linalg.norm(direction_vector)
        directions[feature_name] = direction_norm
        
        print(f"    - {feature_name.capitalize()} Vector | R^2 Score: {r2_score:.4f}")
        
        # Save the vector
        out_path = os.path.join(OUTPUT_DIR, f"dir_{feature_name}.npy")
        np.save(out_path, direction_norm)
        print(f"      -> Saved: {out_path}")

    print("\n" + "="*60)
    print("All semantic directions have been successfully extracted and saved!")
    print(f"Check folder: {OUTPUT_DIR}")
    print("============================================================")

if __name__ == "__main__":
    main()
