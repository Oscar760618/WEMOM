import torch
import numpy as np
import json
import os
from MusicVAE import MusicAttrRegGMVAE
from utils1 import magenta_decode_midi
import shutil
from midi2audio import FluidSynth
from pydub import AudioSegment

# Load config
with open('D:/PolyU/URIS/Part2_projects/WEMOM_V1/EmoMusic/MusicVAE.json') as f:
    args = json.load(f)

source_path = 'D:/PolyU/URIS/Part2_projects/WEMOM_V1/Data/music_source.npy'

EVENT_DIMS = 342
RHYTHM_DIMS = 3
NOTE_DIMS = 16
CHROMA_DIMS = 24

# Load model
model = MusicAttrRegGMVAE(
    roll_dims=EVENT_DIMS, rhythm_dims=RHYTHM_DIMS, note_dims=NOTE_DIMS, 
    chroma_dims=CHROMA_DIMS,
    hidden_dims=args['hidden_dim'], z_dims=args['z_dim'], 
    n_step=args['time_step'],
    n_component=args['num_clusters']
)
checkpoint_path = 'D:/PolyU/URIS/Part2_projects/WEMOM_V1/EmoMusic/params/music_attr_vae_reg_gmm_long_v_100.pt'
checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
if torch.cuda.is_available():
    model.cuda()

# Load single music feature
music_feature = np.load(source_path)  # shape: (256,)
print("Loaded music feature:", music_feature.shape)
if music_feature.ndim == 1:
    music_feature = music_feature.reshape(1, -1)  # shape: (1, 256)

# Output folder
output_folder = "generated_midis"
os.makedirs(output_folder, exist_ok=True)

# Prepare input tensor
feature = torch.tensor(music_feature, dtype=torch.float32)
if torch.cuda.is_available():
    feature = feature.cuda()
chroma = torch.ones(feature.shape[0], 24).cuda() if torch.cuda.is_available() else torch.ones(feature.shape[0], 24)
z = torch.cat([feature, chroma], dim=1)  # shape: [1, 256 + 24]

# Generate sequence
out = model.global_decoder(z, steps=300)  # steps can be adjusted

# Convert to MIDI and save
midi_path = os.path.join(output_folder, "music_0.mid")
notes = torch.argmax(out, dim=-1).cpu().detach().numpy().squeeze()
notes = np.trim_zeros(notes)

# Filter notes to valid event indices
NUM_RESERVED_TOKENS = 2
NUM_CLASSES = out.shape[-1]
valid_min = NUM_RESERVED_TOKENS
valid_max = NUM_RESERVED_TOKENS + NUM_CLASSES - 1
notes = notes[(notes >= valid_min) & (notes <= valid_max)]
# print(f"Filtered notes to valid indices [{valid_min}, {valid_max}]:", notes)


midi_tmp_path = magenta_decode_midi(notes=notes)

soundfont_path = 'D:/PolyU/URIS/Part2_projects/WEMOM_V1/EmoMusic/FluidR3_GM.sf2' 

# Copy MIDI to output folder
shutil.copy(midi_tmp_path, midi_path)
print(f"Saved MIDI to {midi_path}")

# Convert MIDI to WAV
wav_path = os.path.join(output_folder, "music_1.wav")
fs = FluidSynth(soundfont_path)
fs.midi_to_audio(midi_path, wav_path)
print(f"Saved WAV to {wav_path}")

# Convert WAV to MP3
mp3_path = os.path.join(output_folder, "music_1.mp3")
audio = AudioSegment.from_wav(wav_path)
audio.export(mp3_path, format="mp3")
print(f"Saved MP3 to {mp3_path}")

if os.path.exists(midi_path):
    print("MIDI file created. Size:", os.path.getsize(midi_path), "bytes")
else:
    print("MIDI file not found after saving.")