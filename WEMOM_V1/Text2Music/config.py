import os

# Base Directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data Paths
# Ensure these files exist. MUSE_ROOT should point to the output of EmoMusic/refer.py
TEXT_ROOT = os.path.join(BASE_DIR, "Data", "All_Data", "text_features.npy") 
MUSE_ROOT = os.path.join(BASE_DIR, "Data", "All_Data", "music_features.npy") 

# Sample Pair Paths
SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Samples")

TEST_MUSE_POS_ROOT = os.path.join(SAMPLES_DIR, "test_music_pos_samples.txt")
TEST_MUSE_NEG_ROOT = os.path.join(SAMPLES_DIR, "test_music_neg_samples.txt")
TEST_TEXT_POS_ROOT = os.path.join(SAMPLES_DIR, "test_text_pos_samples.txt")
TEST_TEXT_NEG_ROOT = os.path.join(SAMPLES_DIR, "test_text_neg_samples.txt")

# Check if train files exist, otherwise fallback or you need to create them
TRAIN_MUSE_POS_ROOT = os.path.join(SAMPLES_DIR, "train_music_pos_samples.txt")
TRAIN_MUSE_NEG_ROOT = os.path.join(SAMPLES_DIR, "train_music_neg_samples.txt")
TRAIN_TEXT_POS_ROOT = os.path.join(SAMPLES_DIR, "train_text_pos_samples.txt")
TRAIN_TEXT_NEG_ROOT = os.path.join(SAMPLES_DIR, "train_text_neg_samples.txt")

# Model Parameters
TXT_DIM = 256 # Updated to match TextVAE Latent Space
MUS_DIM = 640  # Critical Update: 128 * 5 (rhythm, note, dynamic, chord, chroma)
CL_DIM = 128 # Reduced from 256 to prevent overfitting
MID_DIM = 256 # Reduced from 512 to prevent overfitting
bw = 30 # unused?
init_lr = 0.001 # Increased LR for faster convergence
BATCH_SIZE = 256 # Increased Batch Size for better Contrastive Learning
EPOCH = 1500 # Increased training time
