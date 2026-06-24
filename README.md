# WEMOM: Writing Emotional Music

WEMOM is an end-to-end framework that translates text sentences into emotional music snippets. It maps text features to a shared latent space and decodes them into symbolic music (MIDI format).

## Architecture

1. **TextVAE**: Extracts latent representations from emotional text input.
2. **MusicVAE**: Extracts and decodes latent representations for MIDI music.
3. **Text2Music**: A mapper that translates text latent vectors into matching music latent vectors.
4. **Music Generation (EmoMusic)**: Generates the actual MIDI based on the translated music latent vector.
5. **Frontend/Backend Interface**: A React frontend for users to input sentences and a FastAPI backend to handle the inference pipeline.

## Project Structure
- `WEMOM_V1/backend/`: FastAPI backend handling the generation pipeline.
- `WEMOM_V1/frontend/`: React + Vite frontend for UI.
- `WEMOM_V1/EmoMusic/`: MusicVAE implementation and generation logic.
- `WEMOM_V1/EmoText/`: TextVAE implementation.
- `WEMOM_V1/Text2Music/`: The latent space translation model.

## Setup & Installation

### 1. Model Weights & Data
For the project to run, you need the pre-trained model weights (`params/` directories) and the FluidSynth SoundFont (`FluidR3_GM.sf2`).
> **Note**: Due to file size limits, the model parameters and datasets are not included in this repository. 
> Please download the required files from (https://connectpolyu-my.sharepoint.com/:f:/g/personal/23100294d_connect_polyu_hk/IgDTw3FYAAgURZ_5ogC2mPYpAWdcmRTABHWOBn5HxiTVlbg?e=ralR32) and extract them into their respective directories according to the structure below:
> - `WEMOM_V1/EmoMusic/params/`
> - `WEMOM_V1/EmoText/params/`
> - `WEMOM_V1/Text2Music/params/`
> - `WEMOM_V1/EmoMusic/FluidR3_GM.sf2`

### 2. Environment Setup (Backend)
Requires **Python 3.8+**.
```bash
# Clone the repository
git clone <your-repo-url>
cd WEMOM_V1

# Install required python packages
pip install torch numpy fastapi uvicorn mido pydub 
# (You may need to freeze a requirements.txt later)
```

### 3. Frontend Setup
Requires **Node.js**.
```bash
cd WEMOM_V1/frontend
npm install
```

## Running the Project

**1. Start the Backend Server**
Open a terminal in the `WEMOM_V1` directory and run:
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

**2. Start the Frontend Server**
Open another terminal in the `WEMOM_V1/frontend` directory and run:
```bash
npm run dev
```

**3. Test the Interface**
Open `http://localhost:5173` in your browser. Type a sentence, press Enter, and wait for the music snippet to generate!
