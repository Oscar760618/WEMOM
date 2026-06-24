**Mus_dataset**:
1. 1-3000 midi files for music data

**Text_dataset**:
1. text_data.csv
based on the following files:
a. dataset-fb-valence-arousal-anon.csv
b. emobank.csv
2. text_VA.csv
Processed text_data for Text_VA prediction task

**Saves**
1. Text_CL: Arousal, Valence, Sentence, Label Lists
   For CL model training:
   Valence, Arousal --> Pos, Neg Pairs
   Sentence --> VAE ---> Latent Features --> Training

2. Text_VAE: Arousal, Valence, Sentence, Label Lists
   For text VAE training, including training and testing

3. Music_CL: Arousal, Valence, Chroma, Rhythm, Note Density, Data, Label Lists
   For CL model training:
   Valence, Arousal --> Pos, Neg Pairs
   Chroma, Rhythm, Note Density, Data --> VAE --> Latent Features --> Training

4. Music_VAE: Arousal, Valence, Sentence, Label Lists
   For music VAE training, including training and testing

5. Orgininal data:
*Text:*
id2word_text
vocab_text
sentences_text
*Music:*
chord
chroma
data
dynamic
note
rhythm

**ALL Data**
1. Music Latent Features from Music_CL, used for CL Model Training

**Text Feature**
1. Text Latent Features from Text_CL, used for CL Model Training

Text test sentence --> Test VAE --> Text Latent Features --> CL Model --> Music Latent Features --> Music VAE --> Music VAE + VA values--> Music 

---

**Diary UI (React + FastAPI) Requirements and Run Steps**

Python packages (backend):
- fastapi
- uvicorn

Node packages (frontend):
- react
- react-dom
- vite
- @vitejs/plugin-react

Run order:
1) Start backend API
```bash
python -m uvicorn backend.main:app --reload
```

2) Start frontend UI
```bash
cd frontend
npm install
npm run dev
```