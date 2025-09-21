import torch
import numpy as np
import json

from TextVAE import TextVAE

checkpoint_path = 'D:/PolyU/URIS/Part2_projects/WEMOM_V1/EmoText/params/TextVAE_epoch_50.pt'
save_path = 'D:/PolyU/URIS/Part2_projects/WEMOM_V1/Data/'

with open('D:/PolyU/URIS/Part2_projects/WEMOM_V1/EmoText/TextVAE_config.json') as f:
    args = json.load(f)

id2word = np.load('D:/PolyU/URIS/Part2_projects/WEMOM_V1/Data/All_Data/id2word_text.npy', allow_pickle=True).item()
word2id = {v: k for k, v in id2word.items()}

def sentence_to_tensor(sentence, word2id, max_len=32):
    tokens = sentence.lower().split()
    ids = [word2id.get(token, word2id.get('<unk>', 0)) for token in tokens]
    ids = [word2id.get('<sos>', 1)] + ids + [word2id.get('<eos>', 2)]
    ids = ids[:max_len] + [word2id.get('<pad>', 0)] * (max_len - len(ids))
    return torch.tensor([ids], dtype=torch.long)

def make_latent_feature(sentence):
    model = TextVAE(
        vocab_size=args['vocab_size'],
        embed_size=args['embedding_size'],
        hidden_size=args['hidden_size'],
        num_layers=args['num_layers'],
        dropout=args['dropout']
    )

    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(checkpoint['model_state_dict']) 

    sentence_tensor = sentence_to_tensor(sentence, word2id, max_len=args.get('max_len', 32))
    if torch.cuda.is_available():
        sentence_tensor = sentence_tensor.cuda()

    with torch.no_grad():
        encoding, mean, std = model.encode(sentence_tensor[:, :-1])
        z = encoding.cpu().numpy()

    np.save(f"{save_path}/text_z_single2.npy", z)
    print("Latent space z shape:", z.shape)
    print(f"Saved as {save_path}/text_z_single2.npy")

if __name__ == "__main__":
    sample_sentence = "I feel so sad."
    make_latent_feature(sample_sentence)