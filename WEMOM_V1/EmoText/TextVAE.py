import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import json

with open('D:/PolyU/URIS/Part2_projects/WEMOM_V1/EmoText/TextVAE_config.json') as f:
    args = json.load(f)

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.rnn = nn.GRU(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = dropout

    def forward(self, text):
        text_mask = (text != args["PAD_INDEX"])
        text_lens = text_mask.long().sum(dim=1)
        text_embedding = self.embedding(text)
        text = F.dropout(text_embedding, p=self.dropout, training=self.training)

        text_lens, sort_index = text_lens.sort(descending=True)
        text = text.index_select(dim=0, index=sort_index)
        text_lens = text_lens.cpu()

        packed_text = pack_padded_sequence(text, text_lens, batch_first=True, enforce_sorted=True)
        packed_output, final_states = self.rnn(packed_text)

        reorder_index = sort_index.argsort(descending=False)
        final_states = final_states.index_select(dim=1, index=reorder_index)

        num_layers = self.rnn.num_layers
        num_directions = 2 if self.rnn.bidirectional else 1
        final_states = final_states.view(num_layers, num_directions, -1, self.rnn.hidden_size)
        final_states = final_states.permute(2, 0, 1, 3).contiguous() 
        final_states = final_states.view(final_states.size(0), -1)  
        return final_states

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.rnn = nn.GRU(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.output_projection = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, target=None, max_len=50):
    
        num_layers = self.rnn.num_layers
        batch_size = hidden.size(0)
        device = hidden.device
        hidden_init = torch.zeros(num_layers, batch_size, hidden.size(1), device=device)
        hidden_init[0] = hidden

        if target is not None:  
            token_embedding = self.embedding(target)
            output, _ = self.rnn(token_embedding, hidden_init)
            logits = self.output_projection(output)
        else:  
            token = torch.full((batch_size,), args["SOS_INDEX"], dtype=torch.long, device=device)
            logits = []
            ended = torch.zeros(batch_size, dtype=torch.bool, device=device)
            hidden_state = hidden_init
            for t in range(max_len):
                token_embedding = self.embedding(token).unsqueeze(1)
                output, hidden_state = self.rnn(token_embedding, hidden_state)
                token_logit = self.output_projection(output.squeeze(1))
                token = token_logit.argmax(dim=-1)
                logits.append(token_logit)
                ended = ended | (token == args["EOS_INDEX"])
                if ended.all():
                    break
            logits = torch.stack(logits, dim=1)
        return logits

class TextVAE(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout, num_labels=4):
        super(TextVAE, self).__init__()
        self.encoder = Encoder(vocab_size, embed_size, hidden_size, num_layers, dropout)
        self.decoder = Decoder(vocab_size, embed_size, hidden_size, num_layers, dropout)
       
        self.mean_projection = nn.Linear(num_layers*2*hidden_size, hidden_size)
        self.std_projection = nn.Linear(num_layers*2*hidden_size, hidden_size)
        self.decoder_projection = nn.Linear(vocab_size, hidden_size)
        self.classifier = nn.Linear(hidden_size*2, num_labels)

    def forward(self, text):
       
        text_input = text[:, :-1]     
        target_input = text[:, :-1]    
        target_output = text[:, 1:]   
        encoding, mean, std = self.encode(text_input)
        decoder_output = self.decoder(encoding, target_input)

      
        pad_idx = args["PAD_INDEX"]
        mask = (target_output != pad_idx).float() 
        mask_sum = mask.sum(dim=1, keepdim=True) + 1e-8
        decoder_output_softmax = F.softmax(decoder_output, dim=-1)
        decoder_output_mean = (decoder_output_softmax * mask.unsqueeze(-1)).sum(dim=1) / mask_sum
        decoder_features = self.decoder_projection(decoder_output_mean)

        combined_features = torch.cat([encoding, decoder_features], dim=1)
        logits = self.classifier(combined_features)
        return decoder_output, mean, std, logits

    def encode(self, text):
        final_states = self.encoder(text)
        mean = self.mean_projection(final_states)
        std = F.softplus(self.std_projection(final_states)) + 1e-6  
        sample = torch.randn_like(mean)
        encoding = mean + std * sample
        return encoding, mean, std