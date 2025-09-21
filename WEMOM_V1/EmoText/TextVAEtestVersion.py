import torch
import torch.nn as nn

class TextVAEtestVersion(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes=4, pad_idx=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.encoder_rnn = nn.GRU(embed_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc_mu = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_logvar = nn.Linear(hidden_size * 2, hidden_size)
        self.z_to_hidden = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        self.decoder_rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.fc_decoder = nn.Linear(hidden_size, vocab_size)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std, std

    def encode(self, x):
        emb = self.embedding(x)
        _, h = self.encoder_rnn(emb)  # h: (2, batch, hidden)
        h = h.permute(1, 0, 2).reshape(x.size(0), -1)  # (batch, hidden*2)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def decode(self, z, target):
        # z: (batch, hidden)
        h0 = self.z_to_hidden(z).unsqueeze(0)  # (1, batch, hidden)
        emb = self.embedding(target)
        out, _ = self.decoder_rnn(emb, h0)
        logits = self.fc_decoder(out)
        return logits

    def forward(self, text):
        # text: (batch, seq_len)
        text_input = text[:, :-1]
        target_input = text[:, :-1]
        mu, logvar = self.encode(text_input)
        z, std = self.reparameterize(mu, logvar)
        decoder_logits = self.decode(z, target_input)
        logits = self.classifier(z)
        return decoder_logits, mu, std, logits