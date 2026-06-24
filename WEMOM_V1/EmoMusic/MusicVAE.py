'''
The Class for MusicAttrRegGMVAEModel
'''
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
import numpy as np

class MusicAttrRegGMVAE(nn.Module):
    """
    MusicAttrVAE with a GMM as latent prior distribution.
    Modified to include Hierarchical (Conductor-Decoder) structure.
    """
    def __init__(self,
                 roll_dims,
                 rhythm_dims,
                 note_dims,
                 chroma_dims,
                 dynamic_dims,
                 chord_dims,
                 hidden_dims,
                 z_dims,
                 n_step,
                 n_component):

        super(MusicAttrRegGMVAE, self).__init__()

        self.n_component = n_component
        self.latent_dim = z_dims
        self.roll_dims = roll_dims
        self.eps = 100
        
        # Encoders (Experts)
        self.gru_r = nn.GRU(rhythm_dims, hidden_dims, batch_first=True, bidirectional=True)
        self.gru_n = nn.GRU(note_dims, hidden_dims, batch_first=True, bidirectional=True)
        self.gru_d = nn.GRU(dynamic_dims, hidden_dims, batch_first=True, bidirectional=True)
        self.gru_ch = nn.GRU(chord_dims, hidden_dims, batch_first=True, bidirectional=True)
        
        self.linear_enc_c = nn.Sequential(
            nn.Linear(chroma_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, z_dims * 2)
        )

        # Classifiers for GMM
        self.c_r = nn.Linear(z_dims, n_component)
        self.c_n = nn.Linear(z_dims, n_component)
        self.c_d = nn.Linear(z_dims, n_component)
        self.c_ch = nn.Linear(z_dims, n_component)
        self.c_c = nn.Linear(z_dims, n_component)

        # Sub-decoders
        self.gru_d_r = nn.GRU(z_dims + rhythm_dims, hidden_dims, batch_first=True)
        self.gru_d_n = nn.GRU(z_dims + note_dims, hidden_dims, batch_first=True)
        self.gru_d_d = nn.GRU(z_dims + dynamic_dims, hidden_dims, batch_first=True)
        self.gru_d_ch = nn.GRU(z_dims + chord_dims, hidden_dims, batch_first=True)
        
        self.linear_dec_c = nn.Sequential(
            nn.Linear(z_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, chroma_dims)
        )
        
        # Mu and Logvar
        self.mu_r, self.var_r = nn.Linear(hidden_dims * 2, z_dims), nn.Linear(hidden_dims * 2, z_dims)
        self.mu_n, self.var_n = nn.Linear(hidden_dims * 2, z_dims), nn.Linear(hidden_dims * 2, z_dims)
        self.mu_d, self.var_d = nn.Linear(hidden_dims * 2, z_dims), nn.Linear(hidden_dims * 2, z_dims)
        self.mu_ch, self.var_ch = nn.Linear(hidden_dims * 2, z_dims), nn.Linear(hidden_dims * 2, z_dims)
    
        # ==========================================================
        # UPGRADE: Hierarchical Decoder (Conductor - Core Decoder)
        # ==========================================================
        num_z = 5 
        # Latent Z dims = 5 * z_dims
        self.total_z_dim = z_dims * num_z
        
        # Configuration for Hierarchy
        # If total steps = 448 (from reconstruct.py) or n_step
        # Let's say segment_len = 16 (1 bar approx or 1/2 bar)
        self.segment_len = 16 
        
        # 1. Conductor RNN
        # Input: Latent Z (or previous conductor state)
        # Output: Conductor Vector (used to initialize decoder for the segment)
        self.conductor_hidden_dim = hidden_dims
        self.conductor_input_dim = self.total_z_dim
        self.conductor_cell = nn.GRUCell(self.conductor_input_dim, self.conductor_hidden_dim)
        
        # Initial projection for conductor hidden state from Z
        self.z_to_conductor_hidden = nn.Linear(self.total_z_dim, self.conductor_hidden_dim)

        # Conductor output projection -> Decoder Context
        self.conductor_out_to_decoder_ctx = nn.Linear(self.conductor_hidden_dim, hidden_dims)

        # 2. Core Decoder RNN (The "Performer")
        # Takes: [Previous Token, Conductor Context]
        # We concatenate Conductor Context to input at every step of the segment
        self.decoder_hidden_dim = hidden_dims
        self.decoder_input_dim = roll_dims + hidden_dims # Input + Context
        
        self.grucell_g = nn.GRUCell(self.decoder_input_dim, self.decoder_hidden_dim)
        self.grucell_g_2 = nn.GRUCell(self.decoder_hidden_dim, self.decoder_hidden_dim)

        # Initialize decoder hidden state from conductor output
        self.conductor_to_decoder_init = nn.Linear(self.conductor_hidden_dim, self.decoder_hidden_dim)

        # Auxiliary VA head
        self.va_head = nn.Linear(self.total_z_dim, 2)
        
        # Attention (Optional but good)
        # Self-attention over the generated sequence (or conductor sequence)
        self.attn_enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.decoder_hidden_dim, nhead=8, batch_first=True),
            num_layers=1
        )

        self.y_proj = nn.Linear(2, z_dims)
        self._tf_eps = 1.0  

        self.linear_init_r = nn.Linear(z_dims, hidden_dims)
        self.linear_init_n = nn.Linear(z_dims, hidden_dims)
        self.linear_init_d = nn.Linear(z_dims, hidden_dims)
        self.linear_init_ch = nn.Linear(z_dims, hidden_dims)

        self.linear_out_r = nn.Linear(hidden_dims, rhythm_dims)
        self.linear_out_n = nn.Linear(hidden_dims, note_dims)
        self.linear_out_d = nn.Linear(hidden_dims, dynamic_dims)
        self.linear_out_ch = nn.Linear(hidden_dims, chord_dims)
        self.linear_out_g = nn.Linear(hidden_dims, roll_dims) # Final Logits

        self._build_mu_lookup()
        self._build_logvar_lookup(pow_exp=-2)       
    
    def _sampling(self, x):
        idx = x.max(1)[1]
        x = torch.zeros_like(x)
        arange = torch.arange(x.size(0)).long()
        arange = arange.to(x.device)
        x[arange, idx] = 1
        return x

    def encode(self, rhythm, note, dynamic, chord, chroma):
        # rhythm encoder
        x_r = self.gru_r(rhythm)[-1]
        x_r = x_r.transpose_(0, 1).contiguous().view(x_r.size(0), -1)
        mu_r = self.mu_r(x_r)
        var_r = torch.exp(torch.clamp(self.var_r(x_r), max=10))
        
        # note encoder
        x_n = self.gru_n(note)[-1]
        x_n = x_n.transpose_(0, 1).contiguous().view(x_n.size(0), -1)
        mu_n = self.mu_n(x_n)
        var_n = torch.exp(torch.clamp(self.var_n(x_n), max=10))

        # dynamic encoder
        x_d = self.gru_d(dynamic)[-1]
        x_d = x_d.transpose_(0, 1).contiguous().view(x_d.size(0), -1)
        mu_d = self.mu_d(x_d)
        var_d = torch.exp(torch.clamp(self.var_d(x_d), max=10))

        # chord encoder
        x_ch = self.gru_ch(chord)[-1]
        x_ch = x_ch.transpose_(0, 1).contiguous().view(x_ch.size(0), -1)
        mu_ch = self.mu_ch(x_ch)
        var_ch = torch.exp(torch.clamp(self.var_ch(x_ch), max=10))

        # chroma encoder (global)
        x_c = self.linear_enc_c(chroma)
        mu_c = x_c[:, :self.latent_dim]
        var_c = torch.exp(torch.clamp(x_c[:, self.latent_dim:], max=10))

        var_r = torch.clamp(var_r, min=1e-4)
        var_n = torch.clamp(var_n, min=1e-4)
        var_d = torch.clamp(var_d, min=1e-4)
        var_ch = torch.clamp(var_ch, min=1e-4)
        var_c = torch.clamp(var_c, min=1e-4)

        dis_r = Normal(mu_r, var_r)
        dis_n = Normal(mu_n, var_n)
        dis_d = Normal(mu_d, var_d)
        dis_ch = Normal(mu_ch, var_ch)
        dis_c = Normal(mu_c, var_c)
        
        return dis_r, dis_n, dis_d, dis_ch, dis_c

    def sub_decoders(self, rhythm, z_r, note, z_n, dynamic, z_d, chord, z_ch):

        def get_hidden_and_concat_latent(input, z_latent):
            z_latent_stack = torch.stack([z_latent] * input.shape[1], dim=1)
            input_in = torch.cat([input, z_latent_stack], dim=-1)
            return input_in

        rhythm_in = get_hidden_and_concat_latent(rhythm, z_r)
        h_r = self.linear_init_r(z_r).unsqueeze(0)
        rhythm_out = self.gru_d_r(rhythm_in, h_r)[0]
        rhythm_out = F.log_softmax(self.linear_out_r(rhythm_out), 1)

        note_in = get_hidden_and_concat_latent(note, z_n)
        h_n = self.linear_init_n(z_n).unsqueeze(0)
        note_out = self.gru_d_n(note_in, h_n)[0]
        note_out = F.log_softmax(self.linear_out_n(note_out), 1)

        dynamic_in = get_hidden_and_concat_latent(dynamic, z_d)
        h_d = self.linear_init_d(z_d).unsqueeze(0)
        dynamic_out = self.gru_d_d(dynamic_in, h_d)[0]
        dynamic_out = self.linear_out_d(dynamic_out) # Continuous

        chord_in = get_hidden_and_concat_latent(chord, z_ch)
        h_ch = self.linear_init_ch(z_ch).unsqueeze(0)
        chord_out = self.gru_d_ch(chord_in, h_ch)[0]
        chord_out = F.log_softmax(self.linear_out_ch(chord_out), 1)

        return rhythm_out, note_out, dynamic_out, chord_out
    
    def global_decoder(self, z, steps):
        """
        Hierarchical Decoder:
        Z -> Conductor RNN (every 16 steps) -> [Context] -> Decoder RNN (every step) -> Note
        """
        device = z.device
        batch_size = z.size(0)
        
        # Calculate number of segments
        num_segments = steps // self.segment_len
        if steps % self.segment_len != 0:
            num_segments += 1
            
        # Initialize Conductor State
        # We start Conductor with Z
        conductor_hidden = self.z_to_conductor_hidden(z) # [B, conductor_hidden_dim]
        
        # Initialize Decoder Inputs
        # The first input is start token (usually zeros with last bit 1 if configured that way, or just zeros)
        # Here we follow existing code: out[:, -1] = 1.
        out_token = torch.zeros((batch_size, self.roll_dims), device=device)
        out_token[:, -1] = 1.
        
        all_logits = []
        all_hiddens = []
        
        # Conductor Loop
        for seg_i in range(num_segments):
            # Run Conductor for this segment
            # Input to conductor: Z (Global intent). 
            # In advanced versions, feedback from previous segment decoding could be added.
            conductor_hidden = self.conductor_cell(z, conductor_hidden)
            # STABILITY FIX: Tanh activation to prevent hidden state explosion in hierarchical RNN
            conductor_hidden = torch.tanh(conductor_hidden)
            
            # Get Context for Decoder for this segment
            decoder_context = self.conductor_out_to_decoder_ctx(conductor_hidden) # [B, hidden_dims]
            
            # Initialize Decoder Hidden State for this segment 
            # (Resetting or Carrying over? Usually carrying over is better for music continuity, 
            # BUT re-initializing helps structure. Let's Carry Over but inject context)
            if seg_i == 0:
                decoder_hidden_l1 = self.conductor_to_decoder_init(conductor_hidden)
                decoder_hidden_l2 = decoder_hidden_l1.clone() # GRU layer 2
            
            # Decoder Loop (Segment)
            steps_in_this_segment = self.segment_len
            if seg_i == num_segments - 1:
                # clamp to actual remaining steps
                steps_in_this_segment = steps - (seg_i * self.segment_len)
                if steps_in_this_segment < 0: steps_in_this_segment = 0
                
            for step_i in range(steps_in_this_segment):
                # Decoder input: Concatenate previous token AND Conductor Context
                # This ensures the decoder constantly sees the "Plan" for this segment
                decoder_input = torch.cat([out_token, decoder_context], dim=1)
                
                # GRU Layer 1
                decoder_hidden_l1 = self.grucell_g(decoder_input, decoder_hidden_l1)
                
                # GRU Layer 2
                decoder_hidden_l2 = self.grucell_g_2(decoder_hidden_l1, decoder_hidden_l2)
                
                # Output Logits
                logits = F.log_softmax(self.linear_out_g(decoder_hidden_l2), dim=1)
                all_logits.append(logits)
                all_hiddens.append(decoder_hidden_l2)
                
                # Sampling for next step (Teacher Forcing or Sampling)
                if self.training:
                    # TEACHER FORCING
                    p = torch.rand(1).item()
                    if p < self._tf_eps:
                        # Need to get ground truth token at current global index
                        global_idx = seg_i * self.segment_len + step_i
                        if self.sample is not None and global_idx < self.sample.shape[1]:
                            out_token = self.sample[:, global_idx, :].to(device)
                        else:
                            out_token = self._sampling(logits) # Fallback
                    else:
                         out_token = self._sampling(logits)
                else:
                    out_token = self._sampling(logits)
                    
        # Stack logits
        logits_seq = torch.stack(all_logits, dim=1) # [B, Steps, roll_dims]
        
        # Apply lightweight attention on hidden sequence then re-project
        # This acts as a post-generation refinement or smoothing
        if hasattr(self, 'attn_enc'):
            H = torch.stack(all_hiddens, dim=1)  # [B, T, hidden_dims]
            H = self.attn_enc(H)
            logits_seq = F.log_softmax(self.linear_out_g(H), dim=-1)
            
        return logits_seq

    def set_tf_eps(self, eps: float):
        self._tf_eps = float(max(0.0, min(1.0, eps)))

    def _build_mu_lookup(self):
        """
        Follow Xavier initialization as in the paper
        """
        def create_lookup():
            lookup = nn.Embedding(self.n_component, self.latent_dim)
            nn.init.xavier_uniform_(lookup.weight)
            lookup.weight.requires_grad = True
            return lookup

        self.mu_r_lookup = create_lookup()
        self.mu_n_lookup = create_lookup()
        self.mu_d_lookup = create_lookup()
        self.mu_ch_lookup = create_lookup()
        self.mu_c_lookup = create_lookup()

    def _build_logvar_lookup(self, pow_exp=0, logvar_trainable=False):
        """
        Follow Table 7 in the paper
        """
        def create_lookup():
            lookup = nn.Embedding(self.n_component, self.latent_dim)
            init_sigma = np.exp(pow_exp)
            init_logvar = np.log(init_sigma ** 2)
            nn.init.constant_(lookup.weight, init_logvar)
            lookup.weight.requires_grad = logvar_trainable
            return lookup

        self.logvar_r_lookup = create_lookup()
        self.logvar_n_lookup = create_lookup()
        self.logvar_d_lookup = create_lookup()
        self.logvar_ch_lookup = create_lookup()
        self.logvar_c_lookup = create_lookup()

    def approx_qy_x(self, z, mu_lookup, logvar_lookup, n_component):
        def log_gauss_lh(z, mu, logvar):
            llh = - 0.5 * (torch.pow(z - mu, 2) / torch.exp(logvar) + logvar + np.log(2 * np.pi))
            llh = torch.sum(llh, dim=1)  # sum over dimensions
            return torch.clamp(llh, min=-1000.0, max=1000.0) # Clamp LLH to prevent Inf

        device = z.device
        logLogit_qy_x = torch.zeros(z.shape[0], n_component, device=device)  # log-logit of q(y|x)
        for k_i in torch.arange(0, n_component, device=device):
            mu_k, logvar_k = mu_lookup(k_i), logvar_lookup(k_i)
            logLogit_qy_x[:, k_i] = log_gauss_lh(z, mu_k, logvar_k) + np.log(1 / n_component)

        qy_x = torch.nn.functional.softmax(logLogit_qy_x, dim=1)
        return logLogit_qy_x, qy_x

    def forward(self, x, rhythm, note, chroma, dynamic, chord, va=None):
        
        if self.training:
            self.sample = x
        
        # ========================== INFERENCE ====================== #
        dis_r, dis_n, dis_d, dis_ch, dis_c = self.encode(rhythm, note, dynamic, chord, chroma)
        
        def repar(mu, stddev, sigma=1):
            device = mu.device
            eps = Normal(0, sigma).sample(sample_shape=stddev.size()).to(device)
            z = mu + stddev * eps  # reparameterization trick
            return z

        z_r = repar(dis_r.mean, dis_r.stddev)
        z_n = repar(dis_n.mean, dis_n.stddev)
        z_d = repar(dis_d.mean, dis_d.stddev)
        z_ch = repar(dis_ch.mean, dis_ch.stddev)
        z_c = repar(dis_c.mean, dis_c.stddev)

        # Emotion conditioning: add y_emb to each latent
        if va is not None:
            # va expected shape [B, 2]
            y_emb = self.y_proj(va)
            z_r = z_r + y_emb
            z_n = z_n + y_emb
            z_d = z_d + y_emb
            z_ch = z_ch + y_emb
            z_c = z_c + y_emb

        # infer gaussian component
        def infer(z, mu_lookup, logvar_lookup):
            logLogit, qy_x = self.approx_qy_x(z, mu_lookup, logvar_lookup, n_component=self.n_component)
            _, y = torch.max(qy_x, dim=1)
            return logLogit, qy_x, y

        logLogit_r, qy_x_r, y_r = infer(z_r, self.mu_r_lookup, self.logvar_r_lookup)
        logLogit_n, qy_x_n, y_n = infer(z_n, self.mu_n_lookup, self.logvar_n_lookup)
        logLogit_d, qy_x_d, y_d = infer(z_d, self.mu_d_lookup, self.logvar_d_lookup)
        logLogit_ch, qy_x_ch, y_ch = infer(z_ch, self.mu_ch_lookup, self.logvar_ch_lookup)
        logLogit_c, qy_x_c, y_c = infer(z_c, self.mu_c_lookup, self.logvar_c_lookup)

         # ========================== GENERATION ====================== #
        # get sub decoders output
        r_out, n_out, d_out, ch_out = self.sub_decoders(rhythm, z_r, note, z_n, dynamic, z_d, chord, z_ch)
        c_out = self.linear_dec_c(z_c)

        # packaging output
        z = torch.cat([z_r, z_n, z_d, z_ch, z_c], dim=1)      
        
        out = self.global_decoder(z, steps=x.shape[1])
        
        # Auxiliary VA prediction from concatenated latent
        va_pred = self.va_head(z)

        output = (out, r_out, n_out, d_out, ch_out, c_out, va_pred)
        dis = (dis_r, dis_n, dis_d, dis_ch, dis_c)
        z_out = (z_r, z_n, z_d, z_ch, z_c)
        qy_x_out = (qy_x_r, qy_x_n, qy_x_d, qy_x_ch, qy_x_c)
        logLogit_out = (logLogit_r, logLogit_n, logLogit_d, logLogit_ch, logLogit_c)
        y_out = (y_r, y_n, y_d, y_ch, y_c)

        res = (output, dis, z_out, logLogit_out, qy_x_out, y_out)
        return res
