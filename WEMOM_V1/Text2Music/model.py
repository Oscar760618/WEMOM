import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from Text2Music.losses import SupConLoss
    import Text2Music.config as config
except ImportError:
    from losses import SupConLoss
    import config

class CLloss(nn.Module):
    def __init__(self, txt_dim, mus_dim, tau=0.1):
        super().__init__()

        # txt -> MID
        self.txtpro1 = self.projector(in_dim=txt_dim, out_dim=config.MID_DIM, use_bias=False, use_bn=True, relu=True)
        # MID -> CL
        self.txtpro2 = self.projector(in_dim=config.MID_DIM, out_dim=config.CL_DIM, use_bias=False, use_bn=True, relu=True)
        
        # Simplified: Removed 3rd layer (txtpro3) to reduce complexity
        # self.txtpro3 = self.projector(in_dim=config.CL_DIM, out_dim=config.CL_DIM, use_bias=False, use_bn=True, relu=True)
        
        # MUS -> CL
        self.muspro = self.projector(in_dim=mus_dim, out_dim=config.CL_DIM, use_bias=False, use_bn=True, relu=True)
        # CL -> MUS
        # CRITICAL UPGRADE: Changed to MLP (Multi-Layer Perceptron).
        # A single Linear layer is too simple to map from Contrastive Space (256) to Music Latent Space (640).
        # We add a hidden layer (512) and ReLU to learn non-linear relationships.
        self.demuspro = nn.Sequential(
            nn.Linear(config.CL_DIM, config.MID_DIM),
            nn.LeakyReLU(0.2, inplace=True), 
            nn.Dropout(0.5), # Increased Dropout to 0.5
            nn.Linear(config.MID_DIM, mus_dim)
        )
        # feature and label inputted
        
        self.loss_func = SupConLoss(temperature=tau)
        self.rec_loss = nn.MSELoss()
        self.last = None

    def forward(self, pos_txt, neg_txt, pos_muse, neg_muse, training=True):
        
        
        
        if training == False:
            pos_txt_emb1 = self.txtpro1(pos_txt)
            pos_txt_emb2 = self.txtpro2(pos_txt_emb1)
            # pos_txt_emb3 = self.txtpro3(pos_txt_emb2)
            pro_muse = self.demuspro(pos_txt_emb2)
            # Removed [0] to support batch processing and consistent dimensions
            return pro_muse
        
        # Remove manual squeeze which might be incorrect for Batch Size > 1
        # pos_txt, neg_txt, pos_muse, neg_muse = pos_txt.squeeze(), neg_txt.squeeze(), pos_muse.squeeze(), neg_muse.squeeze()
        
        # Instead, we need to handle shapes carefully.
        # Input shape from DataLoader(batch_size=N): [N, num_pos_samples, feature_dim]
        # Or if num_pos_samples is 1: [N, 1, feature_dim]
        
        # If the dataset logic provides [num_pos_samples, feature_dim] per item
        # Then batch is [N, num_pos_samples, feature_dim]
        
        # BatchNorm1d expects [N, C] or [N, C, L]
        # Here we likely have [N*num_pos, feature_dim] effectively for the encoder
        
        def flatten_batch(x):
            if x.dim() == 3:
                return x.reshape(-1, x.size(-1)) # [N*S, Dim]
            return x

        pos_txt = flatten_batch(pos_txt)
        neg_txt = flatten_batch(neg_txt)
        pos_muse = flatten_batch(pos_muse)
        neg_muse = flatten_batch(neg_muse)

        # TXT -> MID -> CL -> CL（pos&neg)
        pos_txt_emb1 = self.txtpro1(pos_txt)
        pos_txt_emb2 = self.txtpro2(pos_txt_emb1)
        # pos_txt_emb3 = self.txtpro3(pos_txt_emb2)

        neg_txt_emb1 = self.txtpro1(neg_txt)
        neg_txt_emb2 = self.txtpro2(neg_txt_emb1)
        # neg_txt_emb3 = self.txtpro3(neg_txt_emb2)
        
        # MUS -> CL （pos&neg)
        pos_muse_emb = self.muspro(pos_muse)
        neg_muse_emb = self.muspro(neg_muse)

        # Assuming flatten_batch collapsed [B, Num, Dim] to [B*Num, Dim]
        current_batch_size = pos_txt_emb2.shape[0]

        device = pos_txt_emb2.device
        
        # Feature Construction for SupCon
        intra_feature = torch.cat([pos_txt_emb2, pos_muse_emb, neg_muse_emb], dim=0).unsqueeze(dim=1)
        intra_feature = F.normalize(intra_feature, dim=2)
        
        # Label Construction
        labels_real = torch.arange(current_batch_size, device=device)
        labels_neg = torch.arange(current_batch_size, 2 * current_batch_size, device=device)

        intra_label = torch.cat([labels_real, labels_real, labels_neg], dim=0) 
        intra_loss = self.loss_func(intra_feature, intra_label)

        # txt intra-model
        txt_feature = torch.cat([pos_txt_emb2, neg_txt_emb2], dim=0).unsqueeze(dim=1)
        txt_feature = F.normalize(txt_feature, dim=2)
        txt_label = torch.cat([labels_real, labels_neg], dim=0)
        txt_loss = self.loss_func(txt_feature, txt_label)

        # muse intra-model
        muse_feature = torch.cat([pos_muse_emb, neg_muse_emb], dim=0).unsqueeze(dim=1)
        muse_feature = F.normalize(muse_feature, dim=2)
        muse_label = torch.cat([labels_real, labels_neg], dim=0)
        muse_loss = self.loss_func(muse_feature, muse_label)

        # reconstruction (Music -> CL -> Music)
        true_muse = torch.cat([pos_muse, neg_muse], dim=0)
        muse = torch.cat([pos_muse_emb, neg_muse_emb], dim=0)

        re_muse = self.demuspro(muse)
        re_loss = self.rec_loss(true_muse, re_muse)

        # Cross-Modal Reconstruction (Text -> CL -> Music)
        # This is the most important loss for GENERATION!
        # We ensure: Text_Emb -> Decoder -> Music Latent
        # Fixed: pos_txt_emb3 -> pos_txt_emb2
        re_cross_muse = self.demuspro(pos_txt_emb2)
        cross_rec_loss = self.rec_loss(pos_muse, re_cross_muse)

        return intra_loss, txt_loss, muse_loss, re_loss, cross_rec_loss

    def projector(self, in_dim, out_dim, use_bias=True, use_bn=False, relu=False):
        net = nn.Sequential()
        net.add_module("FC1", nn.Linear(in_dim, out_dim, bias=use_bias))
        if use_bn:
            net.add_module("BN", nn.BatchNorm1d(out_dim))
        if relu:
            net.add_module("ReLU", nn.ReLU())
        return net

