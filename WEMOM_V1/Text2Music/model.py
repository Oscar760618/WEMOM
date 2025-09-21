import torch
import torch.nn as nn
import torch.nn.functional as F
from losses import SupConLoss
import config

class CLloss(nn.Module):
    def __init__(self, txt_dim, mus_dim, tau=0.1):
        super().__init__()

        # txt -> MID
        self.txtpro1 = self.projector(in_dim=txt_dim, out_dim=config.MID_DIM, use_bias=False, use_bn=True, relu=True)
        # MID -> CL
        self.txtpro2 = self.projector(in_dim=config.MID_DIM, out_dim=config.CL_DIM, use_bias=False, use_bn=True, relu=True)
        # CL -> CL
        self.txtpro3 = self.projector(in_dim=config.CL_DIM, out_dim=config.CL_DIM, use_bias=False, use_bn=True, relu=True)
        # MUS -> CL
        self.muspro = self.projector(in_dim=mus_dim, out_dim=config.CL_DIM, use_bias=False, use_bn=True, relu=True)
        # CL -> MUS
        self.demuspro = self.projector(in_dim=config.CL_DIM, out_dim=mus_dim, use_bias=False, use_bn=True, relu=True)
        # feature and label inputted
        
        self.loss_func = SupConLoss(temperature=tau)
        self.rec_loss = nn.MSELoss()
        self.last = None

    def forward(self, pos_txt, neg_txt, pos_muse, neg_muse, training=True):
        
        
        
        if training == False:
            pos_txt_emb1 = self.txtpro1(pos_txt)
            pos_txt_emb2 = self.txtpro2(pos_txt_emb1)
            pos_txt_emb3 = self.txtpro3(pos_txt_emb2)
            pro_muse = self.demuspro(pos_txt_emb3)[0]
            return pro_muse
        
        pos_txt, neg_txt, pos_muse, neg_muse = pos_txt.squeeze(), neg_txt.squeeze(), pos_muse.squeeze(), neg_muse.squeeze()

        # TXT -> MID -> CL -> CL（pos&neg)
        pos_txt_emb1 = self.txtpro1(pos_txt)
        pos_txt_emb2 = self.txtpro2(pos_txt_emb1)
        pos_txt_emb3 = self.txtpro3(pos_txt_emb2)

        neg_txt_emb1 = self.txtpro1(neg_txt)
        neg_txt_emb2 = self.txtpro2(neg_txt_emb1)
        neg_txt_emb3 = self.txtpro3(neg_txt_emb2)
        
        # MUS -> CL （pos&neg)
        pos_muse_emb = self.muspro(pos_muse)
        neg_muse_emb = self.muspro(neg_muse)

        # inter-model: 
        # POS_TXT_CL(0):4, POS_MUS_CL(0):4, NEG_MUS_CL(1)：4
        intra_feature = torch.cat([pos_txt_emb3, pos_muse_emb, neg_muse_emb], dim=0).unsqueeze(dim=1)
        intra_feature = F.normalize(intra_feature, dim=2)
        intra_label = torch.Tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
        intra_loss = self.loss_func(intra_feature, intra_label)
    
        # txt intra-model
        # POS_txt_CL(0):4, NEG_txt_CL(1):4
        txt_feature = torch.cat([pos_txt_emb3, neg_txt_emb3], dim=0).unsqueeze(dim=1)
        txt_feature = F.normalize(txt_feature, dim=2)
        txt_label = torch.Tensor([0, 0, 0, 0, 1, 1, 1, 1])
        txt_loss = self.loss_func(txt_feature, txt_label)

        # muse intra-model
        # POS_MUS_CL(0):4, NEG_MUS_CL(1):4
        muse_feature = torch.cat([pos_muse_emb, neg_muse_emb], dim=0).unsqueeze(dim=1)
        muse_feature = F.normalize(muse_feature, dim=2)
        muse_label = torch.Tensor([0, 0, 0, 0, 1, 1, 1, 1])
        muse_loss = self.loss_func(muse_feature, muse_label)

        # reconstruction
        true_muse = torch.cat([pos_muse.squeeze(), neg_muse.squeeze()], dim=0)
        muse = torch.cat([pos_muse_emb, neg_muse_emb], dim=0)

        re_muse = self.demuspro(muse)
        re_loss = self.rec_loss(true_muse, re_muse)

        if training == False:
            pos_txt_emb1 = self.txtpro1(pos_txt)
            pos_txt_emb2 = self.txtpro2(pos_txt_emb1)
            pos_txt_emb3 = self.txtpro3(pos_txt_emb2)
            pro_muse = self.demuspro(pos_txt_emb3)[0]
            return pro_muse
            

        return intra_loss, txt_loss, muse_loss, re_loss

    def projector(self, in_dim, out_dim, use_bias=True, use_bn=False, relu=False):
        net = nn.Sequential()
        net.add_module("FC1", nn.Linear(in_dim, out_dim, bias=use_bias))
        if use_bn:
            net.add_module("BN", nn.BatchNorm1d(out_dim))
        if relu:
            net.add_module("ReLU", nn.ReLU())
        return net

