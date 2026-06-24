'''
Music FaderNets, GM-VAE model.
'''
import json
import torch
import os
import numpy as np
# from MusicVAE import MusicAttrRegGMVAE
# CRITICAL UPDATE: Using Hierarchical VAE for better long-term structure
from WEMOM_V1.EmoMusic.MusicVAE import MusicAttrRegGMVAE
from torch import optim
from torch.distributions import kl_divergence, Normal
from torch.nn import functional as F
from torch import nn
from music_dataset import VGMIDIDataset, get_vgmidi
from datetime import datetime
from torch.utils.data import DataLoader
import torch

# some initialization
with open('D:/Projects/URIS/URIS/WEMOM_V1/EmoMusic/MusicVAE.json') as f:
    args = json.load(f)

save_path = 'D:/Projects/URIS/URIS/WEMOM_V1/EmoMusic/params/'
# Old checkpoint is incompatible with the new Hierarchical architecture. 
# We must train from scratch.
# resume_path = 'D:/Projects/URIS/URIS/WEMOM_V1/EmoMusic/params/MusicVAE_80.pt' 
resume_path = '' 

if not os.path.exists(save_path):
    os.makedirs(save_path)

# ====================== MODELS ===================== #

EVENT_DIMS = 342
RHYTHM_DIMS = 3
NOTE_DIMS = 16
CHROMA_DIMS = 24
DYNAMIC_DIMS = 5
CHORD_DIMS = 24

model = MusicAttrRegGMVAE(roll_dims=EVENT_DIMS, rhythm_dims=RHYTHM_DIMS, note_dims=NOTE_DIMS, chroma_dims=CHROMA_DIMS, dynamic_dims=DYNAMIC_DIMS, chord_dims=CHORD_DIMS,
                        hidden_dims=args['hidden_dim'], z_dims=args['z_dim'], 
                        n_step=args['time_step'],
                        n_component=args['num_clusters'])

optimizer = optim.Adam(model.parameters(), lr=args['lr'])


if torch.cuda.is_available():
    print('Using: ', torch.cuda.get_device_name(torch.cuda.current_device()))
    model.cuda()
else:
    print('CPU mode')

step, pre_epoch = 0, 0
batch_size = args["batch_size"]
model.train()


# vgmidi dataloaders
print("Loading VGMIDI...")
is_shuffle = True
data_lst, rhythm_lst, note_density_lst, arousal_lst, valence_lst, chroma_lst, dynamic_lst, chord_lst, label_lst = get_vgmidi()
#----------------------------------------------------------------------------------------
# print("Class distribution in arousal labels:")
# print(Counter(arousal_lst))
#----------------------------------------------------------------------------------------
vgm_train_ds_dist = VGMIDIDataset(data_lst, rhythm_lst, note_density_lst, chroma_lst, dynamic_lst, chord_lst, arousal_lst, valence_lst, label_lst, mode="train")
vgm_train_dl_dist = DataLoader(vgm_train_ds_dist, batch_size=batch_size, shuffle=is_shuffle, num_workers=0, drop_last=True)
vgm_val_ds_dist = VGMIDIDataset(data_lst, rhythm_lst, note_density_lst, chroma_lst, dynamic_lst, chord_lst, arousal_lst, valence_lst, label_lst, mode="val")
vgm_val_dl_dist = DataLoader(vgm_val_ds_dist, batch_size=batch_size, shuffle=is_shuffle, num_workers=0, drop_last=True)

print("VGMIDI: Train / Test")
print(len(vgm_train_ds_dist), len(vgm_val_ds_dist))
print()

# ====================== TRAINING ===================== #
def std_normal(shape):
    N = Normal(torch.zeros(shape), torch.ones(shape))
    if torch.cuda.is_available():
        N.loc = N.loc.cuda()
        N.scale = N.scale.cuda()
    return N


def loss_function(out, d,
                r_out, r,
                n_out, n,
                d_out, dyn,
                ch_out, ch,
                c_out, c,
                dis,
                qy_x_out,
                logLogit_out,
                step,
                beta=.1,
                y_label=None):
    '''
    Following loss function defined for GMM-VAE:
    Unsupervised: E[log p(x|z)] - sum{l} q(y_l|X) * KL[q(z|x) || p(z|y_l)] - KL[q(y|x) || p(y)]
    Supervised: E[log p(x|z)] - KL[q(z|x) || p(z|y)]
    '''
    # anneal beta
    if step < 1000:
        beta0 = 0
    else:
        beta0 = min((step - 1000) / 1000 * beta, beta) 

    # Reconstruction loss
    CE_X = F.nll_loss(out.reshape(-1, out.size(-1)),
                    d.reshape(-1), reduction='mean')
    CE_R = F.nll_loss(r_out.reshape(-1, r_out.size(-1)),
                    r.reshape(-1), reduction='mean')
    CE_N = F.nll_loss(n_out.reshape(-1, n_out.size(-1)),
                    n.reshape(-1), reduction='mean')
    
    # Dynamic loss (MSE)
    CE_D = F.mse_loss(d_out, dyn, reduction='mean')
    
    # Chord loss (NLL)
    CE_CH = F.nll_loss(ch_out.reshape(-1, ch_out.size(-1)),
                     torch.argmax(ch, dim=-1).reshape(-1), reduction='mean')
    
    # Chroma loss (MSE)
    CE_C = F.mse_loss(c_out, c, reduction='mean')

    CE = 5 * CE_X + CE_R + CE_N + CE_D + CE_CH + CE_C

    # package output
    dis_r, dis_n, dis_d, dis_ch, dis_c = dis
    qy_x_r, qy_x_n, qy_x_d, qy_x_ch, qy_x_c = qy_x_out
    logLogit_qy_x_r, logLogit_qy_x_n, logLogit_qy_x_d, logLogit_qy_x_ch, logLogit_qy_x_c = logLogit_out
    
    # KLD latent and class loss
    
    def get_kld(dis, mu_lookup, logvar_lookup, y_label):
        device = dis.mean.device
        mu_pz_y, var_pz_y = mu_lookup(y_label.to(device).long()), logvar_lookup(y_label.to(device).long()).exp_()
        dis_pz_y = Normal(mu_pz_y, var_pz_y)
        kld_lat = torch.mean(kl_divergence(dis, dis_pz_y), dim=-1)
        return kld_lat.mean()

    kld_lat_r_total = get_kld(dis_r, model.mu_r_lookup, model.logvar_r_lookup, y_label)
    kld_lat_n_total = get_kld(dis_n, model.mu_n_lookup, model.logvar_n_lookup, y_label)
    kld_lat_d_total = get_kld(dis_d, model.mu_d_lookup, model.logvar_d_lookup, y_label)
    kld_lat_ch_total = get_kld(dis_ch, model.mu_ch_lookup, model.logvar_ch_lookup, y_label)
    kld_lat_c_total = get_kld(dis_c, model.mu_c_lookup, model.logvar_c_lookup, y_label)

    device = qy_x_r.device
    label_clf_loss = nn.CrossEntropyLoss()(qy_x_r, y_label.to(device).long()) + \
                        nn.CrossEntropyLoss()(qy_x_n, y_label.to(device).long()) + \
                        nn.CrossEntropyLoss()(qy_x_d, y_label.to(device).long()) + \
                        nn.CrossEntropyLoss()(qy_x_ch, y_label.to(device).long()) + \
                        nn.CrossEntropyLoss()(qy_x_c, y_label.to(device).long())

    kld_total = kld_lat_r_total + kld_lat_n_total + kld_lat_d_total + kld_lat_ch_total + kld_lat_c_total
    
    # 使用退火后的 KL 权重 beta0
    loss = CE + beta0 * kld_total + label_clf_loss
    
    return loss, CE_X, CE_R, CE_N, CE_D, CE_CH, CE_C, kld_total, label_clf_loss


def latent_regularized_loss_function(z_out, r, n):
    # regularization loss - Pati et al. 2019
    z_r, z_n, z_d, z_ch, z_c = z_out

    z_r_new = z_r
    z_n_new = z_n

    # rhythm regularized
    r_density = r
    device = z_r_new.device
    D_attr_r = torch.from_numpy(np.subtract.outer(r_density, r_density)).to(device).float()
    D_z_r = z_r_new[:, 0].reshape(-1, 1) - z_r_new[:, 0]
    l_r = torch.nn.MSELoss(reduction="mean")(torch.tanh(D_z_r), torch.sign(D_attr_r))
        
    n_density = n
    D_attr_n = torch.from_numpy(np.subtract.outer(n_density, n_density)).to(device).float()
    D_z_n = z_n_new[:, 0].reshape(-1, 1) - z_n_new[:, 0]
    l_n = torch.nn.MSELoss(reduction="mean")(torch.tanh(D_z_n), torch.sign(D_attr_n))

    return l_r, l_n


def train(step, d_oh, r_oh, n_oh, d, r, n, c, dyn, ch, a, v, r_density, n_density, y_label=None):
    
    optimizer.zero_grad()
    # pdb.set_trace()
    # Build continuous VA vector [B, 2]
    va_vec = torch.stack([a.float(), v.float()], dim=1)
    res = model(d_oh, r_oh, n_oh, c, dyn, ch, va=va_vec)

    # package output
    output, dis, z_out, logLogit_out, qy_x_out, y_out = res
    out, r_out, n_out, d_out, ch_out, c_out, va_pred = output

    # calculate gmm loss
    loss, CE_X, CE_R, CE_N, CE_D, CE_CH, CE_C, kld_total, label_clf_loss = loss_function(out, d,
                                        r_out, r,
                                        n_out, n,
                                        d_out, dyn,
                                        ch_out, ch,
                                        c_out, c,
                                        dis,
                                        qy_x_out,
                                        logLogit_out,
                                        step,
                                        beta=args['beta'],
                                        y_label=y_label)

    # Add continuous VA regression loss (MSE) with small weight
    va_target = torch.stack([a, v], dim=1).float().to(va_pred.device)
    lambda_va = args.get('lambda_va', 1.0)
    VA_MSE = F.mse_loss(va_pred, va_target, reduction='mean')
    loss += lambda_va * VA_MSE
    
    # calculate latent regularization loss
    l_r, l_n = torch.Tensor([0]), torch.Tensor([0])
    l_r, l_n = latent_regularized_loss_function(z_out, r_density, n_density)
    loss += l_r + l_n
    
    if torch.isnan(loss):
        print(f"Warning: NaN loss detected at step {step}. Skipping batch.")
        optimizer.zero_grad()
        output = (loss.item(), CE_X.item(), CE_R.item(), CE_N.item(), CE_D.item(), CE_CH.item(), CE_C.item(), l_r.item(), l_n.item(), kld_total.item(), label_clf_loss.item(), VA_MSE.item())
        return step, output

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Check for NaN gradients
    is_nan_grad = False
    for param in model.parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                is_nan_grad = True
                break
    
    if not is_nan_grad:
        optimizer.step()
    else:
        print(f"Warning: NaN gradients detected at step {step}. Skipping step.")
        
    step += 1

    output = (loss.item(), CE_X.item(), CE_R.item(), CE_N.item(), CE_D.item(), CE_CH.item(), CE_C.item(), l_r.item(), l_n.item(), kld_total.item(), label_clf_loss.item(), VA_MSE.item())
    return step, output


def evaluate(step, d_oh, r_oh, n_oh, d, r, n, c, dyn, ch, a, v, r_density, n_density, y_label=None):

    va_vec = torch.stack([a.float(), v.float()], dim=1)
    res = model(d_oh, r_oh, n_oh, c, dyn, ch, va=va_vec)

    # package output
    output, dis, z_out, logLogit_out, qy_x_out, y_out = res
    out, r_out, n_out, d_out, ch_out, c_out, va_pred = output
    z_r, z_n, z_d, z_ch, z_c = z_out

    # calculate gmm loss
    loss, CE_X, CE_R, CE_N, CE_D, CE_CH, CE_C, kld_total, label_clf_loss = loss_function(out, d,
                                        r_out, r,
                                        n_out, n,
                                        d_out, dyn,
                                        ch_out, ch,
                                        c_out, c,
                                        dis,
                                        qy_x_out,
                                        logLogit_out,
                                        step,
                                        beta=args['beta'],
                                        y_label=y_label)

    va_target = torch.stack([a, v], dim=1).float().to(va_pred.device)
    lambda_va = args.get('lambda_va', 1.0)
    VA_MSE = F.mse_loss(va_pred, va_target, reduction='mean')
    loss += lambda_va * VA_MSE
    
    # calculate latent regularization loss
    l_r, l_n = torch.Tensor([0]), torch.Tensor([0])
    l_r, l_n = latent_regularized_loss_function(z_out, r_density, n_density)
    loss += l_r + l_n

    output = loss.item(), CE_X.item(), CE_R.item(), CE_N.item(), CE_D.item(), CE_CH.item(), CE_C.item(), l_r.item(), l_n.item(), kld_total.item(), label_clf_loss.item(), VA_MSE.item()
    return output


def convert_to_one_hot(input, dims):
    # Ensure input is LongTensor for scatter_ index
    if input.dtype != torch.int64:
        input = input.long()
        
    if len(input.shape) > 1:
        input_oh = torch.zeros((input.shape[0], input.shape[1], dims)).to(input.device)
        input_oh = input_oh.scatter_(-1, input.unsqueeze(-1), 1.)
    else:
        input_oh = torch.zeros((input.shape[0], dims)).to(input.device)
        input_oh = input_oh.scatter_(-1, input.unsqueeze(-1), 1.)
    return input_oh


def training_phase(step):
    log_file_path = save_path + 'training_log.csv'
    if not os.path.exists(log_file_path):
        with open(log_file_path, 'w') as f:
            f.write('Epoch,Train_Loss,Test_Loss,Train_D,Train_R,Train_N,Train_Dyn,Train_Ch,Train_C,Train_VA,Train_RD,Train_ND,Train_KLD,Test_D,Test_R,Test_N,Test_Dyn,Test_Ch,Test_C,Test_VA,Test_RD,Test_ND,Test_KLD\n')

    print("D - Data, R - Rhythm, N - Note, Dyn - Dynamic, Ch - Chord, C - Chroma, RD - Reg. Rhythm, ND- Reg. Note, KLD-T: KLD Total")
    for i in range(start_epoch, args['n_epochs'] + 1):
        # Teacher forcing anneal: linearly decay from 1.0 to 0.0 over epochs
        tf_eps = max(0.0, 1.0 - (i - 1) / float(max(1, args['n_epochs'])))
        model.set_tf_eps(tf_eps)
        print("Epoch {} / {}".format(i, args['n_epochs']))

        # =================== TRAIN VGMIDI ======================== #

        batch_loss, batch_test_loss = 0, 0
        b_CE_X, b_CE_R, b_CE_N, b_CE_D, b_CE_CH, b_CE_C, b_VA = 0, 0, 0, 0, 0, 0, 0
        t_CE_X, t_CE_R, t_CE_N, t_CE_D, t_CE_CH, t_CE_C, t_VA = 0, 0, 0, 0, 0, 0, 0
        b_l_r, b_l_n, t_l_r, t_l_n = 0, 0, 0, 0
        b_kld_total, t_kld_total  = 0, 0
        
        # train on vgmidi
        for j, x in enumerate(vgm_train_dl_dist):

            d, r, n, c, dyn, ch, a, v, l, r_density, n_density = x
            d, r, n, c, dyn, ch, a, v = d.cuda().long(), r.cuda().long(), \
                         n.cuda().long(), c.cuda().float(), dyn.cuda().float(), ch.cuda().float(), \
                         a.cuda(), v.cuda()

            # 课程训练：按epoch调整序列步长（从128线性增长到448）
            if bool(args.get('use_curriculum', True)):
                min_steps = int(args.get('curriculum_min_steps', 128))
                max_steps = int(args.get('curriculum_max_steps', 448))
                total_epochs = int(args['n_epochs'])
                curr_steps = int(min_steps + (i - 1) / max(1, total_epochs - 1) * (max_steps - min_steps))
                
                # 限制最大步长，防止显存溢出
                curr_steps = min(curr_steps, 384) 

                # 截断到当前步长
                d = d[:, :curr_steps]
                r = r[:, :curr_steps]
                n = n[:, :curr_steps]
                dyn = dyn[:, :curr_steps, :]
                ch = ch[:, :curr_steps, :]

            d_oh = convert_to_one_hot(d, EVENT_DIMS)
            r_oh = convert_to_one_hot(r, RHYTHM_DIMS)
            n_oh = convert_to_one_hot(n, NOTE_DIMS)
            # c_oh = convert_to_one_hot(c, CHROMA_DIMS)

            step, loss = train(step, d_oh, r_oh, n_oh, d, r, n, c, dyn, ch, a, v, r_density, n_density, y_label=l)
            loss, CE_X, CE_R, CE_N, CE_D, CE_CH, CE_C, l_r, l_n, kld_total, label_clf_loss, VA_MSE = loss
            batch_loss += loss

            b_CE_X += CE_X
            b_CE_R += CE_R
            b_CE_N += CE_N
            b_CE_D += CE_D
            b_CE_CH += CE_CH
            b_CE_C += CE_C
            b_VA += VA_MSE
            b_l_r += l_r
            b_l_n += l_n
            b_kld_total += kld_total
            

            print('batch loss {}/{}: {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} | VA:{:.5f}'.format(j, len(vgm_train_dl_dist), loss, CE_X, CE_R, CE_N, CE_D, CE_CH, CE_C, l_r, l_n, kld_total, VA_MSE))
        
        # evaluate on vgmidi
        for j, x in enumerate(vgm_val_dl_dist):
            d, r, n, c, dyn, ch, a, v, l, r_density, n_density = x
            d, r, n, c, dyn, ch, l, a, v = d.cuda().long(), r.cuda().long(), \
                         n.cuda().long(), c.cuda().float(), dyn.cuda().float(), ch.cuda().float(), l.cuda().long(), \
                         a.cuda(), v.cuda()

            if bool(args.get('use_curriculum', True)):
                min_steps = int(args.get('curriculum_min_steps', 128))
                max_steps = int(args.get('curriculum_max_steps', 448))
                total_epochs = int(args['n_epochs'])
                curr_steps = int(min_steps + (i - 1) / max(1, total_epochs - 1) * (max_steps - min_steps))
                
                # 限制最大步长，防止显存溢出
                curr_steps = min(curr_steps, 384)

                d = d[:, :curr_steps]
                r = r[:, :curr_steps]
                n = n[:, :curr_steps]
                dyn = dyn[:, :curr_steps, :]
                ch = ch[:, :curr_steps, :]

            d_oh = convert_to_one_hot(d, EVENT_DIMS)
            r_oh = convert_to_one_hot(r, RHYTHM_DIMS)
            n_oh = convert_to_one_hot(n, NOTE_DIMS)
            # c_oh = convert_to_one_hot(c, CHROMA_DIMS)

            loss = evaluate(step - 1, d_oh, r_oh, n_oh, d, r, n, c, dyn, ch, a, v, r_density, n_density, y_label=l)
            loss, CE_X, CE_R, CE_N, CE_D, CE_CH, CE_C, l_r, l_n, kld_total, label_clf_loss, VA_MSE = loss
            batch_test_loss += loss
            
            t_CE_X += CE_X
            t_CE_R += CE_R
            t_CE_N += CE_N
            t_CE_D += CE_D
            t_CE_CH += CE_CH
            t_CE_C += CE_C
            t_VA += VA_MSE
            t_l_r += l_r
            t_l_n += l_n
            t_kld_total += kld_total
        
        print('epoch loss: {:.5f}  {:.5f}'.format(batch_loss / len(vgm_train_dl_dist),
                                                  batch_test_loss / len(vgm_val_dl_dist)))

        print("train loss by term - D: {:.4f} R: {:.4f} N: {:.4f} Dyn: {:.4f} Ch: {:.4f} C: {:.4f} VA: {:.4f} RD: {:.4f} ND: {:.4f} KLD-T: {:.4f} ".format(
            b_CE_X / len(vgm_train_dl_dist), b_CE_R / len(vgm_train_dl_dist), 
            b_CE_N / len(vgm_train_dl_dist),
            b_CE_D / len(vgm_train_dl_dist), b_CE_CH / len(vgm_train_dl_dist), b_CE_C / len(vgm_train_dl_dist), b_VA / len(vgm_train_dl_dist),
            b_l_r / len(vgm_train_dl_dist), b_l_n / len(vgm_train_dl_dist),
            b_kld_total / len(vgm_train_dl_dist)
        ))
        print("test loss by term - D: {:.4f} R: {:.4f} N: {:.4f} Dyn: {:.4f} Ch: {:.4f} C: {:.4f} VA: {:.4f} RD: {:.4f} ND: {:.4f} KLD-T: {:.4f} ".format(
            t_CE_X / len(vgm_val_dl_dist), t_CE_R / len(vgm_val_dl_dist), 
            t_CE_N / len(vgm_val_dl_dist),
            t_CE_D / len(vgm_val_dl_dist), t_CE_CH / len(vgm_val_dl_dist), t_CE_C / len(vgm_val_dl_dist), t_VA / len(vgm_val_dl_dist),
            t_l_r / len(vgm_val_dl_dist), t_l_n / len(vgm_val_dl_dist),
            t_kld_total / len(vgm_val_dl_dist)
        ))

        with open(log_file_path, 'a') as f:
            f.write("{},{:.5f},{:.5f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(
                i, batch_loss / len(vgm_train_dl_dist), batch_test_loss / len(vgm_val_dl_dist),
                b_CE_X / len(vgm_train_dl_dist), b_CE_R / len(vgm_train_dl_dist), b_CE_N / len(vgm_train_dl_dist),
                b_CE_D / len(vgm_train_dl_dist), b_CE_CH / len(vgm_train_dl_dist), b_CE_C / len(vgm_train_dl_dist), b_VA / len(vgm_train_dl_dist),
                b_l_r / len(vgm_train_dl_dist), b_l_n / len(vgm_train_dl_dist), b_kld_total / len(vgm_train_dl_dist),
                t_CE_X / len(vgm_val_dl_dist), t_CE_R / len(vgm_val_dl_dist), t_CE_N / len(vgm_val_dl_dist),
                t_CE_D / len(vgm_val_dl_dist), t_CE_CH / len(vgm_val_dl_dist), t_CE_C / len(vgm_val_dl_dist), t_VA / len(vgm_val_dl_dist),
                t_l_r / len(vgm_val_dl_dist), t_l_n / len(vgm_val_dl_dist), t_kld_total / len(vgm_val_dl_dist)
            ))

        if i % 5 == 0:
            save_epoch_path = save_path + ("{}.pt".format(args['name'] + "_" + str(i)))
            torch.save({
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': batch_loss
            }, save_epoch_path)
            print("Saving model to ... ", save_epoch_path)


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_path_timing = save_path + ("{}.pt".format(args['name'] + "_" + timestamp))
    torch.save({
        'epoch': 100,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # 'loss': loss
    }, save_path_timing)
    print('Model saved as {}!'.format(save_path_timing))


# def evaluation_phase():
#     print("Evaluate")
#     if torch.cuda.is_available():
#         model.cuda()

#     if os.path.exists(save_path):
#         print("Loading {}".format(save_path))
#         model.load_state_dict(torch.load(save_path))
    
#     def run(dl, is_vgmidi=False):
        
#         t_CE_X, t_CE_R, t_CE_N = 0, 0, 0
#         t_l_r, t_l_n = 0, 0
#         t_kld_latent, t_kld_class = 0, 0
#         t_acc_x, t_acc_r, t_acc_n, t_acc_a_r, t_acc_a_n = 0, 0, 0, 0, 0
#         data_len = 0

#         for i, x in tqdm(enumerate(dl), total=len(dl)):
#             d, r, n, c, a, v, r_density, n_density = x
#             d, r, n, c = d.cuda().long(), r.cuda().long(), \
#                          n.cuda().long(), c.cuda().long()

#             d_oh = convert_to_one_hot(d, EVENT_DIMS)
#             r_oh = convert_to_one_hot(r, RHYTHM_DIMS)
#             n_oh = convert_to_one_hot(n, NOTE_DIMS)

#             res = model(d_oh, r_oh, n_oh, c)

#             # package output
#             output, dis, z_out, logLogit_out, qy_x_out, y_out = res
#             out, r_out, n_out, _, _ = output
#             z_r, z_n = z_out

#             if not is_vgmidi:
#                 # calculate gmm loss
#                 loss, CE_X, CE_R, CE_N, kld_lat_r_total, kld_lat_n_total, \
#                     kld_cls_r, kld_cls_n = loss_function(out, d,
#                                                     r_out, r,
#                                                     n_out, n,
#                                                     dis,
#                                                     qy_x_out,
#                                                     logLogit_out,
#                                                     step,
#                                                     beta=args['beta'])
            
#             else:
#                 # calculate gmm loss
#                 loss, CE_X, CE_R, CE_N, kld_lat_r_total, kld_lat_n_total, \
#                     kld_cls_r, kld_cls_n = loss_function(out, d,
#                                                     r_out, r,
#                                                     n_out, n,
#                                                     dis,
#                                                     qy_x_out,
#                                                     logLogit_out,
#                                                     step,
#                                                     beta=args['beta'],
#                                                     is_supervised=True,
#                                                     y_label=a)
            
#             # calculate latent regularization loss
#             l_r, l_n = latent_regularized_loss_function(z_out, r_density, n_density)

#             # adversarial loss
#             kld_latent, kld_class = kld_lat_r_total.item() +  kld_lat_n_total.item(), \
#                                     kld_cls_r.item() + kld_cls_n.item()
            
#             t_CE_X += CE_X
#             t_CE_R += CE_R
#             t_CE_N += CE_N
#             t_l_r += l_r.item()
#             t_l_n += l_n.item()
#             t_kld_latent += kld_latent
#             t_kld_class += kld_class
            
#             # calculate accuracy
#             def acc(a, b, t, trim=False):
#                 a = torch.argmax(a, dim=-1).squeeze().cpu().detach().numpy()
#                 b = b.squeeze().cpu().detach().numpy()

#                 b_acc = 0
#                 for i in range(len(a)):
#                     a_batch = a[i]
#                     b_batch = b[i]

#                     if trim:
#                         b_batch = np.trim_zeros(b_batch)
#                         a_batch = a_batch[:len(b_batch)]

#                     correct = 0
#                     for j in range(len(a_batch)):
#                         if a_batch[j] == b_batch[j]:
#                             correct += 1
#                     acc = correct / len(a_batch)
#                     b_acc += acc
                
#                 return b_acc

#             acc_x, acc_r, acc_n = acc(out, d, "d", trim=True), \
#                                   acc(r_out, r, "r"), acc(n_out, n, "n")
#             data_len += out.shape[0]

#             if is_vgmidi:
#                 qy_x_r, qy_x_n = qy_x_out
#                 qy_x_r, qy_x_n = torch.argmax(qy_x_r, axis=-1).cpu().detach().numpy(), \
#                                 torch.argmax(qy_x_n, axis=-1).cpu().detach().numpy()
#                 acc_q_x_r = accuracy_score(a.cpu().detach().numpy(), qy_x_r)
#                 acc_q_x_n = accuracy_score(a.cpu().detach().numpy(), qy_x_n)
#             else:
#                 acc_q_x_r, acc_q_x_n = 0, 0

#             t_acc_x += acc_x
#             t_acc_r += acc_r
#             t_acc_n += acc_n
#             t_acc_a_r += acc_q_x_r
#             t_acc_a_n += acc_q_x_n

#         # Print results
#         print("CE: {:.4}  {:.4}  {:.4}".format(t_CE_X / len(dl),
#                                                     t_CE_R / len(dl), 
#                                                     t_CE_N / len(dl)))
        
#         print("Regularized: {:.4}  {:.4}".format(t_l_r / len(dl),
#                                                 t_l_n / len(dl)))

#         # print("Adversarial: {:.4}  {:.4}".format(t_l_adv_r / len(dl),
#         #                                         t_l_adv_n / len(dl)))
        
#         print("Acc: {:.4}  {:.4}  {:.4}  {:.4}  {:.4}".format(t_acc_x / data_len,
#                                                             t_acc_r / data_len, 
#                                                             t_acc_n / data_len,
#                                                             t_acc_a_r / data_len,
#                                                             t_acc_a_n / data_len))

#     # dl = DataLoader(train_ds_dist, batch_size=128, shuffle=False, num_workers=0)
#     # run(dl)
#     # dl = DataLoader(test_ds_dist, batch_size=128, shuffle=False, num_workers=0)
#     # run(dl)
#     # dl = DataLoader(vgm_train_ds_dist, batch_size=32, shuffle=False, num_workers=0)
#     # run(dl, is_vgmidi=True)
#     # dl = DataLoader(vgm_test_ds_dist, batch_size=32, shuffle=False, num_workers=0)
#     # run(dl, is_vgmidi=True)

if os.path.exists(resume_path):
    print(f"Resuming training from {resume_path}...")
    checkpoint = torch.load(resume_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    step = checkpoint.get('step', 0)
    print(f"Resumed from epoch {start_epoch}")
else:
    print("No checkpoint found. Starting training from scratch.")
    start_epoch = 1

training_phase(step)
# evaluation_phase()

