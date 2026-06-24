import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from dataset import MapDataset
from model import CLloss
import config
import os

save_path = 'D:/Projects/URIS/URIS/WEMOM_V1/Text2Music/params/'

def train():

    clnet = CLloss(txt_dim=config.TXT_DIM, mus_dim=config.MUS_DIM, tau=0.1)
    clnet.cuda()
    clnet.train()

    # Added weight_decay for regularization
    optimizer = optim.Adam(clnet.parameters(), lr = config.init_lr, weight_decay=1e-4)
    # Added scheduler to decay LR when loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=True)
    
    train_dataset = MapDataset(mode="train")
    # Increased batch size for stable gradients
    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    val_dataset = MapDataset(mode="test")
    val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    for epoch in range(1, config.EPOCH+1):
        clnet.train() # Switch to training mode
        all_loss = 0
        all_intra_loss = 0
        all_txt_loss = 0
        all_muse_loss = 0
        all_re_loss = 0

        for idx, (txt_num, pos_txt, neg_txt, pos_muse, neg_muse) in enumerate(train_dataloader):
           
            optimizer.zero_grad()

            pos_txt, neg_txt, pos_muse, neg_muse = pos_txt.cuda(), neg_txt.cuda(), pos_muse.cuda(), neg_muse.cuda()

            intra_loss, txt_loss, muse_loss, re_loss, cross_rec_loss = clnet.forward(pos_txt, neg_txt, pos_muse, neg_muse)

            # intra_loss, txt_loss, mus_loss: SupConloss; re_loss: MSE loss;
            # Reduced cross_rec_loss weight from 50 to 5 to balance contrastive learning vs reconstruction
            loss = intra_loss + txt_loss + muse_loss + 10 * re_loss + 5 * cross_rec_loss
            loss.backward()
            
            # Gradient Clipping to prevent exploding gradients (NaNs)
            torch.nn.utils.clip_grad_norm_(clnet.parameters(), max_norm=1.0)
            
            optimizer.step()

            # print("batch loss | {}/{} | {} {} {} {} {} {}".format(idx, len(dataloader), loss.item(), intra_loss.item(), txt_loss.item(), muse_loss.item(), re_loss.item(), cross_rec_loss.item()))

            all_loss += loss
            all_intra_loss += intra_loss
            all_txt_loss += txt_loss
            all_muse_loss += muse_loss
            all_re_loss += re_loss

        all_loss = all_loss / len(train_dataloader)
        all_intra_loss = all_intra_loss / len(train_dataloader)
        all_txt_loss = all_txt_loss / len(train_dataloader)
        all_muse_loss = all_muse_loss / len(train_dataloader)
        all_re_loss = all_re_loss / len(train_dataloader)
        
        # Validation Loop
        clnet.eval()
        val_loss = 0
        with torch.no_grad():
            for idx, (txt_num, pos_txt, neg_txt, pos_muse, neg_muse) in enumerate(val_dataloader):
                pos_txt, neg_txt, pos_muse, neg_muse = pos_txt.cuda(), neg_txt.cuda(), pos_muse.cuda(), neg_muse.cuda()
                intra_loss, txt_loss, muse_loss, re_loss, cross_rec_loss = clnet.forward(pos_txt, neg_txt, pos_muse, neg_muse)
                batch_loss = intra_loss + txt_loss + muse_loss + 10 * re_loss + 50 * cross_rec_loss
                val_loss += batch_loss
        
        val_loss = val_loss / len(val_dataloader)

        # Step the scheduler based on VAL loss, not Train loss (to prevent overfitting)
        scheduler.step(val_loss)

        if epoch % 100 == 0:
            print("Epoch {}/{} | Train Loss: {:.4f} | Val Loss: {:.4f}".format(
            epoch, config.EPOCH, all_loss, val_loss))

        if epoch % 500 == 0:
            save_epoch_path = os.path.join(save_path, f"params_{epoch}_{all_loss:.3f}.pt")
            os.makedirs(os.path.dirname(save_epoch_path), exist_ok=True)  
            torch.save({
                'epoch': epoch,
                'model_state_dict': clnet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': all_loss
            }, save_epoch_path)
            print("Saving model to...", save_epoch_path)
            
            # Save latest model for easy access
            latest_path = os.path.join(save_path, "params_latest.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': clnet.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': all_loss
            }, latest_path)

if __name__ == "__main__":
    train()