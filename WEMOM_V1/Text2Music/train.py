import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from dataset import MapDataset
from model import CLloss
import config
import os

save_path = 'D:/PolyU/URIS/Part2_projects/WEMOM_V1/Text2Music/params/'

def train():

    clnet = CLloss(txt_dim=128, mus_dim=256, tau=0.1)
    clnet.cuda()
    clnet.train()

    optimizer = optim.Adam(clnet.parameters(), lr = config.init_lr)
    dataset = MapDataset(mode="train")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for epoch in range(1, config.EPOCH+1):

        all_loss = 0
        all_intra_loss = 0
        all_txt_loss = 0
        all_muse_loss = 0
        all_re_loss = 0

        for idx, (txt_num, pos_txt, neg_txt, pos_muse, neg_muse) in enumerate(dataloader):
           
            optimizer.zero_grad()

            pos_txt, neg_txt, pos_muse, neg_muse = pos_txt.cuda(), neg_txt.cuda(), pos_muse.cuda(), neg_muse.cuda()

            intra_loss, txt_loss, muse_loss, re_loss = clnet.forward(pos_txt, neg_txt, pos_muse, neg_muse)

            # intra_loss, txt_loss, mus_loss: SupConloss; re_loss: MSE loss;
            loss = intra_loss + txt_loss + muse_loss + 10 * re_loss
            loss.backward()
            optimizer.step()

            # print("batch loss | {}/{} | {} {} {} {} {}".format(idx, len(dataloader), loss.item(), intra_loss.item(), txt_loss.item(), muse_loss.item(), re_loss.item()))

            all_loss += loss
            all_intra_loss += intra_loss
            all_txt_loss += txt_loss
            all_muse_loss += muse_loss
            all_re_loss += re_loss

        all_loss = all_loss / len(dataloader)
        all_intra_loss = all_intra_loss / len(dataloader)
        all_txt_loss = all_txt_loss / len(dataloader)
        all_muse_loss = all_muse_loss / len(dataloader)
        all_re_loss = all_re_loss / len(dataloader)
        

        if epoch % 100 == 0:
            print("epoch loss | {}/{} | {} {} {} {} {}".format(
            epoch, config.EPOCH, all_loss, all_intra_loss, all_txt_loss, all_muse_loss, all_re_loss))

        if epoch % 200 == 0:
            save_epoch_path = os.path.join(save_path, f"params_{epoch}_{all_loss:.3f}.pt")
            os.makedirs(os.path.dirname(save_epoch_path), exist_ok=True)  
            torch.save({
                'epoch': epoch,
                'model_state_dict': clnet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': all_loss
            }, save_epoch_path)
            print("Saving model to...", save_epoch_path)

if __name__ == "__main__":
    train()