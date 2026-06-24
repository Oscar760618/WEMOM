'''
code for getting data from saved numpy arrays and the dataset class for the VGMIDI dataset
'''
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from collections import Counter
from sklearn.model_selection import train_test_split

VA_data_path = 'D:/Projects/URIS/URIS/WEMOM_V1/Text_dataset/labels.csv'
midi_data_path = 'D:/Projects/URIS/URIS/WEMOM_V1/Data/Mus_dataset'

save_path = 'D:/Projects/URIS/URIS/WEMOM_V1/Data/Saves/'
path = 'D:/Projects/URIS/URIS/EIMG-main/EIMG-main/Dataset/dataset/'
def get_vgmidi(mode="VAE"):
    '''
    get the MIDI data from the saved numpy arrays and put them in VGMIDIDataset class.
    '''
    if mode == "VAE":

        data_lst = np.load(save_path + "Music_VAE_data_lst.npy", allow_pickle=True)
        rhythm_lst = np.load(save_path + "Music_VAE_rhythm_lst.npy", allow_pickle=True)
        note_density_lst = np.load(save_path + "Music_VAE_note_density_lst.npy", allow_pickle=True)
        chroma_lst = np.load(save_path + "Music_VAE_chroma_lst.npy", allow_pickle=True)
        dynamic_lst = np.load(save_path + "Music_VAE_dynamic_lst.npy", allow_pickle=True)
        chord_lst = np.load(save_path + "Music_VAE_chord_lst.npy", allow_pickle=True)
        valence_lst = np.load(save_path + "Music_VAE_valence_lst.npy")
        arousal_lst = np.load(save_path + "Music_VAE_arousal_lst.npy")
        label_lst = np.load(save_path + "Music_VAE_label_lst.npy")
    
    if mode == "CL":

        data_lst = np.load(save_path + "Music_CL_data_lst.npy", allow_pickle=True)
        rhythm_lst = np.load(save_path + "Music_CL_rhythm_lst.npy", allow_pickle=True)
        note_density_lst = np.load(save_path + "Music_CL_note_density_lst.npy", allow_pickle=True)
        chroma_lst = np.load(save_path + "Music_CL_chroma_lst.npy", allow_pickle=True)
        dynamic_lst = np.load(save_path + "Music_CL_dynamic_lst.npy", allow_pickle=True)
        chord_lst = np.load(save_path + "Music_CL_chord_lst.npy", allow_pickle=True)
        valence_lst = np.load(save_path + "Music_CL_valence_lst.npy")
        arousal_lst = np.load(save_path + "Music_CL_arousal_lst.npy")
        label_lst = np.load(save_path + "Music_CL_label_lst.npy")

    
    print("Shapes for: Data, Rhythm Density, Note Density, Chroma, Dynamic, Chord")
    print(data_lst.shape, rhythm_lst.shape, note_density_lst.shape, chroma_lst.shape, dynamic_lst.shape, chord_lst.shape)
    print("Shapes for: Arousal, Valence, Label")
    print(arousal_lst.shape, valence_lst.shape, label_lst.shape)

    return data_lst, rhythm_lst, note_density_lst, arousal_lst, valence_lst, chroma_lst, dynamic_lst, chord_lst, label_lst
    
class VGMIDIDataset(Dataset):
    '''
    VGMIDI dataset loader.
    '''
    def __init__(self, data, rhythm, note, chroma, dynamic, chord, arousal, valence, label, mode="train"):
        super().__init__()
        
        indexed = []

        random_state = 20
        test_ratio = 0.1

        data_train, data_test, rhythm_train, rhythm_test, note_train, note_test, chroma_train, chroma_test, dynamic_train, dynamic_test, chord_train, chord_test,\
        arousal_train, arousal_test, valence_train, valence_test, label_train, label_test\
        = train_test_split(data, rhythm, note, chroma, dynamic, chord, arousal, valence, label, test_size=test_ratio, random_state=random_state)

        train_data = data_train, rhythm_train, note_train, chroma_train, dynamic_train, chord_train, arousal_train, valence_train, label_train
        test_data = data_test, rhythm_test, note_test, chroma_test, dynamic_test, chord_test, arousal_test, valence_test, label_test

        if mode == "train":
            indexed = train_data
        elif mode == "val":
            indexed = test_data
        elif mode == "ref":
            indexed = data, rhythm, note, chroma, dynamic, chord, arousal, valence, label
        
        self.data, self.rhythm, self.note, self.chroma, self.dynamic, self.chord, self.arousal, self.valence, self.label = indexed
        self.data = [torch.Tensor(np.insert(k, -1, 1)) for k in self.data]
        self.data = torch.nn.utils.rnn.pad_sequence(self.data, batch_first=True)

        # 更准确的密度：基于 dynamic 的第5列（pitch_density：每帧音符数量）
        r_density_list = []
        n_density_list = []
        for dyn_seq in self.dynamic:
            try:
                arr = np.array(dyn_seq)
                if arr.ndim == 2 and arr.shape[1] >= 5 and len(arr) > 0:
                    pitch_den = arr[:, 4]
                    r_den = float(np.count_nonzero(pitch_den) / len(pitch_den))
                    # 将复音密度按上限15做裁剪并归一化到[0,1]
                    n_den = float(np.clip(pitch_den, 0, 15).mean() / 15.0)
                else:
                    # 回退：使用标签非零比例
                    r_den = float(np.count_nonzero(dyn_seq) / len(dyn_seq)) if len(dyn_seq) > 0 else 0.0
                    n_den = r_den
            except Exception:
                r_den = 0.0
                n_den = 0.0
            r_density_list.append(r_den)
            n_density_list.append(n_den)
        self.r_density = r_density_list
        self.n_density = np.array(n_density_list)

        self.rhythm = [torch.Tensor(k) for k in self.rhythm]
        self.note = [torch.Tensor(k) for k in self.note]
        
        # Process Chroma: Reshape flattened list to (-1, 24) and take mean to get global vector
        self.chroma = [torch.Tensor(k).reshape(-1, 24).mean(dim=0) for k in self.chroma]
        self.chroma = torch.stack(self.chroma) # Stack into (N, 24) tensor

        # Process Dynamic and Chord: Pad sequences
        self.dynamic = [torch.Tensor(k) for k in self.dynamic]
        self.chord = [torch.Tensor(k) for k in self.chord]

        self.rhythm = torch.nn.utils.rnn.pad_sequence(self.rhythm, batch_first=True)
        self.note = torch.nn.utils.rnn.pad_sequence(self.note, batch_first=True)
        self.dynamic = torch.nn.utils.rnn.pad_sequence(self.dynamic, batch_first=True)
        self.chord = torch.nn.utils.rnn.pad_sequence(self.chord, batch_first=True)
        # Chroma is already a tensor of (N, 24), no padding needed

        target_length = 448
        r_padding_length = target_length - self.rhythm.size(1)
        n_padding_length = target_length - self.note.size(1)
        d_padding_length = target_length - self.dynamic.size(1)
        ch_padding_length = target_length - self.chord.size(1)
        
        self.rhythm = F.pad(self.rhythm, (0, r_padding_length))
        self.note = F.pad(self.note, (0, n_padding_length))
        self.dynamic = F.pad(self.dynamic, (0, 0, 0, d_padding_length)) # Pad last dim (time)
        self.chord = F.pad(self.chord, (0, 0, 0, ch_padding_length)) # Pad last dim (time)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        r = self.rhythm[idx]
        n = self.note[idx]
        c = self.chroma[idx]
        d = self.dynamic[idx]
        ch = self.chord[idx]
        a = self.arousal[idx]
        v = self.valence[idx]
        l = self.label[idx]
        
        r_density = self.r_density[idx]
        n_density = self.n_density[idx]
        
        return x, r, n, c, d, ch, a, v, l, r_density, n_density
    
if __name__ == '__main__':
    get_vgmidi()
    