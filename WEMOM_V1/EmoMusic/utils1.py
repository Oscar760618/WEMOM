'''
code for processing the midi data, convert them into lists saved as numpy arrays
'''
import os
import torch
import numpy as np
from tqdm import tqdm
import pretty_midi
import pypianoroll
import music21
from music21 import converter, chord
import note_seq
from MidiPerformanceEncoder import MidiPerformanceEncoder
from utils2 import encode_midi, safe_array

midi_data_path = 'D:/PolyU/URIS/URIS/WEMOM_V1/Data/Mus_dataset'
save_path = 'D:/PolyU/URIS/URIS/WEMOM_V1/Data/Saves/'

PR_TIME_STEPS = 64
NUM_VELOCITY_BINS = 64
STEPS_PER_SECOND = 100
MIN_PITCH = 21
MAX_PITCH = 108

del_lst = []

def magenta_encode_midi(midi_filename, is_eos=False):
    mpe = MidiPerformanceEncoder(steps_per_second=STEPS_PER_SECOND, num_velocity_bins=NUM_VELOCITY_BINS, min_pitch=MIN_PITCH, max_pitch=MAX_PITCH, add_eos=is_eos)
    ns = note_seq.midi_file_to_sequence_proto(midi_filename)
    return mpe.encode_note_sequence(ns)

def magenta_decode_midi(notes, is_eos=False):
    mpe = MidiPerformanceEncoder(steps_per_second=STEPS_PER_SECOND, num_velocity_bins=NUM_VELOCITY_BINS, min_pitch=MIN_PITCH, max_pitch=MAX_PITCH, add_eos=is_eos)
    midi_file_path = mpe.decode(notes, strip_extraneous=False)
    return midi_file_path

# Get temp.midi file
def slice_midi(pm, beats, start_idx, end_idx):
    '''
    Slice given pretty_midi object into number of beat segments.
    '''

    new_pm = pretty_midi.PrettyMIDI()
    new_inst = pretty_midi.Instrument(program=pm.instruments[0].program, is_drum=pm.instruments[0].is_drum, name=pm.instruments[0].name)
    start, end = beats[start_idx], beats[end_idx]

    for i in range(len(pm.instruments)):

        for note in pm.instruments[i].notes:
            velocity, pitch = note.velocity, note.pitch
            if note.start > end or note.start < start:
                continue
            else:
                s = note.start - start
                if note.end > end:
                    e = end - start
                else:
                    e = note.end - start
            new_note = pretty_midi.Note(
                velocity=velocity, pitch=pitch, start=s, end=e)
            new_inst.notes.append(new_note)

        for ctrl in pm.instruments[i].control_changes:
            if ctrl.time >= start and ctrl.time < end:
                new_ctrl = pretty_midi.ControlChange(
                    number=ctrl.number, value=ctrl.value, time=ctrl.time - start)
                new_inst.control_changes.append(new_ctrl)

    new_pm.instruments.append(new_inst)
    new_pm.write('tmp.mid')
    return new_pm

# Get Chroma Attributes
def get_harmony_vector(fname, is_one_hot=False):
    CHORD_DICT = {
    "C-": 11, "C": 0, "C#": 1, 
    "D-": 1, "D": 2, "D#": 3, 
    "E-": 3, "E": 4, "E#": 5,
    "F-": 4, "F": 5, "F#": 6, 
    "G-": 6, "G": 7, "G#": 8, 
    "A-": 8, "A": 9, "A#": 10, 
    "B-": 10, "B": 11, "B#": 0}

    try:
        score = music21.converter.parse(fname)
        key = score.analyze('key')
        res = np.zeros(24,)
        name, mode = key.tonic.name, key.mode
        idx = CHORD_DICT[name] + 12 if mode == "minor" else CHORD_DICT[name]

        if not is_one_hot: 
            res[idx] = key.correlationCoefficient
            for i, x in enumerate(key.alternateInterpretations):
                name, mode = x.tonic.name, x.mode
                idx = CHORD_DICT[name] + 12 if mode == "minor" else CHORD_DICT[name]
                res[idx] = x.correlationCoefficient

            res[res < 0.1] = 0
        else:
            if idx:
                res[idx] = 1

        return res

    except Exception as e:
        print(e, "harmony vector")
        return None

# Get Chord Attributes   
def get_chord_progression(midi_file, num_chords=24):
    score = converter.parse(midi_file)
    chords = score.chordify()
    chord_sequence = []

    # Map 12 major + 12 minor keys to 24 classes consistently
    CHORD_DICT = {
        "C major": 0,  "C minor": 12,
        "C# major": 1, "C# minor": 13,
        "D- major": 1, "D- minor": 13,  # enharmonic for C#
        "D major": 2,  "D minor": 14,
        "D# major": 3, "D# minor": 15,
        "E- major": 3, "E- minor": 15,  # enharmonic for D#
        "E major": 4,  "E minor": 16,
        "F major": 5,  "F minor": 17,
        "F# major": 6, "F# minor": 18,
        "G- major": 6, "G- minor": 18,  # enharmonic for F#
        "G major": 7,  "G minor": 19,
        "G# major": 8, "G# minor": 20,
        "A- major": 8, "A- minor": 20,  # enharmonic for G#
        "A major": 9,  "A minor": 21,
        "A# major": 10, "A# minor": 22,
        "B- major": 10, "B- minor": 22,  # enharmonic for A#
        "B major": 11,  "B minor": 23
    }

    for element in chords.flatten().notes:
        if isinstance(element, chord.Chord):
            chord_name = element.commonName
            # Try direct match; fallback to root + quality
            idx = None
            if chord_name in CHORD_DICT:
                idx = CHORD_DICT[chord_name]
            else:
                try:
                    rn = element.root().name
                    qual = "minor" if element.quality == "minor" else "major"
                    key_name = f"{rn} {qual}"
                    if key_name in CHORD_DICT:
                        idx = CHORD_DICT[key_name]
                except Exception:
                    idx = None
            chord_sequence.append(idx if idx is not None else -1)

    chord_sequence_one_hot = np.zeros((len(chord_sequence), num_chords), dtype=int)
    for i, chord_idx in enumerate(chord_sequence):
        if chord_idx >= 0:
            chord_sequence_one_hot[i, chord_idx] = 1

    return chord_sequence_one_hot

def get_chord_progression_aligned(midi_file, target_len, num_chords=24):
    """
    获取和弦序列并对齐到拍网格长度（如 beat=24 的帧数）。
    将原始和弦序列通过等距采样/重复映射到 target_len，返回 (target_len, num_chords) 的 one-hot。
    """
    chord_seq = get_chord_progression(midi_file, num_chords=num_chords)
    if chord_seq is None or len(chord_seq) == 0:
        return np.zeros((target_len, num_chords), dtype=int)
    src_len = chord_seq.shape[0]
    if src_len == target_len:
        return chord_seq
    # 等距索引映射到目标长度
    idx = np.linspace(0, max(0, src_len - 1), num=target_len)
    idx = np.round(idx).astype(int)
    return chord_seq[idx]

def process_data(name, beat_res, num_of_beats, max_tokens):
    data_lst = []
    rhythm_lst = []
    note_lst = []
    dynamic_lst = []
    chroma_lst = []
    chord_lst = []
    
    track = pypianoroll.read(name).tracks

    if len(track) > 0:
        try:
            pm = pretty_midi.PrettyMIDI(name)
            beats = pm.get_beats()
        except Exception as e:
            print(e)
        
        # Select the first track from the MIDI file
        pr = track[0].pianoroll
        
        # Each segment should be beat_res * num_of_beats long
        for j in range(0, len(pr), beat_res * num_of_beats):
            start_idx = j
            end_idx = j + beat_res * num_of_beats

            if end_idx // beat_res >= len(beats):
                end_idx = (len(beats) - 1) * beat_res
                if start_idx >= end_idx:
                    break
            
            new_pr = pr[start_idx : end_idx]
            new_pm = slice_midi(pm, beats, start_idx // beat_res, end_idx // beat_res)
            new_pm.write("tmp.mid")

            if len(new_pm.instruments[0].notes) > 0:

                rhythm, note, dynamic = encode_midi(new_pr, beat=24, is_pr=True)
                chroma = get_harmony_vector("tmp.mid", is_one_hot=False)
                chord = get_chord_progression_aligned("tmp.mid", target_len=len(rhythm), num_chords=24)
                events = magenta_encode_midi("tmp.mid")
                events.append(1)

                if len(events) <= max_tokens:       
                    data_lst.extend(events)
                    rhythm_lst.extend(rhythm)
                    note_lst.extend(note)
                    chroma_lst.extend(chroma)
                    dynamic_lst.extend(dynamic)
                    chord_lst.extend(chord)

    return torch.Tensor(data_lst), rhythm_lst, note_lst, chroma_lst, dynamic_lst, chord_lst
         
def get_classic_piano(midi_data_path, data_type="long"):
    '''
    Saving the preprocessed midi data into numpy arrays.
    '''

    files = os.listdir(midi_data_path)
    # Filter for MIDI files only to exclude CSV or other files
    files = [f for f in files if f.lower().endswith(('.mid', '.midi'))]
    files.sort()
    labelled_midi = [os.path.join(midi_data_path, k) for k in files]
    print("Dataset length:", len(labelled_midi))
    keylst = labelled_midi

    data_lst = []
    rhythm_lst = []
    note_lst = []
    chroma_lst = []
    dynamic_lst = []
    chord_lst = []

    for idx, name in tqdm(enumerate(keylst), total=len(keylst)):

        try: 
            print(f"processing {idx, name}")
            if data_type == "short":
                beat_res, num_of_beats, max_tokens = 4, 4, 100
            elif data_type == "long":
                beat_res, num_of_beats, max_tokens = 4, 16, 1000

            cur_data_lst, cur_rhythm_lst, cur_note_lst, cur_chroma_lst, cur_dynamic_lst, cur_chord_lst = process_data(name, beat_res=beat_res, num_of_beats=num_of_beats, max_tokens=max_tokens)

            # Some midi files cann't be processed, record in the del lst
            # outputs = [cur_data_lst, cur_rhythm_lst, cur_note_lst, cur_chroma_lst, cur_dynamic_lst, cur_chord_lst]
            print(len(cur_data_lst), len(cur_rhythm_lst), len(cur_note_lst), len(cur_chroma_lst), len(cur_dynamic_lst), len(cur_chord_lst))
            
            data_lst.append(cur_data_lst)
            rhythm_lst.append(cur_rhythm_lst)
            note_lst.append(cur_note_lst)
            chroma_lst.append(cur_chroma_lst)
            dynamic_lst.append(cur_dynamic_lst)
            chord_lst.append(cur_chord_lst)

            if os.path.exists("tmp.mid"):
                os.remove("tmp.mid")

        except Exception as e:
            err_msg = str(e)
            del_lst.append((idx, name, err_msg))
            continue

    print(len(data_lst), len(rhythm_lst), len(note_lst), len(chroma_lst), len(dynamic_lst), len(chord_lst))
    
    data_arr = torch.nn.utils.rnn.pad_sequence(data_lst, batch_first=True).numpy().astype(int)

    rhythm_arr = safe_array(rhythm_lst)
    note_arr = safe_array(note_lst)
    chroma_arr = safe_array(chroma_lst)
    dynamic_arr = safe_array(dynamic_lst)
    chord_arr = safe_array(chord_lst)

    np.save(save_path + "data.npy", data_arr)
    np.save(save_path + "rhythm.npy", rhythm_arr)
    np.save(save_path + "note.npy", note_arr)
    np.save(save_path + "chroma.npy", chroma_arr)
    np.save(save_path + "dynamic.npy", dynamic_arr)
    np.save(save_path + "chord.npy", chord_arr)

    return data_arr, rhythm_arr, note_arr, chroma_arr, dynamic_arr, chord_arr

if __name__ == '__main__':
    results = get_classic_piano(midi_data_path, data_type="long")
    print("Returned array shapes:")
    for i, name in enumerate(["data_arr", "rhythm_arr", "note_arr", "chroma_arr", "dynamic_arr", "chord_arr"]):
        print(f"{name}: {results[i].shape}")
    
    print("Error list:")
    for err in del_lst:
        print(err[0])