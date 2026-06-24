import pypianoroll
import numpy as np
from torch import NoneType

# 0-127 note on, 128 start token, 129 end token, 130 shift, 131-258 note off
# 259 empty, 260 - 387 velocity token

OFFSET_DISPLACEMENT = 131    
VELOCITY_DISPLACEMENT = 260

def convert_pr(pr):
    pitch_lst = []
    velocity_lst = []

    for timestep in pr:
        pitches = []
        velocities = []
        for pitch, velocity in enumerate(timestep):
            if velocity > 0:
                pitches.append(pitch)
                velocities.append(velocity)
        pitch_lst.append(pitches)
        velocity_lst.append(velocities)

    return pitch_lst, velocity_lst

def rhythm_dynamic(pr):
    pitch_lst, velocity_lst = convert_pr(pr)

    rhythm_lst = []
    dynamic_lst = []

    if len(pitch_lst[0]) > 0:
        rhythm_lst.append(1)
    else:
        rhythm_lst.append(0)
    prev = pitch_lst[0]

    for i in range(1, len(pitch_lst)):
        if len(pitch_lst[i]) == 0:
            rhythm_lst.append(0)
        elif pitch_lst[i] == prev or all(elem in prev for elem in pitch_lst[i]):
            rhythm_lst.append(2)
        else:
            rhythm_lst.append(1)
        prev = pitch_lst[i]

    for v in velocity_lst:
        if v and len(v) > 0:
            dynamic_lst.append(int(sum(v) / len(v)))
        else:
            dynamic_lst.append(0)

    return list(zip(rhythm_lst, dynamic_lst))

def note_dynamic(pr):
    pitch_lst, velocity_lst = convert_pr(pr)
    note_density = [len(k) for k in pitch_lst]
    dynamic_lst = []

    for v in velocity_lst:
        if v and len(v) > 0:
            dynamic_lst.append(int(sum(v) / len(v)))
        else:
            dynamic_lst.append(0)

    return list(zip(note_density, dynamic_lst))

def bucketize_polyphony(densities, num_buckets=16, method="quantile"):
    """
    将复音数映射到离散桶（默认16类），支持分位数或阈值法。
    densities: List[int] 每步的复音数量。
    返回：List[int] 桶索引（0..num_buckets-1）。
    """
    arr = np.asarray(densities).astype(float)
    if arr.size == 0:
        return []
    if method == "quantile":
        # 根据分位数构建桶边界，避免类别极不均衡
        qs = np.linspace(0, 1, num_buckets + 1)
        edges = np.quantile(arr, qs)
        # 保证边界单调递增
        edges = np.asarray(edges)
        # 使用digitize得到桶索引
        buckets = np.clip(np.digitize(arr, edges[1:-1], right=False), 0, num_buckets - 1)
    else:
        # 固定阈值：例如每2个为一档（可按需调整）
        step = max(1, int(np.ceil((arr.max() + 1) / num_buckets)))
        edges = np.arange(0, step * num_buckets + 1, step)
        buckets = np.clip(np.digitize(arr, edges[1:-1], right=False), 0, num_buckets - 1)
    return buckets.tolist()

def get_dynamic(pr, num_bins=32):
    pitch_lst, velocity_lst = convert_pr(pr)
    bin_edges = np.linspace(0, 128, num_bins + 1)
    max_velocity = []
    var_velocity = []
    velocity_bins = []
    pitch_density = []

    mean_velocity_lst = [np.mean(v) if v else 0 for v in velocity_lst]
    gradient_velocity = np.gradient(mean_velocity_lst)

    for p, v in zip(pitch_lst, velocity_lst):
        if v and len(v) > 0:
            max_v = max(v)
            var_v = np.var(v)
            mean_v = np.mean(v)
            bin_v = int(np.digitize(mean_v, bin_edges)) - 1
        else:
            max_v = 0
            var_v = 0
            bin_v = 0

        max_velocity.append(max_v)
        var_velocity.append(var_v)
        velocity_bins.append(bin_v)
        pitch_density.append(len(p))

    dynamics_features = np.stack(
        [max_velocity, var_velocity, gradient_velocity, velocity_bins, pitch_density], axis=-1
    )

    return dynamics_features

def encode_midi(fname, beat=24, is_pr=False):

    if not is_pr:
        track = pypianoroll.parse(fname, beat_resolution=beat)
        pr = track.get_merged_pianoroll()[:beat*8]
    else:
        pr = fname

    rhythm_pairs = rhythm_dynamic(pr)  # [(rhythm_label, dyn_mean), ...]
    note_pairs = note_dynamic(pr)      # [(note_density, dyn_mean), ...]
    dynamic = get_dynamic(pr)          
    rhythm = [int(r) for (r, _dyn) in rhythm_pairs]
    # 复音分桶离散化到16类，替代简单截断
    densities = [int(nd) for (nd, _dyn) in note_pairs]
    note = bucketize_polyphony(densities, num_buckets=16, method="quantile")

    # sanity-check: 打印桶分布（仅限长度足够时）
    if len(note) > 0:
        counts = np.bincount(np.asarray(note), minlength=16)
        print("[polyphony buckets]", counts.tolist())

    return rhythm, note, dynamic

def shift():
    return 130

def note_on(pitch):
    return pitch

def note_off(pitch):
    return pitch + OFFSET_DISPLACEMENT      

def vel(velocity):
    return int(velocity) + VELOCITY_DISPLACEMENT

def safe_array(lst):
    try:
        return np.array(lst)
    except Exception:
        return np.array(lst, dtype=object)

