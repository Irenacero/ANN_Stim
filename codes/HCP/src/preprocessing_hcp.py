import numpy as np
from scipy.signal import butter, filtfilt, detrend
import numpy as np
from scipy import stats

TR = 0.72  # seconds
LOW_HZ, HIGH_HZ = 0.008, 0.08
ORDER = 2

def bandpass_filter_timeseries(X, tr=TR, low=LOW_HZ, high=HIGH_HZ, order=ORDER):
    """
    X: (T, R) timepoints × regions
    Returns: filtered array (T, R)
    """
    fs = 1.0 / tr
    nyq = fs / 2.0
    low_n = low / nyq
    high_n = high / nyq
    if not (0 < low_n < high_n < 1):
        raise ValueError(f"Bad band ({low}, {high}) for TR={tr}s (Nyquist={nyq} Hz).")

    b, a = butter(order, [low_n, high_n], btype='bandpass')

    # Detrend to reduce edge/low-frequency artifacts (common in fMRI)
    Xd = detrend(X, axis=0, type='linear')

    # filtfilt pad length safeguard for short T
    padlen_default = 3 * max(len(a), len(b))
    padlen = min(padlen_default, max(0, Xd.shape[0] - 1))

    # Zero-phase filter along time axis
    Xf = filtfilt(b, a, Xd, axis=0, padlen=padlen)
    return stats.zscore(Xf,axis=0)

def preprocess_groups(groups, tr=TR, low=LOW_HZ, high=HIGH_HZ, order=ORDER):
    """Apply band-pass to a list of subjects, each (T, R)."""
    return [bandpass_filter_timeseries(subj, tr, low, high, order) for subj in groups]

# ---- existing unpacking code ----
import numpy as np

def unpack_group(group_array):
    """
    Convert MATLAB cell-style (1, N) arrays into a list of numpy arrays,
    each (T, R) = timepoints × regions for one subject.
    """
    n_subjects = group_array.shape[1]
    subjects = []
    for i in range(n_subjects):
        subj_data = group_array[0, i]   # pick subject i
        subjects.append(np.array(subj_data).T)
    return subjects
