# import necessary libraries


import pywt
import numpy as np
from scipy.signal import find_peaks
import warnings


import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
import matplotlib.font_manager
import matplotlib.colors


def pad_signal(signal):
    """
    Since pywt.swt() needs the input data to be a multiple of 2^scale,
    we pad the signal to be able to decompose until the desired scale.  

    Parameters
    ----------
    signal : array_like
        1D input signal

    Returns
    -------
    signal : array_like
        1D padded signal
    pad_length : int
        Different in length between original signal and padded signal.

    """
    candidate = len(signal)
    with warnings.catch_warnings():  # suppress userwarning for candidates that are not divisible by 2
        warnings.simplefilter("ignore", UserWarning)
        while pywt.swt_max_level(candidate) < max_scale:
            candidate += 1
    pad_length = candidate - len(signal)

    if pad_length == 0:
        return signal, pad_length
    elif pad_length % 2 == 0:
        signal = pywt.pad(signal, (int(pad_length/2), int(pad_length/2)),
                          mode=pad_mode)

        return signal, pad_length
    else:
        signal = pywt.pad(signal, (int(np.floor(pad_length/2)),
                          int(np.ceil(pad_length/2))), mode=pad_mode)

        return signal, pad_length


def revert_signal_length(signal, pad_length):
    """
    Revert signal to original length.

    Parameters
    ----------
    signal : array_like
        1D signal 
    pad_length : int
        Different in length between original signal and padded signal. 

    Returns
    -------
    signal : array_like
        Signal with length of original input signal.

    """
    if pad_length == 0:
        return signal
    elif pad_length % 2 == 0:
        signal = signal[int(pad_length/2):-int(pad_length/2)]
        return signal
    else:
        signal = signal[int(np.floor(pad_length/2)):-
                        int(np.ceil(pad_length/2))]
        return signal


def multilevel_swt(signal, wavelet, desired_scale):
    """
    Perform a two level stationary wavelet transform.
    Two level means that detail coefficients from all scales get decomposed again,
    as if they were a signal.

    Parameters
    ----------
    signal : array_like
        1D input signal
    wavelet : str
        Name of the wavelet to be used for decomposition.
    desired_scale : int
        Decompose signal into that many scales.

    Returns
    -------
    coeffs : list
        List containing coefficient arrays for all scales. 
    dec_tree : dict of str:list
        Dictionary for storing the lists of swt for every scale of detail coefficients.

    Notes
    ----
    trim_approx changes the output format.
    """
    dec_tree = {}

    if wavelet.orthogonal == True:
        norm = True  # If norm is True then the energy of the coefficients and the signal is equal
    else:
        norm = False

    coeffs = pywt.swt(signal, wavelet, level=desired_scale,
                      trim_approx=True, norm=norm)

    for i, k in zip(nodes, range(desired_scale)):
        dec_tree[i] = pywt.swt(
            coeffs[k+1], wavelet, level=desired_scale, trim_approx=True, norm=norm)

    return coeffs, dec_tree


def multilevel_iswt(coeffs, alt_tree, wavelet, desired_scale):
    """
    Perform a two level inverse stationary wavelet transform.

    Parameters
    ----------
    coeffs : list
       List containing coefficient arrays for all scales.
    alt_tree : dict
        Dictionary for storing the lists of swt for every scale of detail coefficients.
        Coefficients are already truncated.
    wavelet : str
        Name of the wavelet that was used for decomposition.
    desired_scale : int
        Signal is decomposed into that many scales. 

    Returns
    -------
    rec_signal :array_like
        1D reconstructed signal

    """

    if wavelet.orthogonal == True:
        norm = True  # If norm is True then the energy of the coefficients and the signal is equal
    else:
        norm = False

    for i, k in zip(nodes, range(desired_scale)):
        coeffs[k+1] = pywt.iswt(alt_tree[i],
                                wavelet, norm=norm)

    rec_signal = pywt.iswt(coeffs, wavelet, norm=norm)

    return rec_signal


def calc_rms(data, num_chunks, use_chunks):
    """
    Calculate RMS.

    Parameters
    ----------
    data : array_like
        1D data to calculate rms for.
    num_chunks : int
        number of blocks to seperate data into for a more accurate rms calculation.
    use_chunks : TYPE
        Use this many blocks for the rms calculation.

    Returns
    -------
    rms : float
        RMS of the data.

    Notes
    ----
    Data can be split into chunks before calculation to possibly get a more accurate result.
    num_chunks must be larger or equal to use_chunks.

    """
    if num_chunks > use_chunks:
        chunks = np.array_split(data, num_chunks)
        sorted_by_rms = sorted([np.std(x, ddof=1) for x in chunks])
        rms = np.mean(sorted_by_rms[:use_chunks])
    elif num_chunks == use_chunks:
        chunks = np.array_split(data, num_chunks)
        sorted_by_rms = sorted([np.std(x, ddof=1) for x in chunks])
        rms = np.mean(sorted_by_rms)
    return rms


def calc_baseline(data, num_chunks, use_chunks):
    """
    data : array_like
        1D data to calculate baseline for.
    num_chunks : int
        number of blocks to seperate data into for a more accurate baseline calculation.
    use_chunks : TYPE
        Use this many blocks for the baseline calculation.

    Returns
    -------
    baseline : float
        Baseline of the data

    Notes
    ----
    Data can be split into chunks before calculation to possibly get a more accurate result.

    """
    chunks = np.array_split(data, num_chunks)
    sorted_by_mean = sorted([np.mean(x) for x in chunks])
    baseline = np.mean(sorted_by_mean[:use_chunks])
    return baseline


# Function to add noise to signal
def add_noise(spectrum):
    """
    Adds noise to a signal

    Parameters
    ----------
    spectrum : array_like
        Artificial Spectrum
    wavelengths : array_like
        Data Points for x-axis
    wavelength_min : int
        Minimal wavelength
    wavelength_max : int
        Maximal wavelength

    Returns
    -------
    array_like
        Artificial spectrum with added noise

    """
    rng = np.random.default_rng(seed=noise_seed)
    noise = rng.normal(loc=noise_mean, scale=noise_level, size=len(spectrum))
    return spectrum + noise


def noise_estimation(signal, wavelet, desired_scale):
    """
    Estimate the noise of the detail coefficients by creating and decomposing a signal of pure noise.
    RMS of the respective scales is the noise estimate.

    Parameters
    ----------
    signal : array_like
        1D input signal
    wavelet : str
        Name of the wavelet to be used for decomposition.
    desired_scale : int
        Decompose signal into that many scales.

    Returns
    -------
    noise_tree : dict of str:list
        Dictionary for storing the noise estimation for every scale of detail coefficients.

    """

    sigma = np.std(signal)

    # create a signal of pure gaussian noise
    rng = np.random.default_rng(seed=noise_seed)
    purenoise = rng.normal(loc=0, scale=sigma, size=len(signal))

    noise_coeffs, noise_tree = multilevel_swt(
        purenoise, wavelet, desired_scale)
    for i in nodes:
        for k in range(desired_scale):
            noise_tree[i][k +
                          1] = calc_rms(noise_tree[i][k+1], num_chunks, use_chunks)

    return noise_tree


def alt_data(dec_tree, noise_tree, desired_scale):
    """
    Truncate the second level detail coefficients using the noise estimates.

    Parameters
    ----------
    dec_tree : dict of str:list
        Dictionary for storing the lists of swt for every scale of detail coefficients.
    noise_tree : dict of str: list
        Dictionary for storing the noise estimation for every scale of detail coefficients.
    desired_scale : int
        Signal is decomposed into that many scales.

    Returns
    -------
    dec_tree : dict of str:list
        Dictionary for storing the lists of swt for every scale of detail coefficients.
        Second level detail coefficients are truncated.

    """
    for i in nodes:
        for k in range(desired_scale):
            threshold = noise_tree[i][k+1]
            dec_tree[i][k+1] = pywt.threshold(dec_tree[i][k+1],
                                              confidence * threshold, mode=thr_mode, substitute=0)

    return dec_tree


def normalize(data, t_min, t_max):
    """
    Normalize data from t_min to t_max.


    Parameters
    ----------
    data : array_like
        1D data to normalize
    t_min : float
        Lower boundary.
    t_max : float 
        Upper boundary.

    Returns
    -------
    array_like
        Normalized data

    """
    norm_data = []
    diff = t_max - t_min
    diff_data = np.max(data) - np.min(data)
    eps = np.finfo(diff_data.dtype).eps
    diff_data += eps  # avoid division by zero
    for i in data:
        temp = (((i - np.min(data))*diff)/diff_data) + t_min
        norm_data.append(temp)
    return np.array(norm_data)


def calc_bfom(signal, rec_signal):
    """
    Calculate the blind figure of merit (bfom).

    Parameters
    ----------
    signal : array_like
        1D input signal.
    rec_signal : array_like
        1D reconstructed signal.

    Returns
    -------
    bfom_arr : array_like
        Array containing the three figures of merit used for bfom calculation.

    """
    roughness = np.sum(np.abs(np.convolve(
        [-1, 1], rec_signal, mode='valid')))/np.sum(np.abs(rec_signal))

    N_0 = np.mean(np.divide(np.abs(np.fft.fft(signal))**2, len(signal)**2))
    N_1 = np.mean(np.divide(np.abs(np.fft.fft(rec_signal))**2,
                  len(rec_signal)**2))
    NR = N_0/N_1
    inverse_NR = NR**(-1)

    rec_signal_mean = np.mean(rec_signal)

    RNU = (100/rec_signal_mean) * \
        np.sqrt(np.sum(np.square(rec_signal-rec_signal_mean))/len(signal))

    bfom_arr = [roughness, inverse_NR, RNU]

    return bfom_arr


def full_transform(signal, wavelet, desired_scale):
    """
    Perform transform, truncation and inverse transform.

    Parameters
    ----------
    signal : array_like
        1D input signal.
    wavelet : str
        Name of the wavelet to be used for decomposition.
    desired_scale : int
        Decompose signal into that many scales.

    Returns
    -------
    rec_signal : array_like
        1D reconstructed signal
    bfom_arr : array_like
        Array containing the three figures of merit used for bfom calculation.

    """
    signal, diff = pad_signal(signal)
    coeffs, dec_tree = multilevel_swt(
        signal, wavelet, desired_scale)
    noise_tree = noise_estimation(signal, wavelet, desired_scale)
    alt_tree = alt_data(dec_tree, noise_tree, desired_scale)
    rec_signal = multilevel_iswt(coeffs, alt_tree, wavelet, desired_scale)
    rec_signal = revert_signal_length(rec_signal, diff)
    bfom_arr = calc_bfom(signal, rec_signal)
    return rec_signal, bfom_arr


def find_best_mse(results, mse_dict):
    """
    Find reconstructed signal and respective wavelet with lowest mean squared error (mse).

    Parameters
    ----------
    results : dict of str:array_like
        Dictionary containing the reconstructed signal for all used wavelets.
    mse_dict : dict of str:float
        Dictionary containing the mse of the reconstructed signal for all used wavelets.

    Returns
    -------
    res_mse : array_like
        Tuple containing information regarding lowest mse.

    """
    min_mse_key = min(mse_dict, key=mse_dict.get)
    best_mse = mse_dict[min_mse_key]
    best_rec_mse = results[min_mse_key]
    res_mse = (min_mse_key, best_mse, best_rec_mse)
    return res_mse


def find_best_bfom(results, bfom_dict):
    """
    Calculate bfom and find reconstructed signal and respective wavelet with lowest bfom.

    Parameters
    ----------
    results : dict of str:array_like
        Dictionary containing the reconstructed signal for all used wavelets.
    bfom_dict : dict of str:array_like
        Dictionary containing the bfom_arr of the reconstructed signal for all used wavelets.

    Returns
    -------
    res_bfom : array_like
        Tuple containing information regarding lowest bfom.

    """

    roughness_all = np.array([value[0] for value in bfom_dict.values()])
    inverse_NR_all = np.array([value[1] for value in bfom_dict.values()])
    RNU_all = np.array([value[2] for value in bfom_dict.values()])
    range_to_normalize = (0, 1)

    roughness_norm = normalize(
        roughness_all, range_to_normalize[0], range_to_normalize[1])
    inverse_NR_norm = normalize(
        inverse_NR_all, range_to_normalize[0], range_to_normalize[1])
    RNU_norm = normalize(RNU_all, range_to_normalize[0], range_to_normalize[1])

    bfom = np.sqrt((roughness_norm**2+inverse_NR_norm**2+RNU_norm**2)/3)

    # sometimes at low scales bfom can be zero for some wavelets.
    bfom = [i for i in bfom if i != 0]

    min_bfom_index = np.argmin(bfom)
    min_bfom_key = list(bfom_dict.keys())[min_bfom_index]
    min_bfom = bfom[min_bfom_index]

    # min_bfom_key = min(bfom_dict, key=bfom_dict.get)
    best_rec_bfom = results[min_bfom_key]
    res_bfom = (min_bfom_key, min_bfom, best_rec_bfom)

    return res_bfom


def find_best_wavelet(signal, desired_scale):
    """
    Perform full transform for multiple wavelets and search for best result with chosen figure of merit.

    Parameters
    ----------
    signal : array_like
        1D input signal
    desired_scale : int
        Decompose signal into that many scales.

    Returns
    -------
    res_snr : array_like
        Tuple containing information regarding highest snr.
    res_mse : array_like
        Tuple containing information regarding lowest mse.
    res_bfom : array_like
        Tuple containing information regarding lowest bfom.

    Notes
    ----
    With prior knowledge of the signal e.g.
    when the pure signal without noise is known,
    use mse_dict instead of bfom_dict
    and find_best_mse instead of find_best_bfom.

    Using bfom will not work when wavelist has only one item
    because the normalization is not possible with 1 value.

    """
    bfom_dict = {}
    # mse_dict = {}
    results = {}

    wavelist = pywt.wavelist(kind='discrete')
    for i in wavelist:

        rec_signal, bfom_arr = full_transform(
            signal, pywt.Wavelet(i), desired_scale)
        results[i] = rec_signal

        bfom_dict[i] = bfom_arr
        # mse_dict[i] = sum(np.square(rec_signal-og_signal))/len(og_signal)

    res = find_best_bfom(results, bfom_dict)
    # res = find_best_mse(results, mse_dict)

    return res


def find_lines(signal):
    """
    Find the signatures of a signal.

    Parameters
    ----------
    signal : array-like
        1D signal.

    Returns
    -------
    lines : list
        List that contains the indices for all found signatures.

    """

    rms = calc_rms(signal, num_chunks, use_chunks)
    baseline = calc_baseline(signal, num_chunks, use_chunks)

    # Find the signatures with an snr over the set snr_limit
    lines, _ = find_peaks(
        signal, height=baseline+rms*snr_limit, width=detection_width)

    # Note that chosen width depend heavily on the data
    # and needs to be adjusted by hand.

    return lines


# Set parameters
desired_scale = 5  # Scale of decomposition
confidence = 3  # Level of confidence
thr_mode = "hard"  # Mode used for the threshold
noise_seed = 23  # Seed used for the random generator
max_scale = 10  # Desired maximum decomposition scale after padding
pad_mode = 'periodic'  # Mode of padding
num_chunks = 100  # Number of blocks to divide signal into
use_chunks = 80  # Use this amount of blocks for rms and baseline calculation
snr_limit = 5  # Detection limit
detection_width = None  # Minimum number of samples that a signature needs to be wide


# These parameters are only relevant if add_noise is used
noise_level = 1  # Generated noise level
noise_mean = 0  # Mean of the noise distribution

# Create list of tree nodes
nodes = []
for i in range(desired_scale):
    nodes.append('D' + str(desired_scale - i))


############################################################################


############################################################################

# To perform the algorithm, define a signal,
# set the parameters and add the following 3 lines of code

# result = find_best_wavelet(signal, desired_scale)
# best_wavelet, best_bfom, best_rec = result
# lines = find_lines(best_rec)


############################################################################
