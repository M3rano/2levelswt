# 2levelswt
Algorithm for denoising a 1D signal while preserving weak signatures. It utilizes a two level multi-scale stationary wavelet transform.
Signatures with a signal-to-noise ratio of 5 or higher are marked. The best suited wavelet is chosen from the pywt library of wavelets by searching for the lowest bFoM which is a blind figure of merit. bFoM allows to evaluate and compare the results of different wavelets without prior knowledge.

# Usage
If `signal` is your 1D input data and `desired_scale` is your chosen decomposition scale, the best reconstruction,  its respective wavelet and the lowest bFoM value is given by:

    result = find_best_wavelet(signal, desired_scale)
    best_wavelet, best_bfom, best_rec = result

`best_wavelet` is the respective wavelet.
`best_bfom` is the bFoM value.
`best_rec` is the reconstructed and denoised signal.

Signatures are found by:

    lines = find_lines(best_rec)

`lines` is a list with indices of all found signatures.

# Dependencies
- pywt
- numpy 
- scipy

