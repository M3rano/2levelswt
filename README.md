# 2levelswt
Algorithm for denoising a 1D signal while preserving weak signatures. It utilizes a two level multi-scale stationary wavelet transform.
Signatures with a signal-to-noise ratio of 5 or higher are marked. The best suited wavelet is chosen from the pywt library of wavelets by searching for the lowest bFoM which is a blind figure of merit. bFoM allows to evaluate and compare the results of different wavelets without prior knowledge.

# Usage
If `signal` is your 1D input data and `desired_scale` is your chosen decomposition scale, the best reconstruction e.g. the lowest bFoM and its respective wavelet is given by:

    res_bfom = find_best_wavelet(signal, desired_scale)
    best_wavelet_bfom, best_bfom, best_rec_bfom = res_bfom

`best_wavelet_bfom` is the respective wavelet.
`best_bfom` is the bfom value.
`best_rec_bfom` is the reconstructed and denoised signal.

# Dependencies
- pywt
- numpy 
- scipy

