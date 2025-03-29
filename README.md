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

### Notes:

Lines from before and after a transform can not be directly compared i.e. when looking for false positives.
Sometimes the transform will shift peak indices slightly. Therefore a range of indices needs to be set as tolerance. This can be done using the following code:

    def has_close_number(num, other_list, distance):
        """
        Check if index of signature is close to other indices.

        Parameters
        ----------
        num : float
            Index in x of signature 
        other_list : list
            Indices of signatures before transform
        distance : int
            Allowed distance between indices of a signature.

        Returns
        -------
        boolean
            True if there is a close index. False if not.

        Notes
        ----
        Index of a signature can shift through the transformation.
        Using this function we make sure that signatures from after transformation are paired correctly with the signatures from before. 

        """
        return any(abs(num - x) <= distance for x in other_list)
        
        
    tolerance = 50
    
    false_positives = [
    x for x in lines_after
    if (x not in lines_before) and not has_close_number(x, lines_before, tolerance)
    ]

`lines_before` denotes the lines found before transforming.\
`lines_after` denotes the lines found after transforming.\
An appropriate `tolerance` depends heavily on the size of the data. \
`false_positives` is a list containing all the indices from `lines_after` that do not appear in `lines_before` with regard for the set `tolerance`.

    

## Basic test
Use the following function to create a simple emission spectrum:

    def emission_spectrum(wavelength, lines):
        """
        Create artificial 1D emission spectrum

        Parameters
        ----------
        wavelength : array_like
            Data points for x-axis
        lines : array_like
            List of tuples.
            Each tuple contains location, width and amplitude of a line

        Returns
        -------
        spectrum : array_like
            1D spectrum

        """
        spectrum = np.zeros_like(wavelength)
        for line in lines:
            center, amplitude, width = line
            spectrum += amplitude * \
                np.exp(-0.5 * ((wavelength - center) / width) ** 2)
        return spectrum


    wavelength_min = 0  # minimal wavelength
    wavelength_max = 1000  # maximum wavelength
    num_points = 25000  # number of points

    # parameters for spectral lines (center of wavelength, amplitude, width)
    lines = [
        (300, 1.0, 2),
        (400, 0.5, 1),
        (800, 0.8, 4),
        (150, 0.7, 0.5),
        (550, 0.6, 2),
        (557, 0.35, 2),
        (700, 0.1, 0.5),
        (50, 0.1, 1),
        (250, 0.3, 0.5)
        # add more tuples for more lines
    ]

    # create wavelength range
    wavelengths = np.linspace(wavelength_min, wavelength_max, num_points)

    # calculate emission spectrum
    spectrum = emission_spectrum(wavelengths, lines)

Add noise to the spectrum and define signal with:

    # add noise
    spectrum_with_noise = add_noise(spectrum)

    # define x and signal
    x = wavelengths
    og_signal = spectrum
    signal = spectrum_with_noise


Since we have prior knowledge of the underlying signal, mean squared error is used instead of bFoM.
For that, see docstring of the function find_best_wavelet().




# Dependencies
- pywt
- numpy 
- scipy

# Citation
If you use this code, we kindly request that you acknowledge our work. Please include at least the following citation:

(1) Eqbal, Mehran. Application of the Wavelet Transform Technique to the Analysis of Weak Signatures in the Titan Spectra Obtained with Herschel/PACS. 2025, University of Göttingen, Bachelor's thesis.

(2) Mehran Eqbal & Miriam Rengel.
  
