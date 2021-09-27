"""
-------------------------------------------------------------------------------
Created: 11.02.2021, 12:40
-------------------------------------------------------------------------------
Author: Matthieu Scherpf
Email: Matthieu.Scherpf@tu-dresden.de
Website: https://becuriouss.github.io/matthieu-scherpf/
Project page: tba
-------------------------------------------------------------------------------
Purpose: A collection of the most relevant signal processing used for the extraction of parameters of the remotely acquired photoplethysmogram (rPPG; also called imaging photoplethysmogram - iPPG). No differentiation between heart rate and pulse rate allthough it must be distinguished within a medical/physiological context.
-------------------------------------------------------------------------------
"""
import numpy as np
from scipy import signal, interpolate
import warnings
from datetime import timedelta


def compute_window_based_feature(seq,
                                 sample_freq,
                                 func_handle,
                                 window_length,
                                 window_stride,
                                 verbose=False,
                                 **kwargs):
    """Use this function to compute any metric within a sliding window.

    Parameters
    ----------
    seq : 1D array like object
        e.g. a blood volume pulse sequence, a continuous blood pressure sequence, a heart rate sequence etc.
    sample_freq : float
        the sampling frequency of the sequence; if the signal is not sampled at a constant sampling frequency than resample it beforehand
    func_handle : function handle
        the handle of the function to apply on each window
    window_length : float
        the length of each window in seconds
    window_stride : float
        the stride between two consecutive windows in seconds
    verbose : bool, optional
        if the intermediate results of each window should be collected and returned as well, by default False

    Returns
    -------
    numpy.ndarray of shape [n,] where n corresponds to the number of windows built
        the extracted metric value for each window
    dict containing {'w_data', 'w_masks'} the intermediate results of each window and the corresponding boolean mask used to extract the window; is only returned, when verbose is True

    Raises
    ------
    TypeError
        if a 'ref_hr_bpm' key is set in kwargs to compute the snr metric but the corresponding value is not a list
    """
    # create timestamps for the sequence
    seq = np.squeeze(seq)
    seq_ts = np.arange(0, len(seq)) * 1/sample_freq
    res = []
    ver = {'w_data': [], 'w_masks': []}
    # set loop indexes
    ts = 0
    i = 0
    # check kwargs
    ref_hr_bpm = kwargs.pop('ref_hr_bpm', None)
    if ref_hr_bpm is not None:
        ref_hr_bpm = np.squeeze(ref_hr_bpm)

    while ts + window_length <= seq_ts[-1]:
        mask = (seq_ts >= ts) & (seq_ts < ts + window_length)
        if ref_hr_bpm is not None:
            kwargs['ref_hr_bpm'] = ref_hr_bpm[i]
        out = func_handle(seq[mask], sample_freq, verbose=True, **kwargs)
        res.append(out[0])
        if verbose:
            ver['w_data'].append(out[1])
            ver['w_masks'].append(mask)
        ts += window_stride
        i += 1

    if verbose:
        return np.asarray(res), ver
    return np.asarray(res)


def compute_hr_from_hr_sequence(seq,
                                *args,
                                filter_method='median',
                                verbose=False):
    """Use this function if you want to extract the overall mean or median heart rate from a given heart rate sequence.

    Parameters
    ----------
    seq : 1D array like object
        the heart rate sequence that you want to process
    filter_method : {'mean', 'median'}, optional
        the method you want to use to filter; by default 'median'
    verbose : bool, optional
        this is for compatibility when used with windowed computation, by default False

    Returns
    -------
    float
        the heart rate resulting from filtering
    dict containing {} nothing - implemented for compatibility; is only returned, when verbose is True

    Raises
    ------
    NotImplementedError
        throwed if the specified filter method does not exist
    """
    seq = np.squeeze(seq)
    if filter_method == 'mean':
        hr = np.mean(seq)
    elif filter_method == 'median':
        hr = np.median(seq)
    else:
        raise NotImplementedError(
            'This filter_method <<<{}>>> is not yet implemented.'.format(
                filter_method))
    if verbose:
        return hr, {}
    return hr


def compute_hr_from_spectrum_peak_det(seq,
                                      sample_freq,
                                      freq_limits_bpm=(30, 200),
                                      freq_res_bpm=0.5,
                                      fft_window='hamming',
                                      peak_prom=0.5,
                                      verbose=False):
    """Compute the heart rate from the spectrum using a peak detection. This is useful if the spectrum probably exhibits higher peaks for the harmonic frequencies and a simple maximum detection would fail.

    Parameters
    ----------
    seq : 1D array like object
        any continuous physiological sequence containing the heart beat
    sample_freq : float
        the sampling frequency of the input sequence
    freq_limits_bpm : tuple, optional
        the boundaries within to search for the heart rate in beats per minute, by default (30, 200)
    freq_res_bpm : float, optional
        the resolution of the frequency for the power spectral density computation, by default 0.5
    fft_window : str, optional
        the window to use when computing the power spectral density, by default 'hamming'
    peak_prom : float, optional
        the peak prominence to use to identify the peaks whereas the minimum frequency is then assumed to represent the heart rate, by default 0.5
    verbose : bool, optional
        if additional results from the computation should also be returned, by default False

    Returns
    -------
    float
        the computed heart rate; if no peak could be found returns np.nan
    dict
        dict containing {'freq', 'power', 'power_n', 'peaks', 'nfft'}; is only returned, when verbose is True
    """
    seq = np.squeeze(seq)
    freq_limits_hz = (freq_limits_bpm[0]/60, freq_limits_bpm[1]/60)
    nyquist = sample_freq/2
    n = (60*2*nyquist)/freq_res_bpm
    # filter the signal to allow peak prominence to work correctly
    seq = apply_filter(seq, sample_freq, 3, freq_limits_bpm)
    freq, power = signal.periodogram(
        seq, fs=sample_freq, window=fft_window, nfft=n)
    # normalize to apply prominence properly
    power_n = power/power.max()
    peaks, _ = signal.find_peaks(power_n, prominence=peak_prom)
    # check if at least one peak could be found
    if len(peaks):
        hr = freq[peaks[0]] * 60
    else:
        hr = np.nan
        warnings.warn(
            'No peaks found during peak detection in periodogram...', UserWarning)

    if verbose:
        return hr, {'freq': freq,
                    'power': power,
                    'power_n': power_n,
                    'peaks': peaks,
                    'nfft': n}
    return hr


def compute_hr_from_spectrum_max(seq,
                                 sample_freq,
                                 freq_limits_bpm=(30, 200),
                                 freq_res_bpm=0.5,
                                 fft_window='hamming',
                                 verbose=False):
    """Computing the heart rate from the spectrum using simple maximum detection.

    Parameters
    ----------
    seq : 1D array like object
        any continuous physiological sequence containing the heart beat
    sample_freq : float
        the sampling frequency of the input sequence
    freq_limits_bpm : tuple, optional
        the boundaries within to search for the heart rate in beats per minute, by default (30, 200)
    freq_res_bpm : float, optional
        the resolution of the frequency for the power spectral density computation, by default 0.5
    fft_window : str, optional
        the window to use when computing the power spectral density, by default 'hamming'
    verbose : bool, optional
        if additional results from the computation should also be returned, by default False

    Returns
    -------
    float
        the computed heart rate; if no peak could be found returns np.nan
    dict
        dict containing {'freq', 'power', 'freq_roi', 'power_roi', 'nfft'}; is only returned, when verbose is True
    """
    seq = np.squeeze(seq)
    freq_limits_hz = (freq_limits_bpm[0]/60, freq_limits_bpm[1]/60)
    nyquist = sample_freq/2
    n = (60*2*nyquist)/freq_res_bpm
    # compute periodogram and analyse roi
    freq, power = signal.periodogram(
        seq, fs=sample_freq, window=fft_window, nfft=n)
    roi = (freq >= freq_limits_hz[0]) & (freq <= freq_limits_hz[1])
    freq_roi = freq[roi]
    power_roi = power[roi]
    hr = freq_roi[np.argmax(power_roi)] * 60

    if verbose:
        return hr, {'freq': freq,
                    'power': power,
                    'freq_roi': freq_roi,
                    'power_roi': power_roi,
                    'nfft': n}
    return hr


def compute_hr_from_b2b(seq,
                        sample_freq,
                        freq_limits_bpm=(30, 200)):
    raise NotImplementedError(
        'A robust peak detection for detecting heart beats from a physiological signal is not yet implemented.')


def compute_snr(seq,
                sample_freq,
                ref_hr_bpm,
                freq_limits_bpm=(24, 412),
                bin_ranges_bpm=(6, 12),
                freq_res_bpm=0.5,
                fft_window='hamming',
                verbose=False):
    """Implementation of the signal-to-noise ratio (SNR) computation for pulse rate extraction from blood volume pulse signals according to [1] with adjustments.

    Comment
    -------
    The original implementation has an upper freq limit of 240bpm and therefore systematically excludes the first harmonic for heart/pulse rates higher than 120bpm. To circumvent this issue the freq limit should be enlarged knowing that this lowers the SNR metric in general! See Notes for further information.


    Notes
    -----
        -   According to [1] The frequency range is originally defined as 0.667Hz (40bpm) to 4Hz (240bpm)
        -   Here, the physiological range for the heart rate is assumed to lie in between 0.5Hz (30bpm) and 3.33Hz (200bpm) according to the IEC 60601-2-27
        -   Remember to decrease/increase the lower/upper frequency bounds to include the bins around the fundamental 
            oscillation and the first harmonic for the physiological range
        -   Finally, for the previously defined physiological range of 30 to 200bpm and a bin range of 6 and 12bpm 
            around the fundamental oscillation and the first harmonic the frequency bounds have to be 24 to 412bpm
        -   Nevertheless, the pulse rate peak (fundamental oscillation) is searched between the frequency range not 
            including the bins, i.e. between 30 and 200bpm for the default function values

    References
    ----------
        -   [1] Haan, Gerard de; Jeanne, Vincent (2013): Robust pulse rate from chrominance-based rPPG. In: IEEE
            transactions on bio-medical engineering 60 (10), S. 2878–2886. DOI: 10.1109/TBME.2013.2266196.

    Parameters
    ----------
    seq : 1D array like object
        blood volume pulse (BVP) signal
    sample_freq : float
        sample frequency of the BVP signal
    ref_hr_bpm : float
        the reference heart rate used to calculate the SNR
    freq_limits_bpm : tuple, optional
        frequency limits for the computation of the SNR; Remember this is the combination of the physiological range and the bin range, by default (24, 412) (i.e. assuming a physiological heart rate between 30 and 200bpm)
    bin_ranges_bpm : tuple, optional
        frequency range around the fundamental oscillation and the first harmonic of the BVP signal, by default (6, 12)
    freq_res_bpm : float, optional
        frequency resolution for the fourier transformation, by default 0.5
    fft_window : str, optional
        the window type to use for the computation of the power spectral density, by default 'hamming'
    verbose : bool, optional
        output additional results from fft and signal power computation, by default False

    Returns
    -------
    float
        signal-to-noise ratio (SNR)
    dict
        dict containing {'freq', 'power', 's_power', 'all_power', 'freq_limits_hz', 'roi_f_0', 'roi_f_1', 'roi_f_all', ref_hr_bpm', 'nfft'}; is only returned, when verbose is True
    """
    seq = np.squeeze(seq)
    freq_limits_hz = (freq_limits_bpm[0]/60, freq_limits_bpm[1]/60)
    bin_ranges_hz = (bin_ranges_bpm[0]/60, bin_ranges_bpm[1]/60)

    ref_hr_hz = ref_hr_bpm/60

    nyquist = sample_freq/2
    n = (60*2*nyquist)/freq_res_bpm

    freq, power = signal.periodogram(
        seq, fs=sample_freq, window=fft_window, nfft=n)

    # define freq rois for fundamental oscillation and its first harmonic
    roi_f_0 = (freq >= ref_hr_hz -
               bin_ranges_hz[0]) & (freq <= ref_hr_hz + bin_ranges_hz[0])
    roi_f_1 = (freq >= ref_hr_hz*2 -
               bin_ranges_hz[1]) & (freq <= ref_hr_hz*2 + bin_ranges_hz[1])
    # keep snr calculation in range of the freq_limits
    roi_f_0 = roi_f_0 & (freq >= freq_limits_hz[0]) & (
        freq <= freq_limits_hz[1])
    roi_f_1 = roi_f_1 & (freq >= freq_limits_hz[0]) & (
        freq <= freq_limits_hz[1])

    s_power = np.sum(power[np.logical_or(roi_f_0, roi_f_1)])

    roi_f_all = (freq >= freq_limits_hz[0]) & (freq <= freq_limits_hz[1])
    all_power = np.sum(power[roi_f_all])

    snr = 10*np.log10(s_power/(all_power-s_power))

    if verbose:
        return snr, {'freq': freq,
                     'power': power,
                     's_power': s_power,
                     'all_power': all_power,
                     'freq_limits_hz': freq_limits_hz,
                     'roi_f_0': roi_f_0,
                     'roi_f_1': roi_f_1,
                     'roi_f_all': roi_f_all,
                     'ref_hr_bpm': ref_hr_bpm}
    return snr


def apply_filter(seq,
                 sample_freq,
                 order,
                 cutoff_bpm):
    """Filter some signal using a butterworth lowpass or bandpass filter. Note that the resulting filter order is 4 * order because: order * 2 for bandpass butterworth and order * 2 for zero-phase filtering.

    Parameters
    ----------
    seq : 1D array like object
        the sequence to be filtered
    sample_freq : float
        the sampling frequency of the sequence
    order : int
        the filter order. Note that the resulting filter order will be 4*order
    cutoff_bpm : int or tuple of integers (e.g. (30, 200))
        the cutoff frequencies for the filter in beats per minute not Hz! (for convenience); if int than a lowpass filter is applied; if tuple than a bandpass filter is applied

    Returns
    -------
    numpy.ndarray
        the filtered input sequence
    """
    seq = np.squeeze(seq)
    if type(cutoff_bpm) == tuple:
        cutoff_hz = (cutoff_bpm[0]/60, cutoff_bpm[1]/60)
        btype = 'bandpass'
    else:
        cutoff_hz = cutoff_bpm/60
        btype = 'lowpass'
    sos = signal.butter(order, cutoff_hz, btype=btype,
                        fs=sample_freq, output='sos')
    seq_filt = signal.sosfiltfilt(sos, seq)
    return seq_filt


def resample_sequence(seq,
                      new_sample_freq,
                      sample_freq=None,
                      seq_ts=None,
                      is_datetime=False):
    """Resample a given signal to a new sampling frequency. Note that the timestamps are expected to be either in datetime format or in float (as microseconds!).

    Parameters
    ----------
    seq : 1D or 2D array like object (when 2D than first dimension must represent the channels)
        the signal values that you want to resample
    new_sample_freq : float
        the sample frequency of the returned sequence in Hz
    sample_freq: float
        the sample frequency of the given sequence in Hz; either seq_ts or sample_freq must be given, by default None
    seq_ts : 1D array like object
        the corresponding timestamps of the given sequence; can be timestamps or floats (representing microseconds); either seq_ts or sample_freq must be given, by default None
    is_datetime : bool, optional
        indicate if the seq_ts is given as sequence of datetime or floats, by default False

    Returns
    -------
    dict
        dictionary with keys: seq_interp, ts_interp, interp_func
    """
    if (sample_freq is None and seq_ts is None) or \
       (sample_freq is not None and seq_ts is not None):
        raise ValueError('Either the sample_freq or the seq_ts must be given!')

    seq = np.squeeze(seq)
    if seq_ts is not None:
        if is_datetime:
            # convert timestamps to floats
            seq_ts = np.asarray([(ts/timedelta(microseconds=1))
                                 for ts in (np.asarray(seq_ts) - seq_ts[0])])
    else:
        seq_ts = np.arange(0, len(seq)) * 1/sample_freq * 1e6

    period = 1e6/new_sample_freq
    seq_ts_new = np.arange(seq_ts[0], seq_ts[-1], period)
    interp_func = interpolate.interp1d(seq_ts.squeeze(), seq)

    return {'seq_interp': interp_func(seq_ts_new),
            'ts_interp': seq_ts_new,
            'interp_func': interp_func}
