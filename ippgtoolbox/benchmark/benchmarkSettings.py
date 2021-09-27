"""
------------------------------------------------------------------------------
Created: 06.02.2021, 22:05
------------------------------------------------------------------------------
Author: Matthieu Scherpf
Email: Matthieu.Scherpf@tu-dresden.de
Website: https://becuriouss.github.io/matthieu-scherpf/
Project page: tba
------------------------------------------------------------------------------
Purpose: Basic configuration for the benchmarks (freq. bounds etc.)
------------------------------------------------------------------------------
"""

# shortcuts for export
__all__ = [
    'settings',
]


class BenchmarkSettings:
    """Settings to use for the benchmarking process. These parameters should be changed before you run the whole evaluation! The default parameters are based on literature and specific reasoning (see descriptions for further information). This class definition is primarily intended to improve reproducibility.
    """

    def FFT_WINDOW(self, overwrite=None):
        """Window type to use for both computation of fft and periodogram.

        Parameters
        ----------
        overwrite : str, optional
            representing one of the available numpy window functions, by default None
        """
        return overwrite or 'hamming'

    def HR_SEQUENCE_FILTER_METHOD(self, overwrite=None):
        """The extraction method to use for a sequence of values each representing a measured heart rate. Available values can be found in the corresponding function used within the evaluation.

        Parameters
        ----------
        overwrite : {'mean', 'median'}, optional
            representing one of the available methods, by default None
        """
        return overwrite or 'median'

    def HR_EXTRACTION_FROM_DOMAIN(self, overwrite=None):
        """The extraction method to use for a sequence of values each representing the blood volume pulse, the continuous blood pressure or any other signal containing the heart pulsation. Available values are 'time_domain' and 'freq_domain'

        Parameters
        ----------
        overwrite : {'freq_domain', 'time_domain'}, optional
            representing one of the available methods; available options are 'freq_domain' or 'time_domain'. If 'freq_domain', then the power spectral density is used for the heart rate calculation. If 'time_domain' then a beat-to-beat detection with an additional filter method defined by HR_SEQUENCE_FILTER_METHOD() is used; by default None
        """
        return overwrite or 'freq_domain'

    def HR_EXTRACTION_INTERVAL_LENGTH_SEC(self, overwrite=None):
        """Length of the intervals to compute the given metrics and heart rate.

        Parameters
        ----------
        overwrite : float, optional
            representing seconds and fraction of seconds; the interval length used for the heart rate extraction; note that if the heart rate is extracted from the time domain by using a beat-to-beat approach an additional filter method is used to reduce the values to on single value; by default None
        """
        return overwrite or 10.

    def HR_EXTRACTION_INTERVAL_STRIDE_SEC(self, overwrite=None):
        """Stride to use for computing the metrics and heart rate over a given sequence.

        Parameters
        ----------
        overwrite : float, optional
            representing seconds; the stride used when building intervals , by default None
        """
        return overwrite or 1.

    def PERIODOGRAM_FREQ_RESOLUTION_BPM(self, overwrite=None):
        """Frequency resolution when computing the periodogram.

        Parameters
        ----------
        overwrite : float, optional
            the frequency resolution when computing the power spectral density (shouldn't be greater than 0.5), by default None
        """
        return overwrite or 0.1

    def PERIODOGRAM_PEAK_PROMINENCE(self, overwrite=None):
        """If the peaks are detected in the periodogram rather than the simple maximum there must be the minimum peak prominence defined. It is related to the max-normalized power spectral density. It is usually applied for the reference signal as this delivers a quite clean spectrum with little noise.

        Parameters
        ----------
        overwrite : float, optional
            prominence used for the peak detection in the normalized power spectral density, by default None
        """
        return overwrite or 0.5

    def BOUNDARIES_HR_BPM(self, overwrite=None):
        """Lower and upper frequency boundaries to use for heart/pulse rate estimation in beats per minute. These values are chosen according to the IEC 60601-2-27 (specifications for heart rate ecg measurement)

        Parameters
        ----------
        overwrite : tuple of integers (e.g. (30, 200)), optional
            first and second value describing the lower and upper boundary in the power spectral density, by default None
        """
        return overwrite or (30, 200)

    def BIN_RANGE_HR_BPM(self, overwrite=None):
        """Frequency range around the fundamental oscillation and its first harmonic in beats per minute. The ranges are chosen to capture small heart rate variations.

        Parameters
        ----------
        overwrite : tuple of integers (e.g. (6, 12)), optional
            first and second value describing the lower and upper range around the fundamental oscillation and its first harmonic, by default None
        """
        return overwrite or (6, 12)

    def BOUNDARIES_SNR_BPM(self):
        """Lower and upper frequency boundaries to use for signal-to-noise estimation in beats per minute. The boundaries result from the boundaries for the heart rate estimation and the range around the fundamental oscillation and its first harmonic. 
        """
        lower = self.BOUNDARIES_HR_BPM()[0] - self.BIN_RANGE_HR_BPM()[0]
        upper = self.BOUNDARIES_HR_BPM()[1] * 2 + self.BIN_RANGE_HR_BPM()[1]
        return (lower, upper)

    def INTERVAL_TEMPORAL_NORMALIZATION_SEC(self, overwrite=None):
        """Interval size in seconds to use for the temporal normalization. The value is chosen according to Wang et al. in:

        W. Wang, A. C. Den Brinker, S. Stuijk, and G. de Haan, “Algorithmic Principles of Remote PPG,” IEEE transactions on bio-medical engineering, vol. 64, no. 7, pp. 1479–1491, 2017, doi: 10.1109/TBME.2016.2609282.

        Parameters
        ----------
        overwrite : float, optional
            representing seconds; the length of the interval used for performing the temporal normalization, by default None
        """
        return overwrite or 2.

    def INTERVAL_COLOR_DISTORTION_FILTERING_SEC(self, overwrite=None):
        """Interval size in seconds to use for performing the color distortion filtering. 'max' means the whole signal will be used. Results from Wang et al. implicitly indicate: the longer the better:

        W. Wang, A. C. Den Brinker, S. Stuijk, and G. de Haan, “Algorithmic Principles of Remote PPG,” IEEE transactions on bio-medical engineering, vol. 64, no. 7, pp. 1479–1491, 2017, doi: 10.1109/TBME.2016.2609282.

        Parameters
        ----------
        overwrite : float, optional
            representing seconds; the length of the interval used for performing the color distortion filtering, by default None
        """
        return overwrite or 'max'

    def INTERVAL_OVERLAP_COLOR_DISTORTION_FILTERING_PERC(self, overwrite=None):
        """Interval overlap if INTERVAL_SEC_COLOR_DISTORTION_FILTERING is not set to 'max'. Lies between in the range [0., 1.)

        Parameters
        ----------
        overwrite : float between [0., 1.), optional
            overlap of the intervals built within the color distortion filtering, by default None
        """
        return overwrite or 0.

    def APPLY_COLOR_DISTORTION_FILTERING(self, overwrite=None):
        """If the color distortion filtering technique should be applied.

        Parameters
        ----------
        overwrite : boolean, optional
            if the color distortion should be applied or not, by default None
        """
        return overwrite or False

    def RESPECT_BORDER_EFFECTS(self, overwrite=None):
        """If border effects should be respected by cutting the signal, i.e. for CHROM or POS.

        Parameters
        ----------
        overwrite : boolean, optional
            if border effects should be respected or not, by default None
        """
        return overwrite or False

    def FILTER_ORDER(self, overwrite=None):
        """Order of the butterworth filter that is used. Consider that the filtering is done using zero-phase filtering which means the filtering is done in both directions of the signal. Therefore, the resulting filter order is the double. When using the (butterworth) 'bandpass' filter option the filter order is as well the double. Therefore, when using 'bandpass' filter the resulting filter order is 4 times larger than specified here.

        Parameters
        ----------
        overwrite : int, optional
            the filter order used for the butterworth filter design, by default None
        """
        return overwrite or 3

    def FILTER_TYPE(self, overwrite=None):
        """Filter type to use.

        Parameters
        ----------
        overwrite : {'bandpass', 'bandstop'}, optional
            type of filter to use. Note that this is also dependent of the boundaries chosen for filtering. Options are 'bandpass' and 'bandstop'; by default None
        """
        return overwrite or 'bandpass'


# create shortcuts for export
settings = BenchmarkSettings()
