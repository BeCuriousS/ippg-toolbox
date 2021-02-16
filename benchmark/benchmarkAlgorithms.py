"""
------------------------------------------------------------------------------
Created: 05.02.2021, 21:27
------------------------------------------------------------------------------
Author: Matthieu Scherpf
Email: Matthieu.Scherpf@tu-dresden.de
Website: https://becuriouss.github.io/matthieuscherpf/
Project page: tba
------------------------------------------------------------------------------
Purpose: Implementation of the "standard" algorithms used for comparison in the related publication
------------------------------------------------------------------------------
"""
import numpy as np
from .benchmarkSettings import settings
from .processing import apply_filter


class BenchmarkAlgorithms:
    """Selection of different state of the art algorithms for comparison. The color distortion filtering from Wang et al. is also implemented (see method description for further information).
    """

    def __init__(self, rgb_seq, sample_freq, apply_cdf=False, normalize=False):
        """Use this class to evaluate any [R,G,B]-sequence extracted from the mean of any region of interest.

        Parameters
        ----------
        rgb_seq : numpy.ndarray with shape [n, 3] where n is the number of sample points
            the color channel wise mean of any region of interest of some video recording
        sample_freq : float
            the sample frequency of the rgb_seq, i.e. the recorded video (equal to frames per second)
        apply_cdf : bool, optional
            if the color distortion filtering method should be applied before extracting the blood volume pulse, by default False
        normalize : bool, optional
            if the signal should be l2-normalized before extracting the blood volume pulse, by default False
        """
        # ----------------------------------------------------------
        # define shortcuts
        # ----------------------------------------------------------
        self._fft_w = settings.FFT_WINDOW()
        self._snr_bounds = settings.BOUNDARIES_SNR_BPM()
        self._tmp_norm_len = settings.\
            INTERVAL_TEMPORAL_NORMALIZATION_SEC()
        self._cdf_len = settings.\
            INTERVAL_COLOR_DISTORTION_FILTERING_SEC()
        self._cdf_olap = settings.\
            INTERVAL_OVERLAP_COLOR_DISTORTION_FILTERING_PERC()
        self._filt_o = settings.FILTER_ORDER()

        self.sample_freq = sample_freq
        self.apply_cdf = apply_cdf
        self.normalize = normalize

        self.rgb_seq = self._preprocess(rgb_seq)

    def _preprocess(self, rgb_seq):
        if self._cdf_len == 'max':
            self._cdf_len = np.ceil(1/self.sample_freq * rgb_seq.shape[0])
        if self.apply_cdf:
            rgb_seq = self.filter_color_distortions(rgb_seq)
        if self.normalize:
            rgb_seq = rgb_seq / np.linalg.norm(rgb_seq, ord=2,
                                               axis=1, keepdims=True)
        return rgb_seq

    def filter_color_distortions(self, rgb_seq):
        """Color distortion filtering to reduce distortions by wighting selected frequency components; introduced by Wang et al. in:

        Wang, Wenjin; Den Brinker, Albertus C.; Stuijk, Sander; Haan, Gerard de (2017): Color-Distortion Filtering for Remote Photoplethysmography. In: 12th IEEE International Conference on Automatic Face and Gesture Recognition - FG 2017. pp. 71–78. DOI: 10.1109/FG.2017.18.

        Notes
        -----
        We did not implement any FFT windowing as it is not specified in the original paper.

        Parameters
        ----------
        rgb_seq : numpy.ndarray with shape [n, 3] where n is the number of sample points
            the color channel wise mean of any region of interest of some video recording

        Returns
        -------
        same as rgb_seq
            the filtered rgb_seq
        """
        rgb_seq_filt = np.zeros(rgb_seq.shape)
        idx_start = 0
        diff = 0
        while idx_start < rgb_seq.shape[0]:
            idx_end = int(idx_start + self.sample_freq * self._cdf_len)
            wsize = int(np.ceil((idx_end - idx_start) * (1 - self._cdf_olap)))
            interval = rgb_seq[idx_start:idx_end]  # line 0
            x = np.arange(0, interval.shape[0]) * \
                self.sample_freq / interval.shape[0]  # line 1
            mean = np.mean(interval, axis=0)  # line 2
            cn = interval / mean - 1  # line 2
            f = np.fft.fft(cn, axis=0)  # line 3
            p = np.array([-1, 2, -1])[:, np.newaxis] / np.sqrt(6)
            s = np.dot(f, p)  # line 4
            w = (s * np.conj(s)) / np.sum(f * np.conj(f),
                                          axis=1, keepdims=True)  # line 5
            f_mask = (x < self._snr_bounds[0] /
                      60) | (x > self._snr_bounds[1]/60)
            w[f_mask, :] = 0  # line 6
            fw = f * np.tile(w, (1, 3))  # line 7
            cw = mean * (np.real(np.fft.ifft(fw, axis=0)) + 1)  # line 8
            if idx_start > 0:
                diff = np.mean(cw[0:wsize], axis=0) - \
                    np.mean(rgb_seq_filt[idx_start-wsize:idx_start], axis=0)
            rgb_seq_filt[idx_start:idx_start+wsize] = cw[0:wsize] - diff
            idx_start += wsize
        return rgb_seq_filt

    def extract_bvp_CHROM(self):
        """Implementation of chrominance-based blood volume pulse extraction introduced by de Haan et al. in:

        Haan, Gerard de; Jeanne, Vincent (2013): Robust pulse rate from chrominance-based rPPG. In: IEEE transactions on bio-medical engineering 60 (10), S. 2878–2886. DOI: 10.1109/TBME.2013.2266196.

        Notes
        -----
        The frequency boundaries for the filtering steps are adjusted! This is done to avoid possible systematic improvement over the other benchmark algorithms.

        Returns
        -------
        numpy.ndarray of shape [n, 1] where n not necessarily matches the size of the original rgb_seq due to conditions regarding the sample frequency and the interval size
            the blood volume pulse extracted from the rgb_seq
        """
        tmp_norm_n_frames = int(np.ceil(self.sample_freq * self._tmp_norm_len))
        if tmp_norm_n_frames % 2 > 0:
            tmp_norm_n_frames += 1
        # (final) projection matrix used by CHROM (Pc * M)
        pm = np.array([
            [3,    -2,    0],
            [1.5,   1,   -1.5]
        ])
        # compute bvp
        n = self.rgb_seq.shape[0]  # number of sample points, i.e. frames
        w_hann = np.hanning(tmp_norm_n_frames)
        segments = []
        c = self.rgb_seq
        # loop over overlapping windows
        i_start = 0
        i_end = i_start + tmp_norm_n_frames
        while i_end <= n:
            # temporal normalization
            cn = c[i_start:i_end] / np.mean(c[i_start:i_end], axis=0) - 1
            # projection
            s = np.matmul(pm, np.transpose(cn))
            xs = s[0, :]
            ys = s[1, :]
            # filtering
            xf = apply_filter(xs,
                              self.sample_freq,
                              order=self._filt_o,
                              cutoff_bpm=self._snr_bounds)
            yf = apply_filter(ys,
                              self.sample_freq,
                              order=self._filt_o,
                              cutoff_bpm=self._snr_bounds)
            # tuning
            h = xf - (xf.std() / yf.std()) * yf
            # weighting with hann window
            h = w_hann * h
            # insert weighted segment with related indexes to list
            segments.append((i_start, i_end, h))
            # update indexes
            i_start += tmp_norm_n_frames // 2
            i_end = i_start + tmp_norm_n_frames
        # combine segments by overlap adding
        h_length = (len(segments) + 1) * (tmp_norm_n_frames//2)
        h = np.zeros((1, h_length))
        for i_start, i_end, s in segments:
            h[0, i_start:i_end] += s

        return h.squeeze()[:, np.newaxis]

    def extract_bvp_POS(self):
        """Implementation of plane-orthogonal-to-skin based blood volume pulse extraction introduced by Wang et al. in:

        W. Wang, A. C. Den Brinker, S. Stuijk, and G. de Haan, “Algorithmic Principles of Remote PPG,” IEEE transactions on bio-medical engineering, vol. 64, no. 7, pp. 1479–1491, 2017, doi: 10.1109/TBME.2016.2609282.

        Returns
        -------
        numpy.ndarray of shape [n, 1] where n matches the size of the original rgb_seq
            the blood volume pulse extracted from the rgb_seq
        """
        tmp_norm_n_frames = int(np.ceil(self.sample_freq * self._tmp_norm_len))
        # projection matrix used by POS
        pm = np.array([
            [0,   1,  -1],
            [-2,   1,   1]
        ])
        # compute bvp
        n = self.rgb_seq.shape[0]  # number of sample points, i.e. frames
        h = np.zeros((1, n))
        c = self.rgb_seq
        # loop over overlapping windows
        for i in range(n):
            m = i - tmp_norm_n_frames
            if m >= 0:
                # temporal normalization
                cn = c[m:i] / np.mean(c[m:i], axis=0)
                # projection
                s = np.matmul(pm, np.transpose(cn))
                s1 = s[0, :]
                s2 = s[1, :]
                # tuning
                hi = s1 + (s1.std() / s2.std()) * s2
                # overlap-adding
                h[0, m:i] = h[0, m:i] + (hi - hi.mean())

        return h.squeeze()[:, np.newaxis]

    def extract_bvp_O3C(self):
        """Implementation of the optimal color channel combination based blood volume pulse extraction introduced by Ernst et al. in:

        # TODO Add the related official publication as soon as it is available

        Returns
        -------
        numpy.ndarray of shape [n, 1] where n matches the size of the original rgb_seq
            the blood volume pulse extracted from the rgb_seq
        """
        # projection axis used by O3C
        pa = np.array([0.25, -0.83, 0.5])

        return np.matmul(self.rgb_seq, pa).squeeze()[:, np.newaxis]

    def extract_all(self):
        """Convenient method to allow extraction of all benchmark algorithms at once.

        Returns
        -------
        dict with one key/value pair for each algorithm
            each entry in the dictionary contains an numpy.ndarray of shape [n, 3]. Note that the first dimension can vary for CHROM. See algorithm method description for further information.
        """
        bvp_data = {}
        bvp_data['CHROM'] = self.extract_bvp_CHROM()
        bvp_data['POS'] = self.extract_bvp_POS()
        bvp_data['O3C'] = self.extract_bvp_O3C()

        return bvp_data
