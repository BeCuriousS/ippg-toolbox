"""
 -----------------------------------------------------------------------------
Created: 10.02.2021, 18:23
 -----------------------------------------------------------------------------
Author: Matthieu Scherpf
Email: Matthieu.Scherpf@tu-dresden.de
Website: https://becuriouss.github.io/matthieuscherpf/
Project page: tba
 -----------------------------------------------------------------------------
Purpose: Compute the features like heart rate etc. for a given extracted sequence.
 -----------------------------------------------------------------------------
"""
import os
import pickle
import json
import warnings

from .benchmarkSettings import settings
from .benchmarkMetrics import BenchmarkMetrics
from .processing import *


class ProcessExtraction:
    """Evaluator for the computation of the given metrics for a given blood volume pulse signal. Within the documentation of this class, we do not differentiate between heart and pulse rate allthough they need to be distinguished within a medical/physiological context!
    """

    def __init__(self,
                 ref_seq,
                 ref_sample_freq,
                 ref_is_hr_seq=False,
                 resample_freq=None,
                 verbose=False):

        # ----------------------------------------------------------
        # define shortcuts
        # ----------------------------------------------------------
        self._fft_w = settings.FFT_WINDOW()
        self._hr_seq_filt = settings.HR_SEQUENCE_FILTER_METHOD()
        self._hr_from_dom = settings.HR_EXTRACTION_FROM_DOMAIN()
        self._w_length = settings.HR_EXTRACTION_INTERVAL_LENGTH_SEC()
        self._w_stride = settings.HR_EXTRACTION_INTERVAL_STRIDE_SEC()
        self._f_res = settings.PERIODOGRAM_FREQ_RESOLUTION_BPM()
        self._prom = settings.PERIODOGRAM_PEAK_PROMINENCE()
        self._hr_bounds = settings.BOUNDARIES_HR_BPM()
        self._snr_bounds = settings.BOUNDARIES_SNR_BPM()
        self._bin_rng = settings.BIN_RANGE_HR_BPM()

        self.ref_seq = np.squeeze(ref_seq)
        self.ref_sample_freq = ref_sample_freq
        self.ref_is_hr_seq = ref_is_hr_seq
        self.verbose = verbose
        self.report = {}

        self._preprocess_reference_sequence(resample_freq)
        self._compute_reference_features()

    def compute_features_and_metrics(self,
                                     seq,
                                     sample_freq,
                                     dict_key_prefix=None):
        seq = np.squeeze(seq)
        if self._hr_from_dom == 'freq_domain':
            handle = compute_hr_from_spectrum_max
            kwargs = {
                'freq_limits_bpm': self._hr_bounds,
                'freq_res_bpm': self._f_res,
                'fft_window': self._fft_w
            }
        elif self._hr_from_dom == 'time_domain':
            handle = compute_hr_from_b2b
            kwargs = {}
        args = (seq, sample_freq)
        hr, hr_data = self._convenient_helper(handle, args, kwargs)
        self.report[dict_key_prefix+'_hr'] = hr
        self.report[dict_key_prefix+'_hr_data'] = hr_data
        # if from frequency domain, also compute snr
        if self._hr_from_dom == 'freq_domain':
            handle = compute_snr
            # snr vs reference hr
            kwargs = {
                'ref_hr_bpm': self.ref_hr,
                'freq_limits_bpm': self._snr_bounds,
                'bin_ranges_bpm': self._bin_rng,
                'freq_res_bpm': self._f_res,
                'fft_window': self._fft_w,
            }
            snr, snr_data = self._convenient_helper(
                handle, args, kwargs)
            self.report[dict_key_prefix+'_snr_vs_ref'] = snr
            self.report[dict_key_prefix+'_snr_vs_ref_data'] = snr_data
            # snr vs sequence hr
            kwargs = {
                'ref_hr_bpm': hr,
                'freq_limits_bpm': self._snr_bounds,
                'bin_ranges_bpm': self._bin_rng,
                'freq_res_bpm': self._f_res,
                'fft_window': self._fft_w,
            }
            snr, snr_data = self._convenient_helper(
                handle, args, kwargs)
            self.report[dict_key_prefix+'_snr'] = snr
            self.report[dict_key_prefix+'_snr_data'] = snr_data

    def get_report(self):
        return self.report

    def _preprocess_reference_sequence(self, resample_freq):
        if resample_freq is not None:
            self.ref_seq = resample_sequence(
                self.ref_seq, resample_freq, self.ref_sample_freq)
            self.ref_sample_freq = resample_freq
        self.report['ref_seq'] = self.ref_seq
        self.report['ref_sample_freq'] = self.ref_sample_freq

    def _compute_reference_features(self):
        if self.ref_is_hr_seq:
            handle = compute_hr_from_hr_sequence
            kwargs = {'filter_method': self._hr_seq_filt}
        else:
            if self._hr_from_dom == 'freq_domain':
                handle = compute_hr_from_spectrum_peak_det
                kwargs = {
                    'freq_limits_bpm': self._hr_bounds,
                    'freq_res_bpm': self._f_res,
                    'fft_window': self._fft_w,
                    'peak_prom': self._prom
                }
            elif self._hr_from_dom == 'time_domain':
                handle = compute_hr_from_b2b
                kwargs = {}
        args = (self.ref_seq, self.ref_sample_freq)
        self.ref_hr, self.ref_hr_data = self._convenient_helper(
            handle, args, kwargs)
        self.report['ref_hr'] = self.ref_hr
        self.report['ref_hr_data'] = self.ref_hr_data
        # if from frequency domain, also compute snr
        if self._hr_from_dom == 'freq_domain':
            handle = compute_snr
            kwargs = {
                'ref_hr_bpm': self.ref_hr,
                'freq_limits_bpm': self._snr_bounds,
                'bin_ranges_bpm': self._bin_rng,
                'freq_res_bpm': self._f_res,
                'fft_window': self._fft_w,
            }
            self.ref_snr, self.ref_snr_data = self._convenient_helper(
                handle, args, kwargs)
            self.report['ref_snr'] = self.ref_snr
            self.report['ref_snr_data'] = self.ref_snr_data

    def _convenient_helper(self, handle, args, kwargs):
        out, out_data = compute_window_based_feature(
            *args,
            handle,
            self._w_length,
            self._w_stride,
            verbose=True,
            **kwargs
        )
        return out, out_data
