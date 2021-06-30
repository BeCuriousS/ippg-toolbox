"""
-----------------------------------------------------------------------------
Created: 11.02.2021, 18:04
-----------------------------------------------------------------------------
Author: Matthieu Scherpf
Email: Matthieu.Scherpf@tu-dresden.de
Website: https://becuriouss.github.io/matthieuscherpf/
Project page: tba
-----------------------------------------------------------------------------
Purpose: Tests for the implemented processing. The tests are based on a simple plausibility check by plotting the computation results.
-----------------------------------------------------------------------------
"""
# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from ippgtoolbox.processing import processing

TEST_sample_freq_ref_bp = 1000
TEST_abs_file_path_ref_bp = os.path.join(
    './assets/bp4d_example/BP_mmHg.txt')
TEST_sample_freq_ref_hr_seq = 1000
TEST_abs_file_path_ref_hr_seq = os.path.join(
    './assets/bp4d_example/Pulse Rate_BPM.txt')
TEST_sample_freq_bvp = 25
TEST_abs_file_path_bvp = os.path.join(
    './assets/bp4d_example/bvp_CHROM_cdf.mat')

plt.rc('lines', linewidth=0.2)


class TestProcessing:

    def __init__(self):
        self.bp = np.loadtxt(TEST_abs_file_path_ref_bp)
        self.hr_seq = np.loadtxt(TEST_abs_file_path_ref_hr_seq)
        self.bvp = loadmat(TEST_abs_file_path_bvp)['CHROM']
        self._test_plot_seq(self.bp, 'Continuous blood pressure')
        self._test_plot_seq(self.hr_seq, 'Reference HR sequence')
        self._test_plot_seq(self.bvp, 'Extracted blood volume pulse')

    def test_compute_hr_from_hr_sequence(self):
        self.vals, self.data = processing.compute_window_based_feature(
            self.hr_seq,
            TEST_sample_freq_ref_hr_seq,
            processing.compute_hr_from_hr_sequence,
            window_length=10,
            window_stride=1,
            verbose=True
        )
        self._test_plot('HR from HR sequence')

    def test_compute_hr_from_spectrum_peak_det(self):
        self.vals, self.data = processing.compute_window_based_feature(
            self.bp,
            TEST_sample_freq_ref_bp,
            processing.compute_hr_from_spectrum_peak_det,
            window_length=10,
            window_stride=1,
            verbose=True
        )
        self._test_plot('HR from cbp spectrum peak det')

    def test_compute_hr_from_spectrum_max(self):
        self.vals, self.data = processing.compute_window_based_feature(
            self.bp,
            TEST_sample_freq_ref_bp,
            processing.compute_hr_from_spectrum_max,
            window_length=10,
            window_stride=1,
            verbose=True
        )
        self._test_plot('HR from cbp spectrum max')

    def test_compute_snr(self):
        ref_hr = processing.compute_window_based_feature(
            self.bvp,
            TEST_sample_freq_bvp,
            processing.compute_hr_from_spectrum_max,
            window_length=10,
            window_stride=1,
            verbose=False
        )
        self._test_plot_seq(ref_hr, 'HR for extracted blood volume pulse')
        self.vals, self.data = processing.compute_window_based_feature(
            self.bvp,
            TEST_sample_freq_bvp,
            processing.compute_snr,
            window_length=10,
            window_stride=1,
            verbose=True,
            ref_hr_bpm=ref_hr,
            # freq_res_bpm=1,
        )
        self._test_plot('SNR for extracted blood volume pulse')
        # # plot spectrogram
        # f, t, p = ([], [], [])
        # for i, w in enumerate(self.data['w_data']):
        #     f.append(w['freq'][w['roi_f_all']])
        #     p.append(w['power'][w['roi_f_all']])
        #     t.append(i+1)
        # f = f[0] * 60  # convert to bpm
        # t = np.asarray(t)
        # p = np.transpose(np.asarray(p))
        # plt.figure()
        # plt.pcolormesh(t, f, p, shading='gouraud')
        # plt.ylabel('Frequency [Hz]')
        # plt.xlabel('Time [sec]')

    def test_apply_filter(self):
        self.vals = processing.apply_filter(
            self.bvp,
            TEST_sample_freq_bvp,
            order=3,
            cutoff_bpm=(30, 120)
        )
        self._test_plot('Filtered blood volume pulse signal')

    def test_resample_sequence(self):
        self.vals = processing.resample_sequence(
            self.bp,
            new_sample_freq=30,
            sample_freq=TEST_sample_freq_ref_bp,
        )
        self._test_plot('Resampled cbp by sample freq')
        self.vals = processing.resample_sequence(
            self.bp,
            new_sample_freq=30,
            seq_ts=np.arange(0, len(self.bp))*1e3,
        )
        self._test_plot('Resampled cbp by timestamps')

    def _test_plot(self, title):
        plt.figure()
        plt.title(title)
        plt.plot(self.vals)

    def _test_plot_seq(self, seq, title):
        plt.figure()
        plt.title(title)
        plt.plot(seq)


if __name__ == '__main__':

    testProcessing = TestProcessing()
    testProcessing.test_compute_hr_from_hr_sequence()
    testProcessing.test_compute_hr_from_spectrum_peak_det()
    testProcessing.test_compute_hr_from_spectrum_max()
    testProcessing.test_compute_snr()
    testProcessing.test_apply_filter()
    testProcessing.test_resample_sequence()

# %%
