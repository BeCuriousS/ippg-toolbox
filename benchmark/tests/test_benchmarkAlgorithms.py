"""
------------------------------------------------------------------------------
Created: 06.02.2021, 13:04
------------------------------------------------------------------------------
Author: Matthieu Scherpf
Email: Matthieu.Scherpf@tu-dresden.de
Website: https://becuriouss.github.io/matthieuscherpf/
Project page: tba
------------------------------------------------------------------------------
Purpose: Tests the implementation of the standard algorithms used as benchmark.
The tests are based on a simple plausibility check by plotting the signals.
------------------------------------------------------------------------------
"""
# %%
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from benchmark import BenchmarkAlgorithms

TEST_sample_freq = 30
TEST_abs_file_path = os.path.join('./assets/ubfc_example/mean_rgb.mat')


plt.rc('lines', linewidth=0.2)


class TestBenchmarkAlgorithms:

    def __init__(self):
        self.rgb_seq = loadmat(TEST_abs_file_path)['mean_rgb']
        self.ba = BenchmarkAlgorithms(
            self.rgb_seq, TEST_sample_freq)
        self._test_plot('RGB mean from ROI')

    def test_filter_color_distortions(self):
        self.ba = BenchmarkAlgorithms(
            self.rgb_seq, TEST_sample_freq, apply_cdf=True)
        self._test_plot('RGB mean with CDF')

    def test_normalization(self):
        self.ba = BenchmarkAlgorithms(
            self.rgb_seq, TEST_sample_freq, normalize=True)
        self._test_plot('RGB mean with normalization')

    def test_extract_bvp_CHROM(self):
        self.ba = BenchmarkAlgorithms(self.rgb_seq, TEST_sample_freq)
        bvp = self.ba.extract_bvp_CHROM()
        self._test_plot_bvp(bvp, 'BVP CHROM')
        self.ba = BenchmarkAlgorithms(
            self.rgb_seq, TEST_sample_freq, apply_cdf=True)
        bvp = self.ba.extract_bvp_CHROM()
        self._test_plot_bvp(bvp, 'BVP CHROM with CDF')

    def test_extract_bvp_POS(self):
        self.ba = BenchmarkAlgorithms(self.rgb_seq, TEST_sample_freq)
        bvp = self.ba.extract_bvp_POS()
        self._test_plot_bvp(bvp, 'BVP POS')
        self.ba = BenchmarkAlgorithms(
            self.rgb_seq, TEST_sample_freq, apply_cdf=True)
        bvp = self.ba.extract_bvp_POS()
        self._test_plot_bvp(bvp, 'BVP POS with CDF')

    def test_extract_bvp_O3C(self):
        self.ba = BenchmarkAlgorithms(self.rgb_seq, TEST_sample_freq)
        bvp = self.ba.extract_bvp_O3C()
        self._test_plot_bvp(bvp, 'BVP O3C')
        self.ba = BenchmarkAlgorithms(
            self.rgb_seq, TEST_sample_freq, apply_cdf=True)
        bvp = self.ba.extract_bvp_O3C()
        self._test_plot_bvp(bvp, 'BVP O3C with CDF')

    def test_extract_all(self):
        self.ba = BenchmarkAlgorithms(self.rgb_seq, TEST_sample_freq)
        self.all = self.ba.extract_all()

    def _test_plot(self, title):
        colors = ['r', 'g', 'b']
        plt.figure()
        plt.title(title+' with subracted mean')
        plt.gca().set_prop_cycle('color', colors)
        plt.plot((self.ba.rgb_seq - np.mean(self.ba.rgb_seq, axis=0)))

    def _test_plot_bvp(self, bvp, title):
        plt.figure()
        plt.title(title)
        plt.plot(bvp)


if __name__ == '__main__':

    testBenchmarkAlgorithms = TestBenchmarkAlgorithms()
    # test preprocessing functions
    testBenchmarkAlgorithms.test_filter_color_distortions()
    testBenchmarkAlgorithms.test_normalization()
    # test bvp extraction algorithms
    testBenchmarkAlgorithms.test_extract_bvp_CHROM()
    testBenchmarkAlgorithms.test_extract_bvp_POS()
    testBenchmarkAlgorithms.test_extract_bvp_O3C()
    testBenchmarkAlgorithms.test_extract_all()

# %%
