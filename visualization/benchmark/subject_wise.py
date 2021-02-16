"""
-------------------------------------------------------------------------------
Created: 15.02.2021, 09:53
-------------------------------------------------------------------------------
Author: Matthieu Scherpf
Email: Matthieu.Scherpf@tu-dresden.de
Website: https://becuriouss.github.io/matthieuscherpf/
Project page: tba
-------------------------------------------------------------------------------
Purpose: Visualize the benchmark results subject wise.
-------------------------------------------------------------------------------
"""
# %%
import csv
import pickle
from scipy.io import loadmat, savemat
import numpy as np
import os
from pathlib import Path
import traceback
import progressbar
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings

from benchmark import settings


class PlotterUBFC:

    def __init__(self,
                 df_dir,
                 algorithm_names,
                 prefix,
                 sort_by_col='POS_snr_vs_ref'):
        self.df = pd.read_pickle(df_dir).transpose()
        self.plot_handles = []
        self.summary = []
        self.save_fig_path = 'reports/figures/benchmark/subjects'
        self.save_tbl_path = 'reports/tables/benchmark/subjects'
        Path(self.save_fig_path).mkdir(parents=True, exist_ok=True)
        Path(self.save_tbl_path).mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.algorithm_names = algorithm_names
        self.sort_by_col = sort_by_col
        # was color distortion used for the benchmark?
        self.suffix = ''
        if settings.APPLY_COLOR_DISTORTION_FILTERING():
            self.suffix += '_cdf'
        # process
        self._prepare_dataframe()
        self._concat_same_subjects()
        self._create_plots()

    def _prepare_dataframe(self):
        cols = self.df.columns
        for c in cols:
            if 'meta' in c:
                self.df = self.df.drop(c, axis=1)

    def _concat_same_subjects(self):
        # only one recording for each subject
        pass

    def _create_plots(self):
        # apply median filter to get one value for snr columns
        self.df['sort_col'] = self.df[self.sort_by_col].apply(np.median)
        # sort by median snr
        self.df = self.df.sort_values('sort_col')
        # create color map for boxes
        def h1(a): return np.median(np.reshape(a, (-1, 3)), axis=0)
        self.df['skin_color'] = self.df['skin_color'].apply(h1)
        color_map = {}
        for i, sn in enumerate(self.df.index.values):
            color_map[i] = self.df.loc[sn, 'skin_color']/255
        # helper
        def h2(a): return [a]
        # create plots
        for alg in self.algorithm_names:
            # snr plot
            fig, axs = plt.subplots(
                1, 1, figsize=FIGSIZE, dpi=DPI, tight_layout=True)
            sns.boxplot(data=self.df[alg+'_snr'].map(h2), ax=axs,
                        saturation=1, palette=color_map)
            axs.set_ylabel('dB')
            _ = axs.set_title('Signal to noise ratio (SNR) - '+alg)
            axs.axhline(0, linestyle='--', color='red')
            self.plot_handles.append([axs, fig, alg+'_snr'])
            # snr vs reference plot
            fig, axs = plt.subplots(
                1, 1, figsize=FIGSIZE, dpi=DPI, tight_layout=True)
            sns.boxplot(data=self.df[alg+'_snr_vs_ref'].map(h2), ax=axs,
                        saturation=1, palette=color_map)
            axs.set_ylabel('dB')
            _ = axs.set_title('Signal to noise ratio (SNR) - '+alg)
            axs.axhline(0, linestyle='--', color='red')
            self.plot_handles.append([axs, fig, alg+'_snr_vs_ref'])
            # mae plot
            fig, axs = plt.subplots(
                1, 1, figsize=FIGSIZE, dpi=DPI, tight_layout=True)
            sns.boxplot(data=self.df[alg+'_mae'].map(h2), ax=axs,
                        saturation=1, palette=color_map)
            axs.set_ylabel('Beats per minute')
            _ = axs.set_title('Mean absolute error (MAE) - '+alg)
            self.plot_handles.append([axs, fig, alg+'_mae'])
        # save
        for ph in self.plot_handles:
            ph[0].grid(True)
            ph[0].set_xticklabels(self.df.index.values)
        self._save_figures()

    def _save_figures(self):
        for ph in self.plot_handles:
            plt.figure(ph[1].number)
            plt.savefig(os.path.join(self.save_fig_path,
                                     self.prefix + '_' + ph[2] + self.suffix + '.svg'))


class PlotterBP4D(PlotterUBFC):

    def _concat_same_subjects(self):
        recordings = self.df.index
        subjects = []
        for rec in recordings:
            subjects.append(rec.split('_')[0])
        subjects = set(subjects)
        new_index = []
        for rec in recordings:
            for subj in subjects:
                if subj in rec:
                    new_index.append(subj)
        self.df['subj_name'] = new_index
        self.df = self.df.groupby('subj_name', as_index=True).agg(list)
        self.df = self.df.applymap(np.hstack)


if __name__ == '__main__':
    # ----------------------------------------------------------
    # plot settings
    # ----------------------------------------------------------
    plt.rc('axes', axisbelow=True)
    plt.style.use('seaborn-dark')
    sns.set_context("paper")
    FIGSIZE = (48, 8)
    DPI = 256

    algorithm_names = [
        'CHROM',
        'POS',
        'O3C',
    ]

    # ----------------------------------------------------------
    # process
    # ----------------------------------------------------------
    df_dir_ubfc = '/media/fast_storage/matthieu_scherpf/2018_12_UBFC_Dataset/processing/sensors_2021_ms/evaluation/eval_dataframe.p'
    df_dir_bp4d = '/media/fast_storage/matthieu_scherpf/2019_06_26_BP4D+_v0.2/processing/sensors_2021_ms/evaluation/eval_dataframe.p'

    PlotterUBFC(df_dir_ubfc, 'UBFC', algorithm_names)
    PlotterBP4D(df_dir_bp4d, 'BP4D', algorithm_names)
