# Package description

This package implements various algorithms for the extraction of the blood volume pulse (BVP) or imaging photoplethysmogram (iPPG) from video recordings of perfused skin. In general, these represent approaches having less requirements regarding processing time and performance. We also implemented several metrics for the analysis of the extraction quality. See the file description below for further information.

## How to use

- Use the _BenchmarkAlgorithms_ class from _benchmarkAlgorithms_ to extract the BVP with the implemented benchmark algorithms from a precomputed rgb sequence
- Use the _ProcessExtraction_ class from _processExtraction_ to extract the window based heart rates from the BVP
- Use the _BenchmarkMetrics_ class from _benchmarkMetrics_ to compute the metrics for a reference heart rate sequence and the corresponding estimated heart rate sequence
- Use the _BenchmarkSettings_ class from _benchmarkSettings_ if you want to adjust or add any setting

## Implemented algorithms in _benchmarkAlgorithms_:

- _CHROM_: Haan, Gerard de; Jeanne, Vincent (2013): Robust pulse rate from chrominance-based rPPG. In: IEEE transactions on bio-medical engineering 60 (10), S. 2878–2886. DOI: 10.1109/TBME.2013.2266196.
- _POS_: Wang, Wenjin; Den Brinker, Albertus C.; Stuijk, Sander; Haan, Gerard de (2017): Algorithmic Principles of Remote PPG. In: IEEE transactions on bio-medical engineering 64 (7), S. 1479–1491. DOI: 10.1109/TBME.2016.2609282.
- _03C_: Ernst, Hannes; Scherpf, Matthieu; Malberg, Hagen; Schmidt, Martin (2021): Optimal color channel combination across skin tones for remote heart rate measurement in camera-based photoplethysmography. In: Biomedical Signal Processing and Control 68, S. 102644. DOI: 10.1016/j.bspc.2021.102644.
- _GREEN_: Simply evaluating the green color channel from the RGB video recording

## Implemented metrics in _benchmarkMetrics_:

The reference heart rate is usually extracted from a ECG recording or some pulse oxymeter. The estimated heart rate is extracted from the video recording.

- _mean absolute error (**mae**)_: The mean absolute error between a reference and an estimated heart rate in beats per minute
- _root mean square error (**rmse**)_: The root of the mean of squared errors between a reference and an estimated heart rate in beats per minute
- _pearson correlation coefficient (**pearsonr**)_: The pearson correlation coefficient between a reference and an estimated heart rate with no unit and ranging between -1 and 1
- _accuracy based on the IEC 60601-2-27 (**acc**)_: Following IEC 60601-2-27 (originally for ECG heart rates), a pulse rate was deemed erroneous if the absolute difference between the remotely acquired pulse rate and ground truth pulse rate exceeds the greater of either 5BPM or 10% of the ground truth pulse rate.
- _accuracy measure using the area under curve (**acc_AUC**)_: Area under curve for errors between [0, 10] bpm vs. corresponding accuracy corresponding to: Wang, Wenjin; Den Brinker, Albertus C.; Stuijk, Sander; Haan, Gerard de (2017): Color-Distortion Filtering for Remote Photoplethysmography. In: 12th IEEE International Conference on Automatic Face and Gesture Recognition - FG 2017. pp. 71–78. DOI: 10.1109/FG.2017.18.
- _accuracy measure using the area under curve with relative errors (**acc_AUC_perc**)_: Area under curve for errors between [0, 10] percent of reference heart rate vs. corresponding accuracy inspired by: Wang, Wenjin; Den Brinker, Albertus C.; Stuijk, Sander; Haan, Gerard de (2017): Color-Distortion Filtering for Remote Photoplethysmography. In: 12th IEEE International Conference on Automatic Face and Gesture Recognition - FG 2017. pp. 71–78. DOI: 10.1109/FG.2017.18.

## Implemented settings for the application of algorithms and the computation of metrics in _benchmarkSettings_:

See the code directly. The settings are well documented in the code.
