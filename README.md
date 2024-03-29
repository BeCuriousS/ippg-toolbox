[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

# ippgtoolbox: What is it

A toolbox providing functionality for extracting and processing of the blood volume pulse (BVP), also called imaging photoplethysmogram (iPPG) signals. If you use this toolbox in your work please cite the following publication:

- tba

## What is implemented

The toolbox consists of 4 packages:

- _benchmark_: Implementation of standard algorithms for BVP/iPPG extraction and standard metrics for extraction quality analysis.
- _detection_: Implementation of different approaches for the detection of a region of interest and skin segmentation.
- _processing_: Implementation of several functions for BVP/iPPG signal processing.
- _utils_: Implementation of several readers for reference signal data and video recording formats.

## How to use

- Clone repository.
- Change directory to the repository root. Install with:

```shell´´´
    pip install ippgtoolbox
```

There are more detailed information on how to use the code in each package.
