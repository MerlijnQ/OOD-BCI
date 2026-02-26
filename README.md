# The Challenge of Out-Of-Distribution Detection in Motor Imagery BCIs (Under Review)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE) [![Pytorch 2.6.0](https://img.shields.io/badge/pytorch-2.6.0-blue.svg)](https://pytorch.org/) [![MOABB 1.2.0](https://img.shields.io/badge/MOABB-1.2.0-green.svg)](https://moabb.neurotechx.com/)

[![Deep Ensemble](https://img.shields.io/badge/Deep%20Ensemble-009E73)](https://proceedings.neurips.cc/paper_files/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf) [![MC Dropout](https://img.shields.io/badge/MC%20Dropout-009E73)](https://proceedings.mlr.press/v48/gal16.html) [![Energy Score](https://img.shields.io/badge/Energy%20Score-E69F00)](https://proceedings.neurips.cc/paper_files/paper/2020/file/f5496252609c43eb8a3d147ab9b9c006-Paper.pdf) [![DDU](https://img.shields.io/badge/DDU-E69F00)](https://ieeexplore.ieee.org/document/10204383) [![KNN](https://img.shields.io/badge/KNN-0072B2)](https://proceedings.mlr.press/v162/sun22d.html) [![DUQ](https://img.shields.io/badge/DUQ-0072B2)](https://proceedings.mlr.press/v119/van-amersfoort20a.html) ![Softmax](https://img.shields.io/badge/Softmax-8C8C8C) [![ReAct](https://img.shields.io/badge/ReAct-red)](https://proceedings.neurips.cc/paper_files/paper/2021/file/01894d6f048493d2cacde3c579c315a3-Paper.pdf)

This repository contains the official implementation for the paper: [*The Challenge of Out-Of-Distribution Detection in Motor Imagery BCIs*](placeholder).
If the code or the paper has been useful in your research, please add a citation to our work:

```
@article{mulder2026,
  title={The Challenge of Out-Of-Distribution Detection in Motor Imagery BCIs},
  author={Mulder, Merlijn Quincent and Valdenegro-Toro, Matias and Sburlea, Andreea Ioana and De Jong, Ivo Pascal},
  journal={PLACEHOLDER},
  doi={PLACEHOLDER},
  year={2026}
}
```

## Dependencies

This project depends on multiple dependencies such as PyTorch and MOABB.
We provide both a pip and a conda environment:
```
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```
```
conda env create -f environment.yml
conda activate ood-bci
```



## Repository Structure

The project is organized into four main modules:

* **`utils/`**: Experiment logic, including the `Experiments` class implementing *LOCO* and `data processing`.
* **`model/`**: Defines the `EEGNeX architecture` in PyTorch and includes a model `trainer`. 
* **`OOD/`**: Implements the `Pipeline` to get the certainty scores used in the experiment, and contains the implementation of the `UQ methods`.
* **`hyperparameters/`**: Contains a single file that manages the `hyperparameters` for KNN and DUQ used in the experiment.
* **`plotting/`**: The plotting logic used to visualize the results as seen in the paper.

## Running the experiment

To run the LOCO experiment, execute the file main.py whilst providing the following (optional) arguments:

| Argument          | Type        | Default              | Description                                                      |
|-------------------|-------------|----------------------|------------------------------------------------------------------|
| `dataset`         | Positional  | `Schirrmeister2017`  | Name of the dataset to use which is available through MOABB.                                      |
| `--subject`       | int         | `1`                  | Specific subject to perform the experiment on                    |
| `--subject-range` | START END   | `None`               | Run a range of subjects (inclusive). Example: `1 10`.            |
| `--tune`          | flag        | `False`              | Enable hyperparameter tuning.                                    |
| `--react`         | flag        | `False`              | Enable ReAct logic.                                              |
| `--inverse`       | flag        | `False`              | Run the experiment two ID classes and selectively inverting certainty scores based on another OOD class.                                                                                                 |

### Organizing results
After running experiments on multiple subjects, results are stored **per subject** in individual JSON files. These per-subject outputs can be **aggregated into a single results file** using the file `group_data.py` in **`utils/`**. The following arguments can be provided:


| Argument    | Type | Default             | Description                                  |
| ----------- | ---- | ------------------- | -------------------------------------------- |
| `--dataset` | str  | `Schirrmeister2017` | Name of the dataset used in the experiment   |
| `--n`       | int  | `14`                | Number of subjects whose results are grouped |
| `--inverse` | flag | `False`             | Group inversion experiment results           |
| `--react`   | flag | `False`             | Group ReAct experiment results               |

The plotting functions expect that a path to the results file is provided. The allowed arguments are specified in the function `main()` of the corresponding files.

## Implemented Methods
<a id="1">[1]</a> **Deep Ensemble.** Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. *Simple and Scalable Predictive Uncertainty Estimation Using Deep Ensembles.* In: Advances in Neural Information Processing Systems (NeurIPS), vol. 30, 2017. URL: https://proceedings.neurips.cc/paper_files/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf

<a id="2">[2]</a> **MC Dropout.** Daily Milanés-Hermosilla et al. *Monte Carlo Dropout for Uncertainty Estimation and Motor Imagery Classification.* In: Sensors, 21(21), 2021. DOI: https://doi.org/10.3390/s21217241

<a id="3">[3]</a> **Energy Score.** Weitang Liu et al. *Energy-Based Out-of-Distribution Detection.* In: Advances in Neural Information Processing Systems (NeurIPS), vol. 33, pp. 21464–21475, 2020. URL: https://proceedings.neurips.cc/paper_files/paper/2020/file/f5496252609c43eb8a3d147ab9b9c006-Paper.pdf

<a id="4">[4]</a> **DDU.** Jishnu Mukhoti et al. *Deep Deterministic Uncertainty: A New Simple Baseline.* In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 24384–24394, 2023. DOI: https://doi.org/10.1109/CVPR52729.2023.02336

<a id="5">[5]</a> **KNN.** Yiyou Sun et al. *Out-of-Distribution Detection with Deep Nearest Neighbors.* In: Proceedings of the 39th International Conference on Machine Learning (ICML), vol. 162, pp. 20827–20840, 2022. URL: https://proceedings.mlr.press/v162/sun22d.html

<a id="6">[6]</a> **DUQ.** Joost van Amersfoort et al. *Uncertainty Estimation Using a Single Deep Deterministic Neural Network.* In: Proceedings of the 37th International Conference on Machine Learning (ICML), vol. 119, pp. 9690–9700, 2020. URL: https://proceedings.mlr.press/v119/van-amersfoort20a.html

<a id="7">[7]</a> **ReAct.** Yiyou Sun, Chuan Guo, and Yixuan Li. *ReAct: Out-of-Distribution Detection with Rectified Activations.* In: Advances in Neural Information Processing Systems (NeurIPS), vol. 34, pp. 144–157, 2021. URL: https://proceedings.neurips.cc/paper_files/paper/2021/file/01894d6f048493d2cacde3c579c315a3-Paper.pdf

### Model & Data
<a id="8">[8]</a> **EEGNeX.** Xia Chen et al. *Toward Reliable Signals Decoding for Electroencephalogram: A Benchmark Study of EEGNeX.* In: Biomedical Signal Processing and Control, vol. 87, p. 105475, 2024. DOI: https://doi.org/10.1016/j.bspc.2023.105475

<a id="9">[9]</a> **MOABB.** Bruno Aristimunha et al. *Mother of All BCI Benchmarks (MOABB)*, Version 1.0.0. 2023. DOI: https://doi.org/10.5281/zenodo.10034223