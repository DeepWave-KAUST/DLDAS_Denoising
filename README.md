![LOGO](https://github.com/DeepWave-Kaust/DAS-Denoising-dev/blob/main/asset/Das_Denoising_Arch.png)

Reproducible material for Distributed Acoustic Sensing Data Denoising Using a Guided Unsupervised Deep Learning Network- **Omar M. Saad, Matteo Ravasi, Tariq Alkhalifah**

[Click here](https://kaust.sharepoint.com/:f:/r/sites/M365_Deepwave_Documents/Shared%20Documents/Restricted%20Area/REPORTS/DW0033?csf=1&web=1&e=L65d1x) to access the Project Report. Authentication to the _Restricted Area_ filespace is required.

# Project structure
This repository is organized as follows:


* :open_file_folder: **asset**: folder containing logo;
* :open_file_folder: **data**: folder containing data;
* :open_file_folder: **notebooks**: set of jupyter notebooks reproducing the experiments in the paper;
* :open_file_folder: **Matlab_CWT_Version**: includes a more stable version for the 2D CWT using Matlab;
* :open_file_folder: **outputs**: includes the denoised data obtained by the proposed framework.


## Getting started :space_invader: :robot:
To ensure reproducibility of the results, we suggest using the `DASDL.yml` file when creating an environment.

Simply run:
```
./install_env.sh
```
It will take some time, if at the end you see the word `Done!` on your terminal you are ready to go. 

Remember to always activate the environment by typing:
```
conda activate DASDL
```
## Scripts (Fully based on Python)
Go to folder `notebooks` and

run the file named "DASDL_Main"

After running, go to folder `outputs` in the root_path, and you will find the denoised data obtained by the proposed framework.

## Scripts (CWT based on Matlab)
This is an alternative way to run the code using a more stable 2D CWT version using Matlab. **It provides less signal leakage compared to the 2D CWT version.**
Go to folder `Matlab_CWT_Version` and

1- run the file named "Prepare_CWT.m", it will obtain the Band-pass filter data and the CWT scale.

2- run the Python file named "DASDL_Main"

After running, go to folder `outputs` in the root_path, and you will find the denoised data obtained by the proposed framework.


**Disclaimer:** All experiments have been carried on a Intel(R) Xeon(R) CPU @ 2.10GHz equipped with a single NVIDIA GEForce RTX 3090 GPU. Different environment 
configurations may be required for different combinations of workstation and GPU.

## Cite us 
```bibtex
@article{saad2024noise,
  title={Noise Attenuation in Distributed Acoustic Sensing Data Using a Guided Unsupervised Deep Learning Network},
  author={Saad, Omar M and Ravasi, Matteo and Alkhalifah, Tariq},
  journal={Geophysics},
  volume={89},
  number={6},
  pages={1--62},
  year={2024},
  publisher={Society of Exploration Geophysicists}
}
