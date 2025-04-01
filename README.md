# Attention U-Net for WMH Segmentation

This repository contains code related to the inclusion of attention blocks in U-Net models for the segmentation of white matter hyperintensities (WMH) in individuals ranging from early adulthood to late elders.

## Repository Contents

- **run_generate_model_2dseg_attention.py**  
  Responsible for generating the U-Net models with attention blocks.

- **run_test_dataset_2D_versionAttention.py**  
  Responsible for running the generated models on the test set and reconstructing the original images with the predicted segmentation masks.

- **generate_Results.py**  
  Responsible for generating the plots used for this study.

## Data Acquisition

Due to privacy agreements, the dataset cannot be shared online. However, researchers interested in acquiring the data may contact us directly. For information about the datasets, please refer to the following sources:

- [CNS, FAVR I, and FAVR II Datasets](https://cumming.ucalgary.ca/labs/vascular-imaging/events-and-news)
- [SIN, Utrecht, and AMS Datasets](https://wmh.isi.uu.nl/)

Additional information about the dataset or any other details can be found in the attached documentation.

## How to Run the Code

Each code file contains detailed instructions on how to run the experiments. Please refer to the header comments in each script for additional information on setup and execution.

## Citation

If you use this work, please cite it as follows:

*Reference: [To be added once the paper is published]*

---

For further details or inquiries regarding the dataset or code, please feel free to contact the authors.
