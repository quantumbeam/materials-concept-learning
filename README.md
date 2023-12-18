# Self-supervised learning of materials concepts from crystal structures via deep neural networks

This is the offitial imprementation of [Self-supervised learning of materials concepts from crystal structures via deep neural networks](https://iopscience.iop.org/article/10.1088/2632-2153/aca23d)

![materials map demo](files_for_web/overview.png)



## Table of Contents
1. PyTorch implementation of proposed method
  - Please see `src/` directory.
2.  Interactive materials map visualisation  
![materials map demo](https://raw.githubusercontent.com/quantumbeam/materials-concept-learning/main/files_for_web/altair_dashboard.gif)
2. Resulting embeddings with a Jupyter notebook for local neighbourhood analysis
3. Local neighbourhood search results
4. List of the target materials from the Materials Project
5. Citation

___
## 1. Interactive materials map visualisation   
- Corresponding to the results of the "Global distribution analysis" section in the main text.
- You can interactively explore our materials map on your web browser.

Click to open:  
➡️ [__Full map__ (pretty heavy: ~20MB)](https://raw.githack.com/quantumbeam/materials-concept-learning/main/files_for_web/embedding_cry_cgcnn.html)

➡️ [__Reduced map__ (randomly sampled to 20%)](https://raw.githack.com/quantumbeam/materials-concept-learning/main/files_for_web/embedding_cry_cgcnn_20per.html)

## 2. Jupyter notebook examples

### Mapping your CIF data with pre-trained model
- Example of exporting embeddings with your data (.CIF files) and ploting in our materials map.

- `examples/export_embedding_from_CIF.ipynb`:  
  ➡️ [__Open in Google Colab__](https://colab.research.google.com/github/quantumbeam/materials-concept-learning/blob/main/examples/export_embedding_from_CIF.ipynb)


### local neighbourhood analysis
  - Corresponding to the results of the "Local neighbourhood analysis" section in the main text and Appendix A in the Supplementary Information (SI).
  - You can analyse our embeddings through the local neighbour search on your web browser.

  - `examples/neighbour_analysis_of_MP.ipynb`:  
  ➡️ [__Open in Google Colab__](https://colab.research.google.com/github/quantumbeam/materials-concept-learning/blob/main/examples/neighbour_analysis_of_MP.ipynb)


## 3. Local neighbourhood search results
- Corresponding to the results of the "Local neighbourhood analysis" section in the main text and Appendix A in the SI.
- You can see the extended lists of the top-1000 neighbourhoods for Tables and SI Tables shown below.

| Table                                          | Method                                      | 
| :--------------------------------------------: | :-----------------------------------------: | 
| Table 1 (Hg-1223)                | [__Ours__](https://github.com/quantumbeam/materials-concept-learning/blob/main/files_for_web/Our_mp-22601_Hg-1223.csv), [__ESM__](https://github.com/quantumbeam/materials-concept-learning/blob/main/files_for_web/ESM_mp-22601_Hg1223.csv), [__SCM__](https://github.com/quantumbeam/materials-concept-learning/blob/main/files_for_web/SCM_mp-22601_Hg1223.csv) | 
| Table 2 (LiCoO2)                 | [__Ours__](https://github.com/quantumbeam/materials-concept-learning/blob/main/files_for_web/Our_mp-22526_LiCoO2.csv), [__ESM__](https://github.com/quantumbeam/materials-concept-learning/blob/main/files_for_web/ESM_mp-22526_LiCoO2.csv), [__SCM__](https://github.com/quantumbeam/materials-concept-learning/blob/main/files_for_web/SCM_mp-22526_LiCoO2.csv) | 
| SI Table S1 (Cr2Ge2Te6) | [__Ours__](https://github.com/quantumbeam/materials-concept-learning/blob/main/files_for_web/Our_mp-541449_CrGeTe3.csv), [__ESM__](https://github.com/quantumbeam/materials-concept-learning/blob/main/files_for_web/ESM_mp-541449_CrGeTe3.csv), [__SCM__](https://github.com/quantumbeam/materials-concept-learning/blob/main/files_for_web/SCM_mp-541449_CrGeTe3.csv) | 
| SI Table S2 (Sm2Co17)   | [__Ours__](https://github.com/quantumbeam/materials-concept-learning/blob/main/files_for_web/Our_mp-1200096_Sm2Co17.csv), [__ESM__](https://github.com/quantumbeam/materials-concept-learning/blob/main/files_for_web/ESM_mp-1200096_Sm2Co17.csv), [__SCM__](https://github.com/quantumbeam/materials-concept-learning/blob/main/files_for_web/SCM_mp-1200096_Sm2Co17.csv) | 
| SI Table S3 (Hg-1223)   | [__Ours__](https://github.com/quantumbeam/materials-concept-learning/blob/main/files_for_web/Our_mp-22601_Hg-1223.csv), [__Crystal structure encoder__](https://github.com/quantumbeam/materials-concept-learning/blob/main/files_for_web/EPA_cry_mp-22601_Hg-1223.csv), [__XRD encoder__](https://github.com/quantumbeam/materials-concept-learning/blob/main/files_for_web/EPA_XRD_mp-22601_Hg-1223.csv) | 
| SI Table S4 (LiCoO2)    | [__Ours__](https://github.com/quantumbeam/materials-concept-learning/blob/main/files_for_web/Our_mp-22526_LiCoO2.csv), [__Crystal structure encoder__](https://github.com/quantumbeam/materials-concept-learning/blob/main/files_for_web/EPA_cry_mp-22526_LiCoO2.csv), [__XRD encoder__](https://github.com/quantumbeam/materials-concept-learning/blob/main/files_for_web/EPA_XRD_mp-22526_LiCoO2.csv) | 

## 4. List of the target materials from the Materials Project
- The full list of the 122,543 target materials used in this study is available: [__metadata.csv__](https://github.com/quantumbeam/materials-concept-learning/blob/main/files_for_web/embedding_metadata.csv)
- You can collect the materials data using the shown Materials Project IDs via the [Materials Project APIs](https://materialsproject.org/open).

## 5. Citation

```
@article{Suzuki_materials_concepts_learning_2022,
doi = {10.1088/2632-2153/aca23d},
url = {https://dx.doi.org/10.1088/2632-2153/aca23d},
year = {2022},
month = {dec},
publisher = {IOP Publishing},
volume = {3},
number = {4},
pages = {045034},
author = {Yuta Suzuki and Tatsunori Taniai and Kotaro Saito and Yoshitaka Ushiku and Kanta Ono},
title = {Self-supervised learning of materials concepts from crystal structures via deep neural networks},
journal = {Machine Learning: Science and Technology},
}
```
