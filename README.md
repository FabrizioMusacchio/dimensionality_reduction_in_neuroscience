# Dimensionality reduction in Neuroscience

This repository contains the code for the exercises of the course "[Dimensionality reduction in Neuroscience](https://www.fabriziomusacchio.com/teaching/teaching_dimensionality_reduction_in_neuroscience/)" (author: Fabrizio Musacchio, Oct 2024).

Each exercise is contained in a separate Jupyter notebook and corresponds to a different chapter of the course. The lecture notes of each chapter can be found on this website.

## Download the data
Additionally to the data provided in the data folder of this repository, you need to download additional data from this [Google Drive folder](https://drive.google.com/drive/folders/1WEKgYTkpYqaVs7WCiXzbR1jnkK1Y_nUE?usp=share_link). Place the downloaded data in the cloned version of this repository (into "data" folder).

In case you are running the notebooks on Google Colab, you need to place the data in your Google Drive and mount it in the notebook. [Here](https://www.fabriziomusacchio.com/blog/2023-03-23-google_colab_file_access/) is description of how to do it.

## Environment setup
For reproducibility:

```bash
conda create -n dimredcution python=3.11 mamba -y
conda activate dimredcution
mamba install -y ipykernel matplotlib numpy scipy scikit-learn umap-learn
```

## Acknowledgements
The data used in this course is from the following public sources:

* "**hypothalamus_calcium_imaging_remedios_et_al.mat**": The dataset is from the 2023's course '[data analysis techniques in neuroscience](https://github.com/cheninstitutecaltech/Caltech_DATASAI_Neuroscience_23)' by the Chen Institute for Neuroscience at Caltech, originally from the paper: Remedios, R., Kennedy, A., Zelikowsky, M. et al. Social behaviour shapes hypothalamic neural  ensemble representations of conspecific sex. Nature 550, 388–392 (2017). <https://doi.org/10.1038/nature23885>
* "**macosko_2015.pkl.gz**": Extracted from the the datasets available in the [openTSEN package](https://opentsne.readthedocs.io/en/stable/examples/01_simple_usage/01_simple_usage.html). Specifically, it is the Macosko 2015 mouse retina data set. 
* "**hippocampus_achilles**": Extracted from the datasets available in the [CEBRA package](https://cebra.ai/docs/demo_notebooks/Demo_hippocampus.html).