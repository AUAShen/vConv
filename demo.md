# A motif discovery demo of vConv-based model

Here we provide a demo of vConv's application in motif discovery using chipseq pick data as input. The input file is in fasta format, each reads is a pick sequence identified from chipseq experiment. The demo will train the model and output model's parameters and the predicted motifs.


# Folder structure:


**../demofasta/**  input folder, saves fasta files. The demo script will first build a vConv-based model. For each fasta file under this folder, it will generate input data and train the model on the dataset. Finally a set of motifs will be generated for each input fasta file. 


**../result/vConvB/** output folder. For each input fasta file, the script will generate a subfolder under this directory, under which predicted motifs will be saved in **recover_PWM** folder and model's parameters will be saved in **ModleParaMeter**


## Prerequisites

### Software

- Python 2 and its packages:
  - numpy
  - h5py
  - pandas
  - seaborn
  - scipy
  - keras (version 2.2.4)
  - tensorflow (version 1.3.0)
  - sklearn

Alternatively, if you want to guarantee working versions of each dependency, you can install via a fully pre-specified environment.
```{bash}
conda env create -f environment_vConv.yml
```

# Q
