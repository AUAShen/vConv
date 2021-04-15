# A motif discovery demo of vConv-based model

vConv is a novel convolutional layer, which can replace the classic conv layer. Here we provide a demo of vConv's application in motif discovery using chipseq peak data as input. The input file is in fasta format, each reads is a peak sequence identified from chipseq experiment. The demo will train the model and output model's parameters and the extract motifs.


# Folder structure:


**../demofasta/**  input folder, saves fasta files. 

**../result/vConvB/** output folder. For each input fasta file, the script will generate a subfolder under this directory, under which predicted motifs will be saved in **recover_PWM** folder and model's parameters will be saved in **ModelParameter**. For example, if the fasta file is "XXX.fasta". The script will save PWMs to **../result/vConvB/XXX/recover_PWM** and save the model parameters to **../result/vConvB/XXX/ModelParameter**


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
  - ushuffle

Alternatively, if you want to guarantee working versions of each dependency, you can install them via a fully pre-specified environment.
```{bash}
conda env create -f environment_vConv.yml
```

# Overview of the pipeline

Run the following command under the **./vConvbaseddiscovery/code/** folder, a demo script of vConv's application in motif discovery will be executed
```{bash}
 python VConvMotifdiscovery.py
```
This demo shows one of the real-world applications of the vConv layer: a single-layered Convolutional Neural network for motif discovery from chipseq data. Detailed workflow is explained below. 


## Generating training and testing dataset from fasta file
```{python}
class GeneRateOneHotMatrix() # in seq_to_matrix.py
```
The input files are in fasta format. Each sequence is collected from a chipseq peak [reference to the data source]. The first step is to generate "negative" samples by shuffling the chipseq reads while keeping the dimer frequency. 
```{python}
def k_mer_shuffle(self,seq_shape, seq_series, k=2) # in class: GeneRateOneHotMatrix
```
Then the reads are one-hot represented in to a 4*L matrix, where L is the length of a read. Finally, both "positive" and "negative" samples are mixed together and divided into "training set" and "test set".  
```{python}
# in class: GeneRateOneHotMatrix
def seq_to_matrix(self,seq, seq_matrix, seq_order)
def GeneRateTrain(self, allData, ValNum=10, RandomSeeds=233)
```
## Build vConv-based neural network

A vConv-based model is builded in a similar way as illustrated in [README.md](https://github.com/AUAShen/vConv/blob/main/README.md). Shannon loss is highly suggested to add into the final loss function, in order to fully use vConv layer's function. 
```{python}
# in build_models.py
def build_vCNN(model_template, number_of_kernel, max_kernel_length, k_pool=1,input_shape=(1000,4))
# add Shanon loss
lossFunction = ShanoyLoss(KernelWeights, MaskWeight, mu=mu)
model.compile(loss=lossFunction, optimizer=sgd, metrics=['accuracy'])
# Shanon loss is defined in vConv_core.py
def ShanoyLoss(KernelWeights, MaskWeight, mu)
```
## Train the model

The model is trained similarly to the normal CNN model. Detailed training strategy refers to the [supplementary material](add the link!)

```{python}
def train_vCNN(input_shape,modelsave_output_prefix,data_set, number_of_kernel, max_ker_len,init_ker_len_dict,random_seed, batch_size, epoch_scheme)
```

## Motif visualisation

After the training process, motifs (in PWM format) are recovered from kernels. In brief, the highest-scored subsequences (from positive sequences) are selected for each kernel to generate a PWM, by normalizing each position's nucleotide composition to 1.

```{python}
############Select Kernel #################### section in VConvMDcore.py
# get highest subsequences for each kernel from positive sequences
KernelSeqs, KSconvValue, seqinfo = KernelSeqDive(kernel, GeneRateOneHotMatrixTest.seq_pos_matrix_out,)
# normalize to get PWM
PwmWork = NormPwm(KernelSeqs, True)
```

# General applications

The demo code here can be applied to a variety of motif discovery problems. At least two additional applications are possible: (1) When "biology-meaningful" negative data is available, one can skip the dimer shuffling and use the "biology-meaningful" negative data instead. It is worth noticing that in this situation, the motif extraction process may be different. Because negative data may also contain motifs, the highest-scored subsequences from negative sequences should be considered. (2) Generalization sequential motif identification. For example, given any set of sequences of interest, identify the common motifs. These motifs are shared, conserved sequence patterns among the input reads. In this scenario, the one-hot encoding of the sequence will no longer be **4*L**, but **N*L**. **N** is the alphabet size. In RNA, **N** is 4 while in protein **N** is 20.

Although the demo model only has one vConv layer, the vConv layer can be adapted into more sophisticated model structures. In other word, vConv is a generalized convolution layer, which can be applied to a variety of model structures. In our manuscript [reference to add], we presented examples of multi-layers vConv based neural network.  





#
