# README

MultiSeqGO, predicts protein functions by using only protein sequence information to generate multi-modal features.

## Install dependencies

#### 1. Install RaptorX

Download RaptorX package by "git clone https://github.com/j3xugit/RaptorX-3DModeling.git" and save it anywhere in your own account, e.g., $HOME/RaptorX-3DModeling/. It contains the following files and subfolders (and a few others):

BuildFeatures/

DL4DistancePrediction4/

DL4PropertyPrediction/

Folding/

params/

raptorx-path.sh raptorx-external.sh

README.md

Server/

This RaptorX package consists of 4 major modules: BuildFeatures/ for generating multiple sequence alignment (MSAs) and input features for angle/contact/distance/orientation prediction, DL4DistancePrediction4/ for contact/distance/orientation prediction, DL4PropertyPrediction/ for local structure property prediction, and Folding/ for building 3D models.

To predict contact/distance/orientation and fold a protein, you may simply run RaptorX-3DModeling/Server/RaptorXFolder.sh, but before doing this some external packages and databases shall be installed and configured.

##### Required modules

1.anaconda or miniconda for Python 2.7

conda install numpy 

Install msgpack-python by "conda install -c anaconda msgpack-python"; it may not work if you intall it through pip.

2.Biopython

pip install biopython==1.76. Note that a newer version may not work with Python 2.7.

3.pygpu and Theano 1.0

conda install numpy scipy mkl and then conda install theano pygpu

Please make sure that the CUDA toolkits and CUDNN library have been installed on your machine with GPUs. Set the environment variable CUDA_ROOT to where cuda is installed, e.g., export CUDA_ROOT=/usr/local/cuda. Make sure that the header and lib64 files of CUDNN are in CUDA_ROOT/include and CUDA_ROOT/lib64, respectively. We have tested Theano 1.04, CUDA 8 to 10.1 and CUDNN 7 to 7.6.5 . Other versions of CUDA and CUDNN may also work.

4.Install HHblits

To compile from source, you will need a recent C/C++ compiler (at least GCC 4.8 or Clang 3.6) and [CMake](http://cmake.org/) 2.8.12 or later.

To download the source code and compile the HH-suite execute the following commands:

git clone https://github.com/soedinglab/hh-suite.git
mkdir -p hh-suite/build && cd hh-suite/build
cmake -DCMAKE_INSTALL_PREFIX=. ..
make -j 4 && make install
export PATH="$(pwd)/bin:$(pwd)/scripts:$PATH"

we download the dataset here（[Index of /~compbiol/uniclust/2023_02](https://wwwuser.gwdg.de/~compbiol/uniclust/2023_02/)）(66GB)

 5.Install deep learning models for contact/distance/orientation/angle/SS/ACC prediction

The deep learning model files for contact/distance/orientation prediction are big (each 100-200M). You may download them at https://doi.org/10.5281/zenodo.4710337 or http://raptorx.uchicago.edu/download/ .

1. The package RXDeepModels4DistOri-FM.tar.gz has 6 models for contact/distance/orientation/ prediction. Unpack it and place all the deep model files (ending with .pkl) at $DL4DistancePredHome/models/

2. The package RXDeepModels4Property.tar.gz has 7 deep models for Phi/Psi angle, Secondary Structure (SS) and Solvent Accessibility (ACC) prediction. Unpack it and place all the deep model files (ending with .pkl) at $DL4PropertyPredHome/models/ . By default the package will just predict Phi/Psi angles. If you also want to predict SS and ACC, please use "-m AllSeqSet10820Models" in running the script programs in DL4PropertyPrediction/Scripts/
   
   

   RaptorX detailed usage can be found here(https://github.com/j3xugit/RaptorX-3DModeling/tree/master).



#### 

#### 2.Install ESM2(esm2_t48_15B_UR50D 15B parameters)

As a prerequisite, you must have PyTorch installed to use this repository.
You can use this one-liner for installation, using the latest release of esm:
pip install fair-esm  # latest release, OR:
pip install git+https://github.com/facebookresearch/esm.git  # bleeding edge, current repo main branch
After pip install, you can load and use a pretrained model as follows:
       import torch
       import esm
       # Load ESM-2 model
       model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
       batch_converter = alphabet.get_batch_converter()
       model.eval()  # disables dropout for deterministic results

       # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
       data = [
           ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
           ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
           ("protein2 with mask","KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
           ("protein3",  "K A <mask> I S Q"),
       ]
       batch_labels, batch_strs, batch_tokens = batch_converter(data)
       batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

       # Extract per-residue representations (on CPU)
       with torch.no_grad():
           results = model(batch_tokens, repr_layers=[48], return_contacts=True)
       token_representations = results["representations"][48]

       # Generate per-sequence representations via averaging
       # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
       sequence_representations = []
       for i, tokens_len in enumerate(batch_lens):
           sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))

       # Look at the unsupervised self-attention map contact predictions
       import matplotlib.pyplot as plt
       for (_, seq), tokens_len, attention_contacts in zip(data, batch_lens, results["contacts"]):
           plt.matshow(attention_contacts[: tokens_len, : tokens_len])
           plt.title(seq)
           plt.show()

#### 3.Dependencies

The code was developed and tested using python 3.7.

Install python dependencies: `pip install -r requirements.txt`



# 
