# 🧬 BEND  - **Ben**chmarking **D**NA Language Models on Biologically Meaningful Tasks

![Stars](https://img.shields.io/github/stars/frederikkemarin/BEND?logo=GitHub&color=yellow)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Documentation Status](https://readthedocs.org/projects/bend/badge/?version=latest)](https://bend.readthedocs.io/en/latest/?badge=latest)

The BEND paper (ICLR 2024) is available here: 

"[BEND: BENCHMARKING DNA LANGUAGE MODELS ON BIOLOGICALLY MEANINGFUL TASKS](https://arxiv.org/abs/2311.12570)"

Frederikke Isa Marin, Felix Teufel, Marc Horlacher, Dennis Madsen, Dennis Pultz, Ole Winther, Wouter Boomsma

## Documentation
[Documentation for the BEND code repository](https://bend.readthedocs.io/en/latest/?badge=latest)

## Data

All data is available for download [here](https://sid.erda.dk/cgi-sid/ls.py?share_id=aNQa0Oz2lY)

The data can be downloaded via a script, see [section 2.5](#2-setup)

## Tutorial

### 1. Data format

The data for each task is stored as a `bed` file. This file includes the genomic coordinates for each sample, as well as its split membership and potentially a label. Together with a reference genome, the file is used to extract the DNA sequences for training. Labels that are too complex to be stored in a column in the text-based `bed` file are stored in a `hdf5` file. The two files share their index, so that sample `i` in the `bed` file matches record `i` in the `hdf5` file.


`bed` is a tab-separated format that can be read like a regular table. All our task files include a column `split`, and optionally `label`. If `label` is missing, the labels are found in the `hdf5` file of the same name.
```
chromosome	start	end     split	label
chr1	    1055037	1055849	train	1
chr3	    1070026	1070436	valid	0
```

### 2. Setup

#### Quick Setup (Recommended - using uv and make)

**Prerequisites:** Install [uv](https://docs.astral.sh/uv/) - a fast Python package manager
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**One-command setup:**
```bash
git clone https://github.com/frederikkemarin/BEND.git
cd BEND
make setup              # Set up environment with Python 3.11
make download-data      # Download dataset via Google Cloud Storage (fast)
source .venv/bin/activate  # Activate environment
```

**Data download options:**
```bash
# Fast download via Google Cloud Storage (recommended)
make download-data      # Requires gsutil (Google Cloud SDK)

# Original download method (fallback)
make download-data-original  # Uses the original download script
```

**Interactive setup:**
```bash
./setup_dev.sh  # Interactive script with setup options
```

**Available make commands:**
```bash
make help               # Show all available commands
make setup              # Basic environment setup
make setup-dev          # Development setup with testing tools
make download-data      # Download dataset (Google Cloud Storage - fast)
make download-data-original  # Download dataset (original method - fallback)
make test               # Run tests
make format             # Format code
make clean              # Clean build artifacts
make status             # Check environment status
```

#### Alternative Setup (Manual)

If you prefer manual setup or don't have `make` available:
1. Clone the BEND repository: `git clone https://github.com/frederikkemarin/BEND.git`
2. Change to the BEND directory: `cd BEND`
3. Install Python 3.11: `uv python install 3.11`
4. Create virtual environment: `uv venv --python 3.11`
5. Install the requirements: `uv pip install -r requirements.txt`
6. Install BEND in development mode: `source .venv/bin/activate && uv pip install -e .`
7. Download the data: 
   - **Fast (recommended)**: `mkdir -p data && gsutil -m cp -r gs://curvebio-mahdibaghbanzadeh/bend/* data/`
   - **Fallback**: `python scripts/download_bend.py`

**Prerequisites for fast download:**
- Install Google Cloud SDK: https://cloud.google.com/sdk/docs/install
- This provides the `gsutil` command for efficient data transfer

**Prerequisites for HyenaDNA models:**
- Install Git LFS (Large File Storage): Required for downloading HyenaDNA model checkpoints
  ```bash
  # On Ubuntu/Debian
  sudo apt-get install git-lfs
  
  # On macOS with Homebrew  
  brew install git-lfs
  
  # On other systems, see: https://git-lfs.github.io/
  ```
- Initialize Git LFS in your repository:
  ```bash
  git lfs install
  ```

**Note:** We recommend Python 3.11 due to compatibility issues with some dependencies in Python 3.12.

### 3. GPU Device Selection

When training models on multi-GPU systems, you may want to specify which GPU to use. Here are some useful tricks for GPU device selection:

#### Option 1: Use CUDA_VISIBLE_DEVICES for single GPU training
This is the cleanest approach - make only a specific GPU visible to the process:
```bash
# Use only GPU 3 (appears as cuda:0 to PyTorch)
CUDA_VISIBLE_DEVICES=3 python scripts/train_on_task.py --config-name gene_finding embedder=dnabert2

# Use only GPU 1
CUDA_VISIBLE_DEVICES=1 python scripts/precompute_embeddings.py --config-name embed task=gene_finding model=hyenadna-medium-160k
```

#### Option 2: Use device_id parameter
You can specify the device ID directly in the command:
```bash
python scripts/train_on_task.py --config-name gene_finding embedder=dnabert2 params.device_id=3
```
**Note:** This may cause issues with DataParallel on multi-GPU systems. Use Option 1 or 3 instead.

#### Option 3: Use CUDA_VISIBLE_DEVICES for multi-GPU training
For training with multiple specific GPUs:
```bash
# Use GPUs 2 and 3 (appear as cuda:0 and cuda:1 to PyTorch)
CUDA_VISIBLE_DEVICES=2,3 python scripts/train_on_task.py --config-name gene_finding embedder=dnabert2

# Use GPUs 0, 1, and 3
CUDA_VISIBLE_DEVICES=0,1,3 python scripts/train_on_task.py --config-name gene_finding embedder=dnabert2
```

#### Why use CUDA_VISIBLE_DEVICES?
- **Avoids device conflicts**: Prevents DataParallel from trying to use all available GPUs
- **Clean resource allocation**: Other processes can't accidentally use your specified GPUs
- **Local effect only**: This is an environment variable that only affects the specific command/process you run it with
  - **No system-wide changes**: Your system GPU configuration remains unchanged
  - **No impact on other terminals**: Other terminal sessions and processes are unaffected
  - **Temporary**: The effect ends when the command completes
  - **Process-specific**: Each command can use different GPU configurations independently
- **Works with all training modes**: Compatible with single-GPU, DataParallel, and DDP training

### 4. Computing embeddings

For training downstream models, it is practical to precompute and save the embeddings to avoid recomputing them at each epoch. As embeddings can grow large when working with genomes, we use [Webdataset](https://github.com/webdataset/webdataset) `tar.gz` files as the format.
Firstly download the desired data from the [data folder](https://sid.erda.dk/cgi-sid/ls.py?share_id=aNQa0Oz2lY&current_dir=data&flags=f) and place it in BEND/ (for ease of use maintain the same folder structure). 
To precompute the embeddings for all models and tasks, run : 
```
python scripts/precompute_embeddings.py 
```
This script automatically calls the hydra config file at ```/../conf/embedding/embed.yaml```. 

By default all embeddings are generated for all tasks. To alter the tasks/model for which to compute the embeddings, please alter the ```tasks``` and/or the ```models``` list in the config file (under ```hydra.sweeper``) or override the behaviour from the commandline in the following manner: 

```
python scripts/precompute_embeddings.py model=resnetlm,awdlstm task=gene_finding,enhancer_annotation
```
Train, validation and test embeddings are saved in chunks of (default) 50,000. To parallelize embeddings generation, you can call `precompute_embeddings.py` as above multiple times, but add additional arguments of the form `chunk=[10,11,12] splits=[train,valid]` to the individual calls in order to only compute specific chunks in a given call. If these arguments are not provided, the command will default to computing all chunks and splits.

#### Embedders overview

If you need to make embeddings for other purposes than preparing downstream task data, [`bend.embedders`](bend/utils/embedders.py) contains wrapper classes around the individual models. Each embedder takes a path (or name, if available on HuggingFace) of a checkpoint as the first argument, and provides an `embed()` method that takes a list of sequences and returns a list of embeddings.   
Embedders have a default-true argument `remove_special_tokens=True` in `embed()` that removes any `[CLS]`, `[SEP]` tokens from the returned embeddings. For models that return less embedding vectors than their number of input nucleotides, [embeddings can be upsampled](#how-are-embeddings-upsampled) to the original input sequence length using the `upsample_embeddings=True` argument in `embed()`.   
Some of the embedders currently also support computing logits and cross entropy losses, see the documentation for more information.

| Embedder | Reference | Models | Info |
| --- | --- | --- | ---|
| [DNABertEmbedder](https://bend.readthedocs.io/en/latest/bend.utils.embedders.html#bend.utils.embedders.DNABertEmbedder) | [Ji et al.](https://academic.oup.com/bioinformatics/article/37/15/2112/6128680) | [4 different k-mer tokenizations available](https://github.com/jerryji1993/DNABERT#32-download-pre-trained-dnabert)  | has an additional argument `kmer=6` to specify the k-mer size.|
|[NucleotideTransformerEmbedder](https://bend.readthedocs.io/en/latest/bend.utils.embedders.html#bend.utils.embedders.NucleotideTransformerEmbedder)| [Dalla-Torre et al.](https://www.biorxiv.org/content/10.1101/2023.01.11.523679v2) | [8 different models available](https://huggingface.co/InstaDeepAI) | |
|[ConvNetEmbedder](https://bend.readthedocs.io/en/latest/bend.utils.embedders.html#bend.utils.embedders.ConvNetEmbedder)| BEND | [1 model available](https://sid.erda.dk/cgi-sid/ls.py?share_id=eXAmVvbRSW&current_dir=pretrained_models&flags=f) | A baseline LM used in BEND.
|[AWDLSTMEmbedder](https://bend.readthedocs.io/en/latest/bend.utils.embedders.html#bend.utils.embedders.AWDLSTMEmbedder)| BEND | [1 model available](https://sid.erda.dk/cgi-sid/ls.py?share_id=eXAmVvbRSW&current_dir=pretrained_models&flags=f) | A baseline LM used in BEND.
|[GPNEmbedder](https://bend.readthedocs.io/en/latest/bend.utils.embedders.html#bend.utils.embedders.GPNEmbedder)| [Benegas et al.](https://www.biorxiv.org/content/10.1101/2022.08.22.504706v2) | Models trained on [*A. thaliana*](https://huggingface.co/songlab/gpn-arabidopsis) and [Brassicales](https://huggingface.co/songlab/gpn-brassicales) available | This LM was not evaluated in BEND as it was not trained on the human genome. |
|[GENALMEmbedder](https://bend.readthedocs.io/en/latest/bend.utils.embedders.html#bend.utils.embedders.GENALMEmbedder) | [Fishman et al.](https://www.biorxiv.org/content/10.1101/2023.06.12.544594v1) |[8 different models available](https://huggingface.co/AIRI-Institute) | |
|[HyenaDNAEmbedder](https://bend.readthedocs.io/en/latest/bend.utils.embedders.html#bend.utils.embedders.HyenaDNAEmbedder) | [Nguyen et al.](https://arxiv.org/abs/2306.15794) | [5 different models available](https://huggingface.co/LongSafari) | **Requires Git LFS to be installed** to automatically download checkpoints. Instead of the HF checkpoint name, the argument when instantiating needs to be of the format `path/to/save/checkpoints/checkpoint_name` |
|[DNABert2Embedder](https://bend.readthedocs.io/en/latest/bend.utils.embedders.html#bend.utils.embedders.DNABert2Embedder) | [Zhou et al.](https://arxiv.org/pdf/2306.15006v1.pdf) | [1 model available](https://huggingface.co/zhihan1996/DNABERT-2-117M) | |
|[GROVEREmbedder](https://bend.readthedocs.io/en/latest/bend.utils.embedders.html#bend.utils.embedders.GROVEREmbedder) | [Sanabria et al.](https://www.nature.com/articles/s42256-024-00872-0) | [1 model available](https://zenodo.org/records/8373117) | The original BPE tokenizer is not available, so we apply MaxMatch for segmentation of the input sequence into tokens.|
|[CaduceusEmbedder](https://bend.readthedocs.io/en/latest/bend.utils.embedders.html#bend.utils.embedders.CaduceusEmbedder) | [Schiff et al.](https://arxiv.org/abs/2403.03234) | [2 different models available](https://github.com/kuleshov-group/caduceus/) | Requires `mamba-ssm==1.2.0.post1` to be installed in the environment. |

All embedders can be used as follows:
```python
from bend.utils.embedders import NucleotideTransformerEmbedder

# load the embedder with a valid checkpoint name or path
embedder = NucleotideTransformerEmbedder('InstaDeepAI/nucleotide-transformer-2.5b-multi-species')

# embed a list of sequences
embeddings = embedder.embed(['AGGATGCCGAGAGTATATGGGA', 'CCCAACCGAGAGTATATGTTAT'])
# or just call directly to embed a single sequence
embedding = embedder('AGGATGCCGAGAGTATATGGGA') 

# This requires git LFS and will automatically download the checkpoint, if not already present
from bend.utils.embedders import HyenaDNAEmbedder
embedder = HyenaDNAEmbedder('pretrained_models/hyenadna/hyenadna-tiny-1k-seqlen')
```


### 5. Evaluating models

#### Training and evaluating supervised models

It is first required that the [above step (computing the embeddings)](#4-computing-embeddings) is completed.
The embeddings should afterwards be located in `BEND/data/{task_name}/{embedder}/*tar.gz`

To run a downstream task run (from `BEND/`):
```
python scripts/train_on_task.py --config-name {tasl}
```
By default the task is run on all embeddings. To alter this either modify the config file or change the settings from the commandline 
E.g. to run gene finding on all embeddings the commandline is:
```
python scripts/train_on_task.py --config-name gene_finding
```
To run only on resnetlm and awdlstm embeddings:
```
python scripts/train_on_task.py --config-name gene_finding embedder=resnetlm,awdlstm
```
The full list of current task names are : 

- `gene_finding`
- `enhancer_annotation`
- `variant_effects`
- `histone_modification`
- `chromatin_accessibility`
- `cpg_methylation`

And the list of available embedders/models used for training on the tasks are : 

- `awdlstm`
- `resnetlm`
- `nt_transformer_ms`
- `nt_transformer_human_ref`
- `dnabert6` 
- `resnet_supervised`
- `onehot`
- `nt_transformer_1000g`
- `dnabert2`
- `gena-lm-bigbird-base-t2t`
- `gena-lm-bert-large-t2`
- `hyenadna-large-1m`
- `hyenadna-tiny-1k`
- `hyenadna-small-32k`
- `hyenadna-medium-160k`
- `grover`


The `train_on_task.py` script calls a trainer class `bend.utils.task_trainer`. All configurations required to adapt these 2 scripts to train on a specific task (input data, downstream model, parameters, evaluation metric etc.) are specified in the task specific [hydra](https://hydra.cc/docs/intro/) config files stored in the [conf](../conf/) directory. This minimizes the changes required to the scripts in order to introduce a potential new task. 

The results of a run can be found at :
```
BEND/downstream_tasks/{task_name}/{embedder}/
```

If desired, the config files can be modified to change parameters, output/input directory etc.

#### Unsupervised tasks

For unsupervised prediction of variant effects, embeddings don't have to be precomputed and stored. Embeddings are generated and directly evaluated using

```bash
python3 scripts/predict_variant_effects.py {variant_file_name}.bed {output_file_name}.csv {model_type} {path_to_checkpoint} {path_to_reference_genome_fasta} --embedding_idx {position_of_embedding}
```

There are two variant effect prediction tasks available for `{variant_file_name}`: Variants with expression effect (eQTLs) in `variant_effects_expression.bed` and disease-causing variants in `variant_effects_disease.bed`.

A notebook with an example of how to run the script and evaluate the results can be found in [examples/unsupervised_variant_effects.ipynb](examples/unsupervised_variant_effects.ipynb). To run all models, you can use the script [scripts/run_variant_effects.sh](scripts/run_variant_effects.sh).

------------
## Extending BEND

### Adding a new embedder

All embedders are defined in [bend/utils/embedders.py](bend/utils/embedders.py) and inherit `BaseEmbedder`. A new embedder needs to implement `load_model`, which should set up all required attributes of the class and handle loading the model checkpoint into memory. It also needs to implement `embed`, which takes a list of sequences, and returns a list of embedding matrices formatted as numpy arrays. The `embed` method should be able to handle sequences of different lengths.

### Adding a new task
As the first step, the data for a new task needs to be formatted in the [bed-based format](#1-data-format). If necessary, a `split` and `label` column should be included. The next step is to add new config files to `../conf/supervised_tasks`. You should create a new directory named after the task, and add a config file for each embedder you want to evaluate. The config files should be named after the embedder.


-------------

## Citation Guidelines

The datasets included in BEND were collected from a variety of sources. When you use any of the datasets, please ensure to correctly cite the respective original publications describing each dataset.

### BEND
    @inproceedings{
    marin2024bend,
    title={{BEND}: Benchmarking {DNA} Language Models on Biologically Meaningful Tasks},
    author={Frederikke Isa Marin and Felix Teufel and Marc Horlacher and Dennis Madsen and Dennis Pultz and Ole Winther and Wouter Boomsma},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=uKB4cFNQFg}
    }

### Gene finding ([GENCODE](https://www.gencodegenes.org/))

    @article{frankish_gencode_2021,
	title = {{GENCODE} 2021},
	volume = {49},
	issn = {0305-1048},
	url = {https://doi.org/10.1093/nar/gkaa1087},
	doi = {10.1093/nar/gkaa1087},
	number = {D1},
	urldate = {2022-09-26},
	journal = {Nucleic Acids Research},
	author = {Frankish, Adam and Diekhans, Mark and Jungreis, Irwin and Lagarde, Julien and Loveland, Jane E and Mudge, Jonathan M and Sisu, Cristina and Wright, James C and Armstrong, Joel and Barnes, If and Berry, Andrew and Bignell, Alexandra and Boix, Carles and Carbonell Sala, Silvia and Cunningham, Fiona and Di Domenico, Tomás and Donaldson, Sarah and Fiddes, Ian T and García Girón, Carlos and Gonzalez, Jose Manuel and Grego, Tiago and Hardy, Matthew and Hourlier, Thibaut and Howe, Kevin L and Hunt, Toby and Izuogu, Osagie G and Johnson, Rory and Martin, Fergal J and Martínez, Laura and Mohanan, Shamika and Muir, Paul and Navarro, Fabio C P and Parker, Anne and Pei, Baikang and Pozo, Fernando and Riera, Ferriol Calvet and Ruffier, Magali and Schmitt, Bianca M and Stapleton, Eloise and Suner, Marie-Marthe and Sycheva, Irina and Uszczynska-Ratajczak, Barbara and Wolf, Maxim Y and Xu, Jinuri and Yang, Yucheng T and Yates, Andrew and Zerbino, Daniel and Zhang, Yan and Choudhary, Jyoti S and Gerstein, Mark and Guigó, Roderic and Hubbard, Tim J P and Kellis, Manolis and Paten, Benedict and Tress, Michael L and Flicek, Paul},
	month = jan,
	year = {2021},
	pages = {D916--D923},
    }

### Chromatin accessibility ([ENCODE](https://www.encodeproject.org/))
### Histone modification ([ENCODE](https://www.encodeproject.org/))
### CpG methylation ([ENCODE](https://www.encodeproject.org/))

    @article{noauthor_integrated_2012,
	title = {An {Integrated} {Encyclopedia} of {DNA} {Elements} in the {Human} {Genome}},
	volume = {489},
	issn = {0028-0836},
	url = {https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3439153/},
	doi = {10.1038/nature11247},
	number = {7414},
	urldate = {2023-05-23},
	journal = {Nature},
	month = sep,
	year = {2012},
	pmid = {22955616},
	pmcid = {PMC3439153},
	pages = {57--74},
    }


### Enhancer annotation ([Fulco et al.](https://www.nature.com/articles/s41588-019-0538-0), [Gasperini et al.](https://www.sciencedirect.com/science/article/pii/S009286741831554X), [Avsec et al.](https://www.nature.com/articles/s41592-021-01252-x) )

**Enhancers**

    @article{fulco_activity-by-contact_2019,
    title = {Activity-by-contact model of enhancer–promoter regulation from thousands of {CRISPR} perturbations},
    volume = {51},
    copyright = {2019 The Author(s), under exclusive licence to Springer Nature America, Inc.},
    issn = {1546-1718},
    url = {https://www.nature.com/articles/s41588-019-0538-0},
    doi = {10.1038/s41588-019-0538-0},
    language = {en},
    number = {12},
    urldate = {2023-05-23},
    journal = {Nature Genetics},
    author = {Fulco, Charles P. and Nasser, Joseph and Jones, Thouis R. and Munson, Glen and Bergman, Drew T. and Subramanian, Vidya and Grossman, Sharon R. and Anyoha, Rockwell and Doughty, Benjamin R. and Patwardhan, Tejal A. and Nguyen, Tung H. and Kane, Michael and Perez, Elizabeth M. and Durand, Neva C. and Lareau, Caleb A. and Stamenova, Elena K. and Aiden, Erez Lieberman and Lander, Eric S. and Engreitz, Jesse M.},
    month = dec,
    year = {2019},
    note = {Number: 12
    Publisher: Nature Publishing Group},
    keywords = {Epigenetics, Epigenomics, Functional genomics, Gene expression, Gene regulation},
    pages = {1664--1669},
    }

**Enhancers**

    @article{gasperini_genome-wide_2019,
    title = {A {Genome}-wide {Framework} for {Mapping} {Gene} {Regulation} via {Cellular} {Genetic} {Screens}},
    volume = {176},
    issn = {0092-8674},
    url = {https://www.sciencedirect.com/science/article/pii/S009286741831554X},
    doi = {10.1016/j.cell.2018.11.029},
    language = {en},
    number = {1},
    urldate = {2023-05-23},
    journal = {Cell},
    author = {Gasperini, Molly and Hill, Andrew J. and McFaline-Figueroa, José L. and Martin, Beth and Kim, Seungsoo and Zhang, Melissa D. and Jackson, Dana and Leith, Anh and Schreiber, Jacob and Noble, William S. and Trapnell, Cole and Ahituv, Nadav and Shendure, Jay},
    month = jan,
    year = {2019},
    keywords = {CRISPR, CRISPRi, RNA-seq, crisprQTL, eQTL, enhancer, gene regulation, genetic screen, human genetics, single cell},
    pages = {377--390.e19},
    }


**Transcription start sites**

    @article{avsec_effective_2021,
    title = {Effective gene expression prediction from sequence by integrating long-range interactions},
    volume = {18},
    copyright = {2021 The Author(s)},
    issn = {1548-7105},
    url = {https://www.nature.com/articles/s41592-021-01252-x},
    doi = {10.1038/s41592-021-01252-x},
    language = {en},
    number = {10},
    urldate = {2023-05-23},
    journal = {Nature Methods},
    author = {Avsec, Žiga and Agarwal, Vikram and Visentin, Daniel and Ledsam, Joseph R. and Grabska-Barwinska, Agnieszka and Taylor, Kyle R. and Assael, Yannis and Jumper, John and Kohli, Pushmeet and Kelley, David R.},
    month = oct,
    year = {2021},
    note = {Number: 10
    Publisher: Nature Publishing Group},
    keywords = {Gene expression, Machine learning, Software, Transcriptomics},
    pages = {1196--1203},
    }


### Noncoding Variant Effects (Expression) ([DeepSEA](https://www.nature.com/articles/nmeth.3547))
DeepSEA's data was sourced from [GRASP](https://grasp.nhlbi.nih.gov/Overview.aspx) and the [1000 Genomes Project](https://www.internationalgenome.org/), which should also be attributed accordingly.

    @article{zhou_predicting_2015,
	title = {Predicting effects of noncoding variants with deep learning–based sequence model},
	url = {https://www.nature.com/articles/nmeth.3547},
	doi = {10.1038/nmeth.3547},
	language = {en},
	number = {10},
	urldate = {2023-06-07},
	journal = {Nature Methods},
	author = {Zhou, Jian and Troyanskaya, Olga G},
	year = {2015},
    }

### Noncoding variant effects (Disease) ([ClinVar](https://www.encodeproject.org/))
In case the variant consequences categories are used, [Ensembl VEP](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-016-0974-4) should be attributed.

    @article{10.1093/nar/gkz972,
    author = {Landrum, Melissa J and Chitipiralla, Shanmuga and Brown, Garth R and Chen, Chao and Gu, Baoshan and Hart, Jennifer and Hoffman, Douglas and Jang, Wonhee and Kaur, Kuljeet and Liu, Chunlei and Lyoshin, Vitaly and Maddipatla, Zenith and Maiti, Rama and Mitchell, Joseph and O’Leary, Nuala and Riley, George R and Shi, Wenyao and Zhou, George and Schneider, Valerie and Maglott, Donna and Holmes, J Bradley and Kattman, Brandi L},
    title = "{ClinVar: improvements to accessing data}",
    journal = {Nucleic Acids Research},
    volume = {48},
    number = {D1},
    pages = {D835-D844},
    year = {2019},
    month = {11},
    issn = {0305-1048},
    doi = {10.1093/nar/gkz972},
    url = {https://doi.org/10.1093/nar/gkz972},
    eprint = {https://academic.oup.com/nar/article-pdf/48/D1/D835/31698033/gkz972.pdf},
    }

## Troubleshooting

### HyenaDNA Models

**Git LFS Not Installed:**
If you get "mkdir: missing operand" errors when trying to use HyenaDNA models, ensure Git LFS is installed:
```bash
# Install Git LFS
sudo apt-get install git-lfs  # Ubuntu/Debian
brew install git-lfs          # macOS

# Initialize in your repository
git lfs install
```

**PyTorch UnpicklingError:**
If you encounter `UnpicklingError` related to `weights_only` when loading HyenaDNA models, this is due to PyTorch 2.6+ security changes. The issue has been fixed in the codebase by setting `weights_only=False` for trusted model checkpoints.

**Model Download Issues:**
- Ensure you have sufficient disk space (HyenaDNA models can be >100MB)
- Check your internet connection for large file downloads
- Verify the model path format: `pretrained_models/hyenadna/model-name`


## FAQ

### How are embeddings upsampled?
Due to tokenization strategies, some models by default return less embedding vectors than their number of input nucleotides. As we still require nucleotide-level input for nucleotide-level prediction tasks, we implement upsampling strategies to match the number of returned embeddings to the number of input nucleotides.

Model | Upsampling strategy
--- | ---
DNABert | The overlapping k-mer tokenization strategy of DNABert causes some "missing embeddings" at the start and the end of the input sequence, as there is no context to build the k-mer tokens from. For `k=3`, we repeat the first and the last embedding vectors once. For `k=4`, we repeat the first once and the last twice. For `k=5`, we repeat the first and the last twice. For `k=6`, we repeat the first twice and the last three times. 
Nucleotide Transformer | Due to 6-mer tokenization, each embedding is repeated 6 times. Remainder tokens are single nucleotides and left as-is.
GENA-LM, DNABERT-2, GROVER | BPE tokens have variable length. We repeat each embedding vector to the length of the sequence represented by its token.

