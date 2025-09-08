# Evidential Classifier Framework

The **Evidential Classifier Framework** integrates knowledge extracted from datasets with expert knowledge derived from literature studies using **Large Language Models (LLMs)**.  
This repository provides implementations, environment setup instructions, and guidance on running experiments.

---

## 📦 Setting Up the Python Environment

We recommend using **Conda** to manage the Python environment.

1. **Install Conda**  
   Download and install from [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

2. **Create and Activate the Conda Environment**
   ```bash
   conda create --name evidential_classifier
   conda activate evidential_classifier

3. **Install Dependencies:The requirements.txt file lists all necessary Python packages. Install them using:
pip install -r requirements.txt

Ensure you are in the Conda environment (evidential_classifier) before running the above command.

# Code Overview
The code is organized as follows:
EvidentialClassifier Folder: Contains the implementation of the proposed Evidential Classifier Framework. This framework integrates dataset-derived knowledge with expert knowledge extracted from literature studies using LLMs.
HEA_experiments.py: Implements experiments to evaluate the framework's performance in interpolation and extrapolation tasks. This script supports running experiments either locally on a single laptop or on a parallel system using Slurm and Apache Spark.

# Running Experiments
The HEA_experiments.py script supports two modes of execution:
1. Running Locally (Single Laptop)
To run the experiments on a single machine without Apache Spark:
python HEA_experiments.py --config ../data/IMCN_quaternary_0.9Tm/interpolation_cv_0.1.yaml --spark False

--config: Path to the configuration file (e.g., interpolation_cv_0.1.yaml).
--spark False: Disables Apache Spark, running the experiment locally.

2. Running on a Parallel System with Slurm and Apache Spark
To run the experiments on a cluster using Slurm and Apache Spark:
sbatch HEA_experiments.sh --config ../data/IMCN_quaternary_0.9Tm/interpolation_cv_0.1.yaml


HEA_experiments.sh: A Slurm script that sets up the environment and runs HEA_experiments.py with Apache Spark enabled.
Ensure the Slurm script is configured with the appropriate cluster settings (e.g., number of nodes, memory, etc.).
The --spark flag is implicitly enabled in the Slurm script to use Apache Spark for parallel processing.

# Dataset
The dataset used in this implementation is located in the data/ directory:
HEA_data.IMCN.quaternaries_0.9Tm.csv: An example CSV file containing the dataset for quaternary alloys at 0.9Tm. This serves as a reference for the data format expected by the framework.
Additional Datasets: Other datasets used in the experiments are published on Zenodo. [Citation and link to be provided later.]

# Configuration Files
The configuration files in the data/IMCN_quaternary_0.9Tm/ directory define the settings for the experiments:
all.yaml: Configures the framework to train on the entire dataset.
interpolation_cv_x.yaml: Configures the framework for interpolation evaluation. Here, x represents the percentage of data (e.g., 0.1 for 10%) randomly selected as training data, with the rest used for testing.
extrapolation.yaml: Configures the framework for extrapolation evaluation. This omits alloys containing a specific element from the training set, using those alloys as the test set to assess the framework's generalization capability.

# Notes
* Ensure the dataset and configuration files are correctly placed in the data/IMCN_quaternary_0.9Tm/ directory before running experiments.
* For cluster runs, verify that your Slurm environment is properly configured with Apache Spark dependencies.
* If you encounter issues, check the Python environment for missing dependencies or consult the requirements.txt file.
For further details or to access the datasets, refer to the Zenodo link [to be provided].