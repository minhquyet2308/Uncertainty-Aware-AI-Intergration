#!/bin/bash
#SBATCH --job-name=HEA-GPT     # create a short name for your job
#SBATCH --partition=general,highmem                # node count
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=32      # number of tasks per node
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=128G                # memory per node
#SBATCH --time=300-96:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=/home/quyet/damlabfs/spark_slurm/%u-%x-%j.out

cd /home/quyet/work/HEA_GPT/code

# find ers/ -name "*.py" -exec zip ./tmp/ers.zip {} \;

CLUSTER="trump"

SPARK_VERSION="spark-3.3.4-bin-hadoop2"
SPARK_HOME="/opt/${CLUSTER}/spark/${SPARK_VERSION}"
if [[ -d "${SPARK_HOME}" ]]; then
    export SPARK_HOME
    PATH="${SPARK_HOME}/sbin:${SPARK_HOME}/bin:${PATH}"
    export SPARK_LOCAL_IP="127.0.0.1"
fi

JAVA_HOME="/opt/${CLUSTER}/java/default"
if [[ -d "${JAVA_HOME}/bin" ]]; then
    export JAVA_HOME
    PATH="${JAVA_HOME}/bin:${PATH}"
fi

SCALA_HOME="/opt/${CLUSTER}/scala/default"
if [[ -d "${SCALA_HOME}/bin" ]]; then
    export SCALA_HOME
    PATH="${SCALA_HOME}/bin:${PATH}"
fi

SBT_HOME="/opt/${CLUSTER}/sbt/default"
if [[ -d "${SBT_HOME}/bin" ]]; then
    export SBT_HOME
    PATH="${SBT_HOME}/bin:${PATH}"
fi

echo $SPARK_LOCAL_IP

# This syntax tells spark to use all cpu cores on the node.
export MASTER="local[*]"

CONFIG=$1

echo $MASTER

CONDA_PYTHON=/home/quyet/anaconda3/envs/vasp-viz-env-spark-3.3.4/bin/python3.10

export PYSPARK_DRIVER_PYTHON=${CONDA_PYTHON}
export PYSPARK_PYTHON=${CONDA_PYTHON}

spark-submit --master $MASTER --py-files ./tmp/ers.zip --conf spark.jars.ivy=/tmp/.ivy HEA_experiments.py --config ${CONFIG} --spark True
