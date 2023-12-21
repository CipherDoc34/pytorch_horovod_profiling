#!/bin/bash
#SBATCH --gpus=4                    # 4 Total GPUs
#SBATCH --mem=24G                   # Host Memory (0 is to use all of it)
#SBATCH --nodes=1
#SBATCH --ntasks=1 
#SBATCH --ntasks-per-core=4
#SBATCH --account=def-queenspp
#SBATCH --time=0:30:00
#SBATCH -o 5epochs.out

module load python/3.10
module load openmpi
module load cuda

# cd lightweight-cuda-mpi-profiler/ && make && cd ../

export CFLAGS="-L/home/keshava/pytorch_horovod/lightweight-cuda-mpi-profiler/build/lib -llwcmp"
export CXXFLAGS="-L/home/keshava/pytorch_horovod/lightweight-cuda-mpi-profiler/build/lib -llwcmp"

export LD_LIBRARY_PATH="-L/home/keshava/pytorch_horovod/lightweight-cuda-mpi-profiler/build/lib -llwcmp"

source /home/keshava/pytorch_horovod/hp-custom-horovod/bin/activate

# HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_MPI=1 HOROVOD_GPU_OPERATIONS=MPI pip install horovod[pytorch]

mpirun --oversubscribe -np 4 \
    -bind-to none -map-by slot \
    -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python3 train_resnet_horovod.py --epochs 5 --json epochs5.json -s mode5.pth -p loss5.png


# srun --ntasks-per-core=4 horovodrun -np 4 --timeline-filename out.json python3 train_resnet_horovod.py

# torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_TRAINERS dlrm/dlrm_s_pytorch.py --mini-batch-size=2 --data-size=6 --debug-mode

# salloc --time=1:0:0 --mem=24G --nodes=1 --ntasks=1 --gpus=4 --account=def-queenspp

# horovodrun -np 4 --timeline-filename test.json python3 train_resnet_horovod.py