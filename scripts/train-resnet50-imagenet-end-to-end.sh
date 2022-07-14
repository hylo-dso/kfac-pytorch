#!/bin/bash

# Wait for prolog to finish
echo "Wait 3s..."
sleep 3
echo "done"

PRELOAD="module load anaconda3;"
PRELOAD+="source activate $SCRATCH/ad-v1;"

# Args of training script
CMD="examples/torch_imagenet_resnet.py --train-dir $SCRATCH/imagenet/train --val-dir $SCRATCH/imagenet/validation --base-lr 0.03125 --kfac-update-freq 200 --kfac-cov-update-freq 20 --batch-size 80 --val-batch-size 80 --kfac-comm-method hybrid-opt --kfac-grad-worker-fraction 0.5 --damping 0.003 --checkpoint-freq 5 --log-dir resnet50-imagenet-end-to-end --epochs 60 $@"

# Distributed system configuration
if [[ -z "${NODEFILE}" ]]; then
    if [[ -n "${SLURM_NODELIST}" ]]; then
        NODEFILE=/tmp/imagenet_slurm_nodelist
        scontrol show hostnames $SLURM_NODELIST > $NODEFILE
    elif [[ -n "${COBALT_NODEFILE}" ]]; then
        NODEFILE=$COBALT_NODEFILE
    fi
fi
if [[ -z "${NODEFILE}" ]]; then
    RANKS=$HOSTNAME
    NNODES=1
else
    MAIN_RANK=$(head -n 1 $NODEFILE)
    RANKS=$(tr '\n' ' ' < $NODEFILE)
    NNODES=$(< $NODEFILE wc -l)
fi

# Torch distributed launcher
LAUNCHER="python -m torch.distributed.launch "
LAUNCHER+="--nnodes=$NNODES --nproc_per_node=4 "

FULL_CMD="$PRELOAD $LAUNCHER $CMD"
echo "Training command: $FULL_CMD"

# Launch the pytorch processes on each worker (use ssh for remote nodes)
RANK=0
for NODE in $RANKS; do
    FULL_CMD+=" --url tcp://$MAIN_RANK:1234"
    echo "node idx cmd: $FULL_CMD"
    if [[ "$NODE" == "$HOSTNAME" ]]; then
        echo "Launching rank $RANK on local node $NODE"
        eval $FULL_CMD &
    else
        echo "Launching rank $RANK on remote node $NODE"
        ssh $NODE "cd $PWD; $FULL_CMD" &
    fi
    RANK=$((RANK+1))
done

wait
