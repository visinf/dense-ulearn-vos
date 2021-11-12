#!/bin/bash

# Set the following variables
# The tensorboard logging will be creating in logs/<EXP>/<EXP_ID>
# The snapshots will be saved in snapshots/<EXP>/<EXP_ID>
EXP=v01_vos
EXP_ID=v01_00_base

#
# No change are necessary starting here
#

DS=$1
SEED=32

case $DS in
oxuva)
  echo "Train dataset: OxUvA"
  TASK="OxUvA_all"
  CFG=configs/oxuva.yaml
  EXP_ID="OX_${EXP_ID}"
  ;;
ytvos)
  echo "Train dataset: YouTube-VOS"
  TASK="YTVOS"
  CFG=configs/ytvos.yaml
  EXP_ID="YT_${EXP_ID}"
  ;;
track)
  echo "Train dataset: TrackingNet"
  TASK="TrackingNet"
  CFG=configs/tracknet.yaml
  EXP_ID="TN_${EXP_ID}"
  ;;
kinetics)
  echo "Train dataset: Kinetics-400"
  CFG=configs/kinetics.yaml
  EXP_ID="KT_${EXP_ID}"
  ;;
*)
  echo "Dataset '$DS' not recognised. Should be one of [oxuva|ytvos|track|kinetics]."
  exit 1
  ;;
esac


CURR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source $CURR_DIR/utils.bash

CMD="python train.py --cfg $CFG --exp $EXP --run $EXP_ID --seed $SEED"
LOG_DIR=logs/${EXP}
LOG_FILE=$LOG_DIR/${EXP_ID}.log
echo "LOG: $LOG_FILE"

check_rundir $LOG_DIR $EXP_ID

NUM_THREADS=12

export OMP_NUM_THREADS=$NUM_THREADS
export MKL_NUM_THREADS=$NUM_THREADS

echo $CMD

CMD_FILE=$LOG_DIR/${EXP_ID}.cmd
echo $CMD > $CMD_FILE

git rev-parse HEAD > $LOG_DIR/${EXP_ID}.head
git diff > $LOG_DIR/${EXP_ID}.diff

nohup $CMD > $LOG_FILE 2>&1 &
sleep 1
tail -f $LOG_FILE
