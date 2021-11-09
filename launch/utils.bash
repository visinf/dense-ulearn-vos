#!/bin/bash

check_rundir()
{
  LOG_DIR="$1"
  EXP_ID="$2"

  if [ ! -d "$LOG_DIR" ]; then
    echo "Creating directory $LOG_DIR"
    mkdir -p $LOG_DIR
  else
    LOGD=$LOG_DIR/$EXP_ID
    if [ -d "$LOGD" ]; then
      echo "Directory $LOGD already exists."
      read -p "Do you want to remove the log files?: " -n 1 -r
      echo 
      if [[ ! $REPLY =~ ^[Yy]$ ]]
      then
        exit;
      else
        rm -rf $LOGD
      fi
    fi
  fi
}

