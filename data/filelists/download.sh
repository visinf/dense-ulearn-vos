#!/bin/bash

echo "Downloading file lists"

ROOT_URL=download.visinf.tu-darmstadt.de/data/2021-neurips-araslanov-vos/filelists

download () {
  echo $1 $ROOT_URL
  curl $ROOT_URL/$1 --create-dirs -o $1
}

# Train
download train_kinetics400.txt
download train_oxuva.txt
download train_tracking.txt
download train_ytvos.txt

# Val
download val2_davis2017_480p.txt
download val_davis2017_480p.txt
download val_davis2017_test.txt
download val_ytvos2018_test.txt

echo "Done"
