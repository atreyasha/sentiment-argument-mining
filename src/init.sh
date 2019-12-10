#!/bin/bash
set -e

read -rep "create pre-commit hook for updating python dependencies? (y/n): " ans
if [ $ans == "y" ]; then
    # move pre-commit hook into local .git folder for activation
    cp ../hooks/pre-commit.sample ../.git/hooks/pre-commit
fi

read -rep "download and deply UNSC/USElection data? (y/n): " ans
if [ $ans == "y" ]; then
    # get lfw-faces data
    cd ./data
    wget -O ./USElectionDebates/ElecDeb60To16.zip https://github.com/ElecDeb60To16/Dataset/raw/master/ElecDeb60To16.zip
    unzip -o ./USElectionDebates/ElecDeb60To16.zip -d ./USElection/
    wget -O ./UNSC/docs.RData https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/KGVSYH/G2DENH
    wget -O ./UNSC/docs_meta.RData https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/KGVSYH/KHJOUV
    cd ..
fi
