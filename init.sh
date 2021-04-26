#!/bin/bash
set -e

read -rep "download and deploy UNSC corpus? (y/n): " ans
if [ $ans == "y" ]; then
  # get us-election and unsc data
  cd ./data
  wget -O ./UNSC/docs.RData "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/KGVSYH/G2DENH"
  wget -O ./UNSC/docs_meta.RData "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/KGVSYH/KHJOUV"
  wget -O ./UNSC/speeches/speeches.tar "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/KGVSYH/YUGPPE"
  tar -C ./UNSC/speeches -xvf ./UNSC/speeches/speeches.tar
  wget -O ./UNSC/meta.tsv "https://dataverse.harvard.edu/api/access/datafile/3457365?format=original&gbrecs=true"
  wget -O ./UNSC/speaker.tsv "https://dataverse.harvard.edu/api/access/datafile/3457366?format=original&gbrecs=true"
  cd ..
fi

read -rep "download and deploy US Election Debate corpus? (y/n): " ans
if [ $ans == "y" ]; then
  # get us-election and unsc data
  cd ./data
  wget -O ./USElectionDebates/ElecDeb60To16.zip "https://github.com/ElecDeb60To16/Dataset/raw/master/ElecDeb60To16.zip"
  unzip -o ./USElectionDebates/ElecDeb60To16.zip -d ./USElectionDebates/
  cd ..
fi

read -rep "create pre-commit hook for updating python dependencies? (y/n): " ans
if [ $ans == "y" ]; then
  # move pre-commit hook into local .git folder for activation
  cp ./hooks/pre-commit ./.git/hooks/pre-commit
fi
