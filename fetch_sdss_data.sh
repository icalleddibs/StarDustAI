#!/bin/bash
wget -nv -r -nH --cut-dirs=7 \
     -i speclist.txt \
     -B https://data.sdss.org/sas/dr17/eboss/spectro/redux/v5_13_2/spectra/full/ \
     -P data/
