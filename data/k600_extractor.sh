# https://github.com/cvdfoundation/kinetics-dataset/blob/main/k600_extractor.sh

#!/bin/bash

curr_dl=data/k600_targz/train
curr_extract=data/k600/train_raw
[ ! -d $curr_extract ] && mkdir -p $curr_extract
find $curr_dl -type f | while read file; do mv "$file" `echo $file | tr ' ' '_'`; done
tar_list=$(ls $curr_dl)
for f in $tar_list
do
	[[ $f == *.tar.gz ]] && echo Extracting $curr_dl/$f to $curr_extract && tar zxf $curr_dl/$f -C $curr_extract
done


curr_dl=data/k600_targz/val
curr_extract=data/k600/val_raw
[ ! -d $curr_extract ] && mkdir -p $curr_extract
find $curr_dl -type f | while read file; do mv "$file" `echo $file | tr ' ' '_'`; done
tar_list=$(ls $curr_dl)
for f in $tar_list
do
	[[ $f == *.tar.gz ]] && echo Extracting $curr_dl/$f to $curr_extract && tar zxf $curr_dl/$f -C $curr_extract
done

# Extraction complete
echo -e "\nExtractions complete!"
