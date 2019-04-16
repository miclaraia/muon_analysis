#!/bin/bash

cd $MUON
svenv

DB=$MUOND/subjects-main/data-main.db
IMAGES=$MUOND/images/mh2_images-main

python $MUON/muon/scripts/data_management/pipeline/single_images/c_generate_image_groups.py --batches 13 $DB
sqlite3 $DB << EOF
SELECT * FROM image_groups;
EOF
echo "Using group 20"
read  -n 1 -p "Continue?"
python $MUON/muon/scripts/data_management/pipeline/d_generate_image.py --groups 20 $DB $IMAGES
