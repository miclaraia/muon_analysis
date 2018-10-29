set -euo pipefail


HUGH=$MUOND/zooniverse/MH2/mh2_golds.csv
EXPORT=$MUOND/zooniverse/MH2/muon-hunters-2-dot-0-classifications.csv

IMAGES=$MUOND/zooniverse/MH2/image_structure/production_data.h5
SUBJECTS=$MUOND/subjects/subject_data_v3.hdf5

LABELS1=$MUOND/zooniverse/MH2/majority_labels.csv
LABELS2=$MUOND/zooniverse/MH2/mh2_golds.csv

cd $MUOND/subjects
cp subject_data_v2.hdf5 ${SUBJECTS}

cd $MUOND/zooniverse/MH2
python $MUON/muon/scripts/mh2_dump_volunteer_labels.py \
  ${EXPORT} ${IMAGES} ${LABELS1}

cd $MUOND/subjects
python $MUON/muon/scripts/add_subject_labels.py volunteer_majority \
  ${LABELS1} ${SUBJECTS}

python $MUON/muon/scripts/add_subject_labels.py hugh \
  ${LABELS2} ${SUBJECTS}


