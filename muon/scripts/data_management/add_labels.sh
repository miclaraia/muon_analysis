set -euo pipefail


HUGH=$MUOND/zooniverse/MH2/mh2_golds.csv
EXPORT=$MUOND/zooniverse/MH2/muon-hunters-2-dot-0-classifications.csv

IMAGES=$MUOND/zooniverse/MH2/image_structure/production_data.h5
SUBJECTS=$MUOND/subjects/subject_data_v3.hdf5

LABELS1=$MUOND/zooniverse/MH2/majority_labels.csv
LABELS2=$MUOND/zooniverse/MH2/mh2_golds.csv

cd $MUOND/subjects
echo "Copying subjects file"
cp subject_data_v2.hdf5 ${SUBJECTS}

# cd $MUOND/zooniverse/MH2
# echo "Dumping volunteer labels"
# python $MUON/muon/scripts/mh2_dump_volunteer_labels.py \
#   ${EXPORT} ${IMAGES} ${LABELS1}

echo "Adding volunteer labels to subjects file"
python $MUON/muon/scripts/add_subject_labels.py \
    --name volunteer_majority \
    --labels_csv ${LABELS1} \
    --subjects_h5 ${SUBJECTS}

echo "Adding hugh labels to subjects file"
python $MUON/muon/scripts/add_subject_labels.py \
    --name hugh \
    --labels_csv ${LABELS2} \
    --subjects_h5 ${SUBJECTS}

python $MUON/muon/scripts/add_subject_labels.py \
    --name swap \
    --labels_csv ${MUOND}/zooniverse/MH2/swap_labels.csv \
    --subjects_h5 ${SUBJECTS}


