sqlite3 -header -csv $MUOND/subjects-main/data-main.db > $MUOND/kevin_sims_id_ref.csv << EOF
select image_subjects.subject_id, image_subjects.image_id,
       group_id, subjects.source_id, subjects.source
from image_subjects
inner join subjects
  on image_subjects.subject_id=subjects.subject_id
where image_subjects.group_id in (21,22)
order by subjects.source;
EOF
