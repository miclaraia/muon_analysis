#!/bin/bash

DB="$MUOND/subjects-main/data-main.db"

sqlite3 $DB << EOF
ALTER TABLE image_groups ADD COLUMN group_type integer NOT NULL DEFAULT 0;
EOF

