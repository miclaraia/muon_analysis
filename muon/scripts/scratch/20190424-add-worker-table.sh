#!/bin/bash

sqlite3 $MUOND/subjects-main/data-main.db << EOF
CREATE TABLE IF NOT EXISTS workers (
    job_id INTEGER PRIMARY KEY,
    job_type TEXT,
    job_status INTEGER
);
CREATE TABLE IF NOT EXISTS worker_images (
    job_id INTEGER,
    image_id INTEGER
);
CREATE INDEX IF NOT EXISTS worker_image_job
    ON worker_images (job_id);
EOF
