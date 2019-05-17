CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY,
    group_id INTEGER,
    cluster INTEGER,
    metadata JSON NOT NULL,
    zoo_id INTEGER,
    fig_dpi INTEGER,
    fig_offset INTEGER,
    fig_height INTEGER,
    fig_width INTEGER,
    fig_rows INTEGER,
    fig_cols INTEGER
);

CREATE TABLE IF NOT EXISTS image_subjects (
    subject_id UUID NOT NULL,
    image_id INTEGER NOT NULL,
    group_id INTEGER NOT NULL,
    image_index INTEGER NOT NULL
);


CREATE TABLE IF NOT EXISTS image_groups (
    group_id INTEGER PRIMARY KEY,
    group_type INTEGER NOT NULL,
    cluster_name TEXT NOT NULL,
    image_count INTEGER,
    image_size INTEGER,
    image_width INTEGER,
    description TEXT,
    permutations INTEGER
);

CREATE TABLE IF NOT EXISTS subjects (
    subject_id UUID PRIMARY KEY NOT NULL, -- assigned subject id

    /* source run, event, and telescope id
       store as run_event_tel */
    source_id TEXT UNIQUE NOT NULL,
    source TEXT NOT NULL, -- source filename

    charge bytea NOT NULL,
    batch_id INTEGER NOT NULL, -- group subjects in batches
    split_id INTEGER DEFAULT 0 -- which split group for training
);

CREATE TABLE IF NOT EXISTS subject_clusters (
    subject_id UUID NOT NULL,
    cluster_name TEXT NOT NULL,
    cluster INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS subject_labels (
    subject_id UUID NOT NULL,
    label_name TEXT NOT NULL,
    label INTEGER NOT NULL
    )
    -- PARTITION BY LIST (label_name)
    UNIQUE (label_name, subject_id);

-- CREATE TABLE IF NOT EXISTS subject_labels_vegas
    -- PARITITON OF subject_labels FOR VALUES IN ('vegas');

-- CREATE TABLE IF NOT EXISTS subject_labels_vegas_cleaned
    -- PARTITION OF subject_labels FOR VALUES IN ('vegas_cleaned');

CREATE TABLE IF NOT EXISTS sources (
    source_id TEXT PRIMARY KEY,
    source_type INTEGER NOT NULL,
    hash VARCHAR(32) NOT NULL,
    updated TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS workers (
    job_id UUID PRIMARY KEY,
    job_type TEXT,
    job_status INTEGER
);

CREATE TABLE IF NOT EXISTS worker_images (
    job_id INTEGER,
    image_id INTEGER
);

/* *********************************************** */
/* ************ Indexes ************************** */
/* *********************************************** */

/* ** Images ************************************* */

CREATE INDEX image_subjects_id
    ON image_subjects (image_id, image_index);

/* ** Subjects *********************************** */

CREATE INDEX IF NOT EXISTS subject_batch
    ON subjects (batch_id, subject_id);
CREATE INDEX IF NOT EXISTS subject_batch_split
    ON subjects (batch_id, split_id);
CREATE INDEX IF NOT EXISTS subject_split
    ON subjects (split_id);
CREATE INDEX IF NOT EXISTS subject_source_id
    ON subjects (source_id);
CREATE INDEX IF NOT EXISTS subject_source
    ON subjects (source, subject_id);

/* ** Subject Clusters *************************** */

CREATE INDEX IF NOT EXISTS id_cluster
    ON subject_clusters (cluster_name, subject_id);

/* ** Subject Labels ***************************** */

CREATE INDEX IF NOT EXISTS subject_labels
    ON subject_labels (label, subject_id);
CREATE INDEX IF NOT EXISTS subject_label_id
    ON subject_labels (subject_id);

/* ** Workers ************************************ */

CREATE INDEX IF NOT EXISTS worker_image_job
    ON worker_images (job_id);

