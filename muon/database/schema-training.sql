CREATE TABLE IF NOT EXISTS subjects (
    subject_id TEXT PRIMARY KEY NOT NULL, -- assigned subject id
    charge bytea NOT NULL,
    label INTEGER NOT NULL,
    split_id INTEGER DEFAULT 0,
    cluster_id INTEGER,
    rand INTEGER
);

CREATE INDEX IF NOT EXISTS split 
    ON subjects (split_id, subject_id);
CREATE INDEX IF NOT EXISTS split 
    ON subjects (split_id, rand);
