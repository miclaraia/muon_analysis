Database file at
`/data/lucifer1.2/users/larai002/muon_data/subjects-main/i
data-pg-backup-20200105.gz`

A subject is one of the individual detections. An image is a grid that is
uploaded to the MH2 project. An image-group is a batch of images uploaded to
the MH2 project.

|--------------------|--------------------------------------------------------|
| `image_groups`     |                                                        |
|--------------------|--------------------------------------------------------|
| `image_subjects`   | defines which subjects are in an image                 |
|--------------------|--------------------------------------------------------|
| `images`           |                                                        |
|--------------------|--------------------------------------------------------|
| `sources`          | keeps track of the source files used to generate the db|
|--------------------|--------------------------------------------------------|
| `subject_clusters` | record of which cluster a subject belongs to           |
|--------------------|--------------------------------------------------------|
| `subject_labels`   | records of labels for a subject. Each subject can have |
|                    | multiple labels from different sources (like vegas,    |
|                    | volunteer, etc)                                        |
|--------------------|--------------------------------------------------------|
| `subjects`         | each subject. includes a binary numpy array of pixel   |
|                    | data                                                   |
|--------------------|--------------------------------------------------------|
| `worker_images`    | workers for parallelizing image generation and upload  |
| `workers`          |                                                        |
|--------------------|--------------------------------------------------------|

`scripts/scratch/20190503-dump-training-images.py` a script that you could use
as a guide. This may be the script I used to generate some images for Kevin
originally, but I'm not certain on that right now. The SQL syntax there was
likely for sqlite, but the database is now postgres so that may need a few
modifications here and there to make it work again. You may want to modify it a
bit to exclude subjects that were included in the original dump. This can
probably be accomplished by creating a temporary table with the ids that Kevin
has and joining against that.

In terms of actually running a postgres server on lucifer, I just compiled
postgres myself and ran it locally. You could probably find some AWS resources
to have a small machine and install PG, or you could do the same as I did and
compile postgres for lucifer. I'll find a way to give you access to one of the
database dumps that I mentioned at the top of this doc.

You'll also want to make a copy of `muon/config/default_config.yml` as
`muon/config/config.yml`, and edit the `database` fields to point to whatever
database you set up.
