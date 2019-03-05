Generating and Uploading Images
===============================

I would set up a python virtualenv so the muon code doesn't clobber your environment
```
python3 -m venv venv
source venv/bin/activate
pip install -e {path to code}
```

Generating muon images
```
python {path to code}/muon/scripts/data_management/pipeline/d_generate_image.py --groups 12 --dpi 50 {path to database} {path to image root}
```

Uploading images to project
```
PANOPTES_USERNAME="{zooniverse username}"
PANOPTES_PASSWORD="{zooniverse password}"
python {path to code}/muon/scripts/data_management/pipeline/e_upload_images.py --groups 12 {path to database} {path to image root}
```

The scripts are designed and pick up where they left off if they fail and are restarted. 
