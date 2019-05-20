#!/bin/bash
source $MUON/venv/bin/activate
set -euo pipefail
source $MUON/muon.env

python $MUON/muon/scripts/scratch/20190516-generate-single.py init-groups
python $MUON/muon/scripts/project/pipeline/workers.py generate 25
python $MUON/muon/scripts/project/pipeline/workers.py generate 26
python $MUON/muon/scripts/project/pipeline/workers.py run
