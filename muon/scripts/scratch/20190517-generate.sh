set -euo pipefail

$MUON/muon/scripts/scratch/20190516-generate-single.py init_groups
python $MUON/muon/scripts/project/pipeline/workers.py generate 25
python $MUON/muon/scripts/project/pipeline/workers.py generate 26
python $MUON/muon/scripts/project/pipeline/workers.py run
