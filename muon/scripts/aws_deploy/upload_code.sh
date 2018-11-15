#!/bin/bash
set -euxo pipefail

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
THIS="$(readlink -f ${BASH_SOURCE[0]})"

cd ${HERE}
aws_url="ec2-user@ec2-18-219-247-64.us-east-2.compute.amazonaws.com"
aws_ssh="ssh -i ${HOME}/.ssh/zoo_aws_larai002.pem"
type=$1

rsync -e "$aws_ssh" ${HERE}/muon.env $aws_url:\${HOME}
rsync -rv -e "$aws_ssh" --exclude="__pycache__" --exclude="*.pkl" --exclude="*.csv" --exclude="*.json" ${MUON}/muon $aws_url:\${HOME}/muon
rsync -e "$aws_ssh" ${MUON}/setup.py $aws_url:\${HOME}/muon
rsync -e "$aws_ssh" ${MUON}/VERSION $aws_url:\${HOME}/muon


$aws_ssh $aws_url << EOF
set -euxvo pipefail
source \${HOME}/muon.env
touch \${MUON}/README.md
if [ ! -f \$(which python3) ]; then
    if [[ ${type} == "ami" ]]; then
        sudo yum install python35-virtualenv.noarch
    elif [[ ${type} == "ubuntu" ]]; then
        sudo apt-get install python3-virtualenv
    fi
fi

cd \${MUON}
python3 -m virtualenv venv
set +u
source venv/bin/activate
set -u
pip install -e .
EOF
# rsync -rPv -e "$aws_ssh" --files-from=/tmp/upload_aws_muon_files.txt ${MUOND} ${aws_url}:/mnt/data
# rm /tmp/upload_aws_muon_files.txt

#$ssha sudo umount /mnt/data
