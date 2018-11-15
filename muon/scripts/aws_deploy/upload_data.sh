#!/bin/bash
set -euxo pipefail

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
THIS="$(readlink -f ${BASH_SOURCE[0]})"

cd ${HERE}
aws_url="ec2-user@ec2-18-219-247-64.us-east-2.compute.amazonaws.com"
ssha="ssh -i ${HOME}/.ssh/zoo_aws_larai002.pem ${aws_url}"

scp -i ${HOME}/.ssh/zoo_aws_larai002.pem ${HERE}/muon.env $aws_url:/home/ec2-user
$ssha << EOF
set -xv
source \${HOME}/muon.env
if [ -z \$(mount | grep /mnt/muon) ]; then
    sudo mkdir -p /mnt/muon
    sudo mount -U 96d0c028-399a-4559-967f-ebaa5177d87e /mnt/muon
    sudo chmod 777 /mnt/muon
fi
if [ ! -d /mnt/muon/data ]; then
    sudo mkdir -p /mnt/muon/data
    sudo mkdir -p /mnt/muon/muon
    sudo chmod 777 /mnt/muon/data /mnt/muon/muon
fi
EOF

cat > /tmp/upload_aws_muon_files.txt << EOF
clustering_models/
subjects/
EOF

rsync -rPv -e "ssh -i ${HOME}/.ssh/zoo_aws_larai002.pem" --files-from=/tmp/upload_aws_muon_files.txt ${MUOND} ${aws_url}:/mnt/muon/data
rm /tmp/upload_aws_muon_files.txt

#$ssha sudo umount /mnt/data
