#!/bin/bash
set -euxo pipefail

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
THIS="$(readlink -f ${BASH_SOURCE[0]})"

POSITIONAL=()
while [[ $# -gt 0 ]]; do
    key="$1"

    case ${key} in
        -h|--host)
            HOST="$2"
            shift 2
            ;;
        -i|--identity)
            IDENTITY="$2"
            shift 2
            ;;
        -t|--type)
            INSTANCE_TYPE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done
set -- "${POSITIONAL[@]:-}"

cd ${HERE}
SSH="$(${HERE}/ssh.sh -h ${HOST} --type ${INSTANCE_TYPE} -p)"
SSH_PRE="$(python3 -c "print(' '.join('${SSH}'.split(' ')[:-1]))")"
HOST="$(python3 -c "print('${SSH}'.split(' ')[-1])")"

rsync -e "$SSH_PRE" ${HERE}/muon.env $HOST:\${HOME}
$SSH << EOF
set -xv
source \${HOME}/muon.env
if [ -z \$(mount | grep /mnt/muon) ]; then
    sudo mkdir -p /mnt/muon
    sudo mount -U ${DEVICE} /mnt/muon
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

rsync -rcPv -e "$SSH_PRE" \
    --files-from=/tmp/upload_aws_muon_files.txt \
    --exclude 'clustering_models/' \
    --exclude 'clustering_models/aws' \
    --exclude 'clustering_models/run*' \
    --exclude 'clustering_models/hugh' \
    --exclude 'clustering_models/volunteer' \
    --exclude 'clustering_models/decv2' \
    ${MUOND} $HOST:/mnt/muon/data
rm /tmp/upload_aws_muon_files.txt

#$ssha sudo umount /mnt/data
