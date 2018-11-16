#!/bin/bash
set -euxo pipefail

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
THIS="$(readlink -f ${BASH_SOURCE[0]})"

POS=()
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
        -c|--clean)
            CLEAN=TRUE
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done
set -- "${POSITIONAL[@]}"

cd ${HERE}
SSH="$(${HERE}/ssh.sh -h ${HOST} --type ${INSTANCE_TYPE} -p)"
SSH_PRE="$(python3 -c "print(' '.join('${SSH}'.split(' ')[:-1]))")"
HOST="$(python3 -c "print('${SSH}'.split(' ')[-1])")"

rsync -e "$SSH_PRE" ${HERE}/muon.env ${HOST}:\${HOME}
rsync -rv -e "$SSH_PRE" \
    --exclude="__pycache__" \
    --exclude="*.pkl" \
    --exclude="*.csv" \
    --exclude="*.json" \
    ${MUON}/muon ${HOST}:\${HOME}/muon
rsync -e "$SSH_PRE" ${MUON}/setup.py ${HOST}:\${HOME}/muon
rsync -e "$SSH_PRE" ${MUON}/VERSION ${HOST}:\${HOME}/muon


if [[ "${CLEAN:-}" == TRUE ]]; then
$SSH << EOF
set -euxvo pipefail
source \${HOME}/muon.env
touch \${MUON}/README.md
if [ ! -f \$(which python3) ] || [ -z \$(python3 -m virtualenv) ]; then
    if [[ ${INSTANCE_TYPE} == "ami" ]]; then
        sudo yum install python35-virtualenv.noarch git
    elif [[ ${INSTANCE_TYPE} == "ubuntu" ]]; then
        sudo apt-get update
        sudo apt-get install python3-virtualenv git
    fi
fi

if [ -z \$(mount | grep /mnt/muon) ]; then
    sudo mkdir -p /mnt/muon
    sudo mount -U 96d0c028-399a-4559-967f-ebaa5177d87e /mnt/muon
    sudo chmod 777 /mnt/muon
fi
find \${MUOND} -not -user \${USER} -exec sudo chown \${USER}:\${USER} \;


if [[ \$(which python3) == "/home/ubuntu/anaconda3/bin//python3" ]]; then
    if [ ! -z "\$(conda env list | grep muon)" ]; then
        conda env remove --name muon -y
    fi

    conda create --name muon --clone tensorflow_p36
cat > \${HOME}/muon-activate << EOF2
source activate muon
EOF2

else
    if [ -d \${MUON}/venv ]; then
        rm -R \${MUON}/venv
    fi
    python3 -m virtualenv --python=python3 \${MUON}/venv
cat > \${HOME}/muon-activate << EOF2
source \${MUON}/venv/bin/activate
EOF2

fi

set +u
. \${HOME}/muon-activate
set -u

pip install --upgrade pip
pip install git+https://github.com/miclaraia/DEC-keras.git
#pip uninstall -y tensorflow tensorflow-cpu
#pip install tensorflow-gpu
pip install -e \${MUON}

python -c "import tensorflow as tf; sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))"
EOF

fi
# rsync -rPv -e "$SSH" --files-from=/tmp/upload_aws_muon_files.txt ${MUOND} ${{HOST}}:/mnt/data
# rm /tmp/upload_aws_muon_files.txt

#$ssha sudo umount /mnt/data
