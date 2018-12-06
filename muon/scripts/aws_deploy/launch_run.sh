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
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done
set -- "${POSITIONAL[@]}"

MODEL=$1
shift

cd ${HERE}
SSH="$(${HERE}/ssh.sh -h ${HOST} --type ${INSTANCE_TYPE} -p)"

set +o pipefail
RAND=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 8 | head -n 1)
set -o pipefail
fname=/tmp/launch_aws_muon_${RAND}

MODEL_FILE="\${MUON}/muon/scripts/models/${MODEL}"
cat > ${fname} << EOF
#!/bin/bash
set -euxvo pipefail
source \${HOME}/muon.env
set +u
. \${HOME}/muon-activate
set -u
echo \$(which python)

${MODEL_FILE} $@
EOF

$SSH << EOF
source \$HOME/muon.env
if [ ! -f ${MODEL_FILE} ]; then
    echo Can\'t find model ${MODEL_FILE}
    exit 1
fi
EOF

cat > ${fname}.screen << EOF
zombie kr
verbose on
EOF
MODEL=$(python -c "print('${MODEL}'.replace('/','_'))")

SSH_PRE="$(python3 -c "print(' '.join('${SSH}'.split(' ')[:-1]))")"
HOST="$(python3 -c "print('${SSH}'.split(' ')[-1])")"
rsync -e "$SSH_PRE" ${fname} ${HOST}:${fname}
rsync -e "$SSH_PRE" ${fname}.screen ${HOST}:\${HOME}/muon.screen
$SSH -tt screen -S ${MODEL} -c \${HOME}/muon.screen bash ${fname}
