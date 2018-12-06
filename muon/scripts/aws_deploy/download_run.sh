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
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done
echo ${POSITIONAL[@]}

set -- "${POSITIONAL[@]:-}"

file=/tmp/down_aws_models.txt

echo > $file
while [[ $# -gt 0 ]]; do
    echo $1 >> $file
    shift
done

cd ${HERE}
SSH="$(${HERE}/ssh.sh -h ${HOST} --type ${INSTANCE_TYPE} -p)"
SSH_PRE="$(python3 -c "print(' '.join('${SSH}'.split(' ')[:-1]))")"
HOST="$(python3 -c "print('${SSH}'.split(' ')[-1])")"

rsync -rPv -e "$SSH_PRE" --files-from=$file $HOST:/mnt/muon/data/clustering_models/ $MUOND/clustering_models/aws
rm $file

for file in $(find ${MUOND}/clustering_models/aws -name config.json); do
    if [ ! -z "$(grep "/mnt/muon/data" $file)" ]; then
        sed -i "s,/mnt/muon/data,${MUOND},g" $file
        sed -i "s,clustering_models,clustering_models/aws,g" $file
    fi
done


#cat > /tmp/upload_aws_muon_files.txt << EOF
#
#subjects/
#EOF
#
#rsync -rPv -e "$SSH_PRE" --files-from=/tmp/upload_aws_muon_files.txt ${MUOND} $HOST:/mnt/muon/data
#rm /tmp/upload_aws_muon_files.txt

#$ssha sudo umount /mnt/data
