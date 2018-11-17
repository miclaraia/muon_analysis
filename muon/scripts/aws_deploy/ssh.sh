#!/bin/bash
set -euxo pipefail

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
THIS="$(readlink -f ${BASH_SOURCE[0]})"

IDENTITY=${HOME}/.ssh/zoo_aws_larai002.pem
INSTANCE_TYPE=ami

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
        -p|--print)
            PRINT=1
            shift
            ;;
        -t|--type)
            INSTANCE_TYPE=$2
            shift 2
            ;; 
        -T)
            TTY=TRUE
            shift
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done
set -- "${POSITIONAL[@]}"
COMMAND="${1:-}"

if [[ ${INSTANCE_TYPE} == "ami" ]]; then
    USER="ec2-user"
elif [[ ${INSTANCE_TYPE} == "ubuntu" ]]; then
    USER="ubuntu"
fi

if [[ ${COMMAND} == "exit" ]]; then
    ssh -O exit ${USER}@${HOST}
    exit
fi

command="ssh -i ${IDENTITY} ${USER}@${HOST} ${COMMAND}"
if [ ${PRINT:-} ]; then
    echo ${command}
elif [[ ${TTY:-} == TRUE ]]; then
    command="${command} -tt source \${HOME}/muon.env && bash"
    $command
else
    $command
fi


#cd ${HERE}
#aws_url="ec2-user@ec2-18-219-247-64.us-east-2.compute.amazonaws.com"
#aws_ssh="ssh -i ${HOME}/.ssh/zoo_aws_larai002.pem"
#type=$1



#$aws_ssh $aws_url << EOF
#set -euxvo pipefail
#source \${HOME}/muon.env
#touch \${MUON}/README.md
#if [ ! -f \$(which python3) ]; then
    #if [[ ${type} == "ami" ]]; then
        #sudo yum install python35-virtualenv.noarch git
    #elif [[ ${type} == "ubuntu" ]]; then
        #sudo apt-get install python3-virtualenv git
    #fi
#fi

#EOF




