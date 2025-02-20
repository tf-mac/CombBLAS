#!/bin/bash

# test MultTimingCUDA
if [ -z "${DLOC}" ]; then
  echo "We use env var DLOC to save dataset, please set it."
  exit 1
else
  echo "DLOC is set to $DLOC."
fi

##!/bin/bash
#
#echo -e "\e[31mThis is red text\e[0m"
#echo -e "\e[32mThis is green text\e[0m"
#echo -e "\e[33mThis is yellow text\e[0m"
#echo -e "\e[34mThis is blue text\e[0m"
#echo -e "\e[35mThis is magenta text\e[0m"
#echo -e "\e[36mThis is cyan text\e[0m"
#echo -e "\e[0mThis is default text"


# Check if hostname is provided as the first argument
if [ -z "$1" ]; then
  echo "Please provide a testbed environment as the first argument (debug, pmt, delta, dai)."
  exit 1
fi

machine="$1"

# Validate the machine using a case statement
case "$machine" in debug|pmt|delta|dai)
        echo "Hostname accepted: $machine"
    ;;
    *)
        echo "Invalid machine. Accepted types are: debug, pmt, delta, dai."
        exit 1
    ;;
esac


REMOTE_USER="exouser"
REMOTE_HOST="149.165.155.206"
REMOTE_PATH="/media/volume/workspace/kk/tfCombBLAS-minor/"

# Define SSH key (optional if using default ~/.ssh/id_rsa)
SSH_KEY_PATH="~/.ssh/id_rsa_kl23395"
# Use rsync to transfer the folder
function snycfromdebug {
    rsync -avz -e "ssh -i $SSH_KEY_PATH" \
    --exclude "debug-*" \
    --exclude "release-*" \
    --exclude ".git" \
    --exclude ".cache" \
    --delete \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH" .
}


# Branch for each hostname type with separate logic
binary=$2


# Get the last parameter, last parameters is mpi processor numbers
last_param="${!#}"

# Check if the last parameter is a number
if [[ ! "$last_param" =~ ^[0-9]+$ ]]; then
    echo "Error: Last parameter '$last_param' is not a number."
    exit 1
fi

nprocs=$last_param

function print_green {
    local input="$1"
    local prefix="========================================"  # 20 '=' characters
    local input_length=${#input}
    local total_length=100
    local prefix_length=${#prefix}

    # Calculate remaining '=' to add after the input
    local suffix_length=$((total_length - prefix_length - input_length))
    local suffix=$(printf '=%.0s' $(seq 1 $suffix_length))

    # Print the output in green
    echo -e "\e[32m${prefix}${input}${suffix}\e[0m"
}

function print_red {
    local input="$1"
    local prefix="========================================"  # 40 '=' characters
    local input_length=${#input}
    local total_length=100
    local prefix_length=${#prefix}

    # Calculate remaining '=' to add after the input
    local suffix_length=$((total_length - prefix_length - input_length))
    local suffix=$(printf '=%.0s' $(seq 1 $suffix_length))

    # Print the output in red
    echo -e "\e[31m${prefix}${input}${suffix}\e[0m"
}

function print_error {
    local input="$1"
    local prefix="ERROR: "  # 40 '=' characters
    local input_length=${#input}
    local total_length=100
    local prefix_length=${#prefix}

    # Calculate remaining '=' to add after the input
    local suffix_length=$((total_length - prefix_length - input_length))
    local suffix=$(printf '=%.0s' $(seq 1 $suffix_length))

    # Print the output in red
    echo -e "\e[31m${prefix}${input}${suffix}\e[0m"
    exit 1
}

function print_info {
    local input="$1"
    local prefix="INFO: "  # 40 '=' characters
    local input_length=${#input}
#    local total_length=100
#    local prefix_length=${#prefix}

    # Calculate remaining '=' to add after the input
#    local suffix_length=$((total_length - prefix_length - input_length))
#    local suffix=$(printf '=%.0s' $(seq 1 $suffix_length))

    # Print the output in red
    echo -e "\e[36m${prefix}${input}\e[0m"
}

# Declare an associative array (Bash 4+ required)
declare -A dataset_map

# Populate the map with dataset names as keys and download links as values
dataset_map=(
    ["1138_bus"]="https://suitesparse-collection-website.herokuapp.com/MM/HB/1138_bus.tar.gz"
    ["dataset2"]="https://example.com/dataset2.tar.gz"
    ["dataset3"]="https://example.com/dataset3.csv"
)

# Function to download a dataset
download_dataset() {
    local dataset_name="$1"
    local url="$2"

    if [ -d "$DLOC/${dataset_name}" ]; then
        echo "$DLOC/${dataset_name} dataset exists."
    else
        rm -f $DLOC/${dataset_name}.tar.gz
        wget -P $DLOC ${url}
        tar -xzf $DLOC/${dataset_name}.tar.gz -C $DLOC
        rm -f $DLOC/${dataset_name}.tar.gz
    fi
}

function checkdatasetanddownload {
    dataset_name=$1
    fullpath=$2
    if [[ ! -f "$fullpath" ]]; then
        print_red "$dataset_name not exists!"
        if [[ -n "${dataset_map[$dataset_name]}" ]]; then
            print_green "DOWNLOAD $datasetname ..."
            url="${dataset_map[$dataset_name]}"
            download_dataset $dataset_name $url
        else
            print_error "$dataset_name NOT IN MY MAP."
        fi
    else
        print_info "FOUND $dataset_name in $fullpath"
    fi


}

function buildandlaunch {
    buildfolder=$1
    mpicmd=$2
    CMAKEARGS=$3
    # now i should be in perlmutter node.
    # first fetch source code from debug node.
    if [[ "$machine" != "debug" ]]; then
        echo "pull latest code from debug node ..."
        snycfromdebug
    else
        echo "on debug node"
    fi
    if [ "$binary" = "multcuda" ]; then
        binary=$(pwd)/$buildfolder/ReleaseTests/MultTimingCUDA
        print_info "Binary fullpath is $binary"
        if [ ! -x "$binary" ]; then
            print_info "configure commands: cmake -S . -B $buildfolder $CMAKEARGS"
            cmake -S . -B $buildfolder $CMAKEARGS
        fi
        cmake --build $buildfolder --target MultTimingCUDA # make sure we have latest binary
        if [ ! -x "$binary" ]; then
            print_error "still can't find $binary after cmake build!"
        else
            print_info "Binary $binary built!"
        fi
    fi

    iter=$6
    testtype=$7
    aname=$8
    bname=$9
    perm=${10}
    func=${11}
    testsr=${12}
    dtype=${13}
    Aname=$DLOC/$aname/$aname.mtx
    Bname=$DLOC/$bname/$bname.mtx

    if [ -z "$mpicmd" ]; then
        print_error "mpi commands empty! check errors!"
    fi

    if [ -z "$binary" ]; then
        print_error "binary empty! check errors!"
    fi
    print_info "binary is $binary"
    checkdatasetanddownload $aname $Aname
    checkdatasetanddownload $bname $Bname

    # ./Benchmarks/combblas/debug.sh debug multcuda 1 test 1138_bus 1138_bus noperm dbuff pt gdld 4
    print_green "LAUNCHING COMMANDS"
    echo $mpicmd $binary $iter $testtype $Aname $Bname $perm $func $testsr $dtype
    print_green "RUNNING BINARY"
    ${mpicmd} $binary $iter $testtype $Aname $Bname $perm $func $testsr $dtype
    print_green "END OF RUNNING"
}

if [ "$machine" = "debug" ]; then
    buildfolder="debug-kkdebug"
    # Check if the last parameter is equal to 4
    if [ $nprocs -ne 4 ]; then
        print_error "Error: MPI number should be 4 in debug node."
    fi
    mpicmd="mpirun -np 4"
    print_green "WE ARE AT DEBUG NODE"
    buildandlaunch "$buildfolder" "$mpicmd" "-DUSE_CUDA=ON -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++" "$@"
elif [ "$machine" = "pmt" ]; then
    buildfolder="release-pmt"
    # single gpu
    # salloc -N 1 -n 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 1 --account=m4293_g --cpus-per-task=16
    # salloc -N 1 -n 4 --qos interactive --time 01:00:00 --constraint gpu --gpus 4 --account=m4293_g --cpus-per-task=16
    # ./Benchmarks/combblas/debug.sh pmt multcuda 1 test 1138_bus 1138_bus noperm dbuff pt gdld 4
    print_green "WE ARE AT PERLMUTTER NODE"
    # Check if the last parameter is equal to 4
    if [[ "$nprocs" -gt 16 ]]; then
        print_error "This is debug script, in pmt, only accept mpi processor <= 16"
    fi
    mpicmd="srun -n $nprocs"

elif [ "$machine" = "delta" ]; then
    buildfolder="release-delta"
    # salloc --account=bdyd-delta-gpu --partition=gpuA100x4-interactive -t 00:30:00 -n 4 -N 1 --gpus-per-node=4
    # ./Benchmarks/combblas/debug.sh dai multcuda 1 test 1138_bus 1138_bus noperm dbuff pt gdld 4
    print_green "WE ARE AT DELTA NODE"


elif [ "$machine" = "dai" ]; then
    buildfolder="release-dai"
    mpicmd="srun -n 4"
    print_green "WE ARE AT DELTA AI NODE"
    # 4 gpus 1 node
    # salloc --account=bdyd-dtai-gh --partition=ghx4-interactive -t 00:30:00 -n 4 -N 1 --gpus-per-node=4
    # ./Benchmarks/combblas/debug.sh dai multcuda 1 test 1138_bus 1138_bus noperm dbuff pt gdld 4
    buildandlaunch "$buildfolder" "$mpicmd" "-DUSE_CUDA=ON -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=CC" "$@"

else
    print_error "INVALID HOSTNAME!"
fi

echo -e """
 _                                             _ _             _                    _   _
| |__   __ _ _ __  _ __  _   _    ___ ___   __| (_)_ __   __ _| |  _   _ _   ___  _(_) | |__   ___  _ __   __ _
| '_ \ / _\` | '_ \| '_ \| | | |  / __/ _ \ / _\` | | '_ \ / _\` | | | | | | | | \ \/ / | | '_ \ / _ \| '_ \ / _\` |
| | | | (_| | |_) | |_) | |_| | | (_| (_) | (_| | | | | | (_| |_| | |_| | |_| |>  <| | | | | | (_) | | | | (_| |
|_| |_|\__,_| .__/| .__/ \__, |  \___\___/ \__,_|_|_| |_|\__, (_)  \__, |\__,_/_/\_\_| |_| |_|\___/|_| |_|\__, |
            |_|   |_|    |___/                           |___/     |___/                                  |___/
"""

