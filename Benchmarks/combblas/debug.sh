#!/bin/bash

# test MultTimingCUDA
if [ -z "${WROOT}" ]; then
  echo "WROOT not set."
  exit 1
else
  echo "WROOT is set to $WROOT."
fi

if [ -z "${DLOC}" ]; then
  echo "We use env var DLOC to save dataset, please set it."
  exit 1
else
  echo "DLOC is set to $DLOC."
fi

WORKFOLDER=$WROOT/kk/tfcombblas-yuxidev
echo "WORKFOLDER: $WORKFOLDER"

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
    # Print the output in red
    echo -e "\e[31m${prefix}${input}\e[0m"
    exit 1
}

function print_info {
    local input="$1"
    local prefix="INFO: "  # 40 '=' characters
    # Print the output in red
    echo -e "\e[36m${prefix}${input}\e[0m"
}

# Declare an associative array (Bash 4+ required)
declare -A dataset_map

# Populate the map with dataset names as keys and download links as values
dataset_map=(
    ["1138_bus"]="https://suitesparse-collection-website.herokuapp.com/MM/HB/1138_bus.tar.gz"
    ["bcspwr08"]="https://suitesparse-collection-website.herokuapp.com/MM/HB/bcspwr08.tar.gz"
    ["bcsstk32"]="https://suitesparse-collection-website.herokuapp.com/MM/HB/bcsstk32.tar.gz"
    ["atmosmdd"]="https://suitesparse-collection-website.herokuapp.com/MM/Bourchtein/atmosmodd.tar.gz"
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
            print_green "DOWNLOAD $dataset_name"
            url="${dataset_map[$dataset_name]}"
            download_dataset $dataset_name $url
        else
            print_error "Dataset $dataset_name NOT IN MY MAP."
        fi
    else
        print_info "FOUND Dataset $dataset_name in $fullpath"
    fi
}


function checkmachine {
    # Get the current hostname
    HOSTNAME=$(hostname)
    # Check if the hostname matches the pattern ghXXX.hpcadvisorycouncil.com
    if [[ $HOSTNAME =~ ^gh[0-9]+\.hpcadvisorycouncil\.com$ ]]; then
        echo "thea"
    elif [[ $HOSTNAME =~ ^kkdebug$ ]]; then
        echo "debug"
    elif [[ $HOSTNAME =~ ^gpua[0-9]+\.delta\.ncsa\.illinois\.edu$ ]]; then
        echo "delta"
    elif [[ $HOSTNAME =~ ^gh[0-9]+\.hsn\.cm\.delta\.internal\.ncsa\.edu$ ]]; then
        echo "dai"
    elif [[ $HOSTNAME =~ ^nid[0-9]+$ ]]; then
        echo "pmt"
    else
        echo "None"
    fi
}

function buildandlaunch {
    buildfolder=$1
    mpicmd=$2
    ompthreads=$3
    CMAKEARGS=$4
    # now i should be in perlmutter node.
    if [ "$binary" = "multcuda" ]; then
        binary=$WORKFOLDER/$buildfolder/ReleaseTests/MultTimingCUDA
        print_info "Binary fullpath is $binary"
        if [ ! -x "$binary" ]; then
            print_info "configure commands: cmake -G Ninja -S $WORKFOLDER -B $buildfolder $CMAKEARGS"
            cmake -G Ninja -S $WORKFOLDER -B $buildfolder $CMAKEARGS || { print_error "CMake config failed."; }
        fi
        cmake --build $buildfolder --target MultTimingCUDA -j16 || { print_error "Cmake build failed."; }
        if [ ! -x "$binary" ]; then
            print_error "I still can't find $binary after cmake build!"
        else
            print_info "Binary $binary built!"
        fi
        iter=$7
        commtest=$8
        aname=$9
        bname=${10}
        Aname=$DLOC/$aname/$aname.mtx
        Bname=$DLOC/$bname/$bname.mtx
        echo "aname" $aname "bname" $bname
        if [ -z "$mpicmd" ]; then
            print_error "mpi commands empty! check errors!"
        fi

        if [ -z "$binary" ]; then
            print_error "binary empty! check errors!"
        fi
        print_info "binary is $binary"
        checkdatasetanddownload $aname $Aname
        checkdatasetanddownload $bname $Bname

        # ./Benchmarks/combblas/debug.sh debug multcuda 1 test 1138_bus 1138_bus noperm dbuff pt double dcsc 4
        print_green "LAUNCHING COMMANDS"
        echo -e "OMP_NUM_THREADS=$ompthreads \\
        $mpicmd $binary \\
        --Iter $iter --COMMTEST $commtest \\
        --Aname $Aname \\
        --Bname $Bname "
        print_green "RUNNING BINARY"
        export OMP_NUM_THREADS=$ompthreads
        ${mpicmd} $binary --Iter $iter --COMMTEST $commtest \
        --Aname $Aname \
        --Bname $Bname
        print_green "END OF RUNNING"
    fi


    if [ "$binary" = "spgemmcuda" ]; then
        binary=$WORKFOLDER/$buildfolder/ReleaseTests/SpGEMMCUDA
        print_info "Binary fullpath is $binary"
        if [ ! -x "$binary" ]; then
            print_info "configure commands: cmake -G Ninja -S $WORKFOLDER -B $buildfolder $CMAKEARGS"
            cmake -G Ninja -S $WORKFOLDER -B $buildfolder $CMAKEARGS || { print_error "CMake config failed."; }
        fi
        cmake --build $buildfolder --target SpGEMMCUDA -j16 || { print_error "Cmake build failed."; }
        if [ ! -x "$binary" ]; then
            print_error "I still can't find $binary after cmake build!"
        else
            print_info "Binary $binary built!"
        fi
        iter=$7
        testtype=$8
        aname=$9
        bname=${10}
        perm=${11}
        func=${12}
        testsr=${13}
        dtype=${14}
        ltype=${15}
        Aname=$DLOC/$aname/$aname.mtx
        Bname=$DLOC/$bname/$bname.mtx
        echo "aname" $aname "bname" $bname
        if [ -z "$mpicmd" ]; then
            print_error "mpi commands empty! check errors!"
        fi

        if [ -z "$binary" ]; then
            print_error "binary empty! check errors!"
        fi
        print_info "binary is $binary"
        checkdatasetanddownload $aname $Aname
        checkdatasetanddownload $bname $Bname

        # ./Benchmarks/combblas/debug.sh debug multcuda 1 test 1138_bus 1138_bus noperm dbuff pt double dcsc 4
        print_green "LAUNCHING COMMANDS"
        echo -e "OMP_NUM_THREADS=$ompthreads \\
        $mpicmd $binary \\
        --Iter $iter --Testype $testtype \\
        --Aname $Aname \\
        --Bname $Bname \\
        --Perm $perm \\
        --Func $func \\
        --SR $testsr --Dtype $dtype --Ltype $ltype"
        print_green "RUNNING BINARY"
        export OMP_NUM_THREADS=$ompthreads
        ${mpicmd} $binary --Iter $iter --Testtype $testtype \
        --Aname $Aname \
        --Bname $Bname \
        --Perm $perm --Func $func --SR $testsr --Dtype $dtype --Ltype $ltype
        print_green "END OF RUNNING"
    fi


}


# Check if hostname is provided as the first argument
if [ -z "$1" ]; then
  echo "Please provide a testbed environment as the first argument (debug, pmt, delta, dai)."
  exit 1
fi

machine="$1"
# retrive machine
if [ "$machine" = "auto" ]; then
    machine=$(checkmachine)
fi

# Validate the machine using a case statement
case "$machine" in debug|pmt|delta|dai|thea)
        echo "Hostname accepted: $machine"
    ;;
    *)
        print_error "Invalid machine."
    ;;
esac
# Branch for each hostname type with separate logic
binary=$2
# Get the last parameter, last parameters is mpi processor numbers
last_param="${!#}"
# Check if the last parameter is a number
if [[ ! "$last_param" =~ ^[0-9]+$ ]]; then
    print_error "Error: Last parameter '$last_param' is not a number."
fi

args=("$@")
second_last_param="${args[$(($# - 2))]}"
# Check if the last parameter is a number
if [[ ! "$second_last_param" =~ ^[0-9]+$ ]]; then
    print_error "Error: Second last parameter '$second_last_param' is not a number."
fi


nprocs=$last_param
ompthreads=$second_last_param

if [ "$machine" = "debug" ]; then
    buildfolder="build"
    # Check if the last parameter is equal to 4
    if [ $nprocs -ne 4 ]; then
        print_error "Error: MPI number should be 4 in debug node."
    fi
    if [ $ompthreads -ne 2 ]; then
        print_error "Error: OpenMP number should be 2 in debug node."
    fi
    mpicmd="mpirun -np 4"
    print_green "WE ARE AT DEBUG NODE"
    buildandlaunch "$buildfolder" "$mpicmd" "$ompthreads" "-DACCELERATOR=cuda -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++" "$@"
elif [ "$machine" = "pmt" ]; then
    print_green "WE ARE AT PERLMUTTER NODE"
    buildfolder="release-gpu"
    # 1 gpu run
    # salloc -N 1 -n 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 1 --account=m4293_g --cpus-per-task=32
    # 4 gpus run
    # salloc -N 1 -n 4 --qos interactive --time 01:00:00 --constraint gpu --gpus 4 --account=m4293_g --cpus-per-task=32
    # $WORKFOLDER/Benchmarks/combblas/debug.sh pmt multcuda 1 test 1138_bus 1138_bus noperm dbuff pt gdld 4
    # Check if the last parameter is equal to 4
    if [[ "$nprocs" -gt 16 ]]; then
        print_error "This is debug script, in pmt, only accept mpi processor <= 16"
    fi
    if [[ "$nprocs" -eq 4 ]]; then
        # mpicmd="srun -t 00:30:00 -N 1 -n 4 -c 32 --cpu-bind=cores -q interactive -C gpu --gpus 4 --account=m4293_g";
        mpicmd="srun";
    elif [[ "$nprocs" -eq 16 ]]; then
        # mpicmd="srun -t 00:30:00 -N 4 -n 16 -c 32 --cpu-bind=cores -q interactive -C gpu --gpus 4 --account=m4293_g";
        mpicmd="srun";
    fi
    echo "mpicmd configure: $mpicmd"
    buildandlaunch "$buildfolder" "$mpicmd" "$ompthreads" "-DACCELERATOR=cuda -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=CC" "$@"

elif [ "$machine" = "delta" ]; then
    # salloc --account=bdyd-delta-gpu --partition=gpuA100x4-interactive -t 00:30:00 -n 4 -N 1 --gpus-per-node=4
    # ./Benchmarks/combblas/debug.sh dai multcuda 1 test 1138_bus 1138_bus noperm dbuff pt gdld 4
    print_green "WE ARE AT DELTA NODE"
    buildfolder="release-gpu"
    mpicmd="srun -n $nprocs"
    if [[ "$nprocs" -gt 16 ]]; then
        print_error "This is debug script, in delta, only accept mpi processor <= 16"
    fi
    buildandlaunch "$buildfolder" "$mpicmd" "-DACCELERATOR=cuda -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++" "$@"

elif [ "$machine" = "dai" ]; then
    # 4 gpus 1 node
    # salloc --account=bdyd-dtai-gh --partition=ghx4-interactive -t 00:30:00 -n 4 -N 1 --gpus-per-node=4
    # ./Benchmarks/combblas/debug.sh dai multcuda 1 test 1138_bus 1138_bus noperm dbuff pt gdld 4
    print_green "WE ARE AT DELTA AI NODE"
    buildfolder="release-dai"
    mpicmd="srun -n $nprocs"
    if [[ "$nprocs" -gt 16 ]]; then
        print_error "This is debug script, in delta-ai, only accept mpi processor <= 16"
    fi

    buildandlaunch "$buildfolder" "$mpicmd" "-DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=CC" "$@"

elif [ "$machine" = "thea" ]; then
    buildfolder="release-gpu"
    mpicmd="OMP_NUM_THREADS=$second_last_param mpirun --mca smsc ^knem -np 4"
    print_green "WE ARE AT THEA NODE"
    # Check if the last parameter is equal to 4
    if [ $nprocs -ne 4 ]; then
        print_error "Error: MPI number should be 4 in thea node."
    fi
    # 4 gpus 1 node
    # salloc -n 4 -N 4 -p gh -t 00:30:00
    # ./Benchmarks/combblas/debug.sh thea multcuda 1 test 1138_bus 1138_bus noperm dbuff pt gdld 4
    buildandlaunch "$buildfolder" "$mpicmd" "-DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++" "$@"

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