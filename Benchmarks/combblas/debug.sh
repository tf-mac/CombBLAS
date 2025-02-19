#!/bin/bash

# test MultTimingCUDA
if [ -z "${DLOC}" ]; then
  echo "We use env var DLOC to save dataset, please set it."
  exit 1
else
  echo "DLOC is set to $DLOC."
fi

# Check if hostname is provided as the first argument
if [ -z "$1" ]; then
  echo "Please provide a testbed environment as the first argument (debug, perlmutter, delta, deltaai)."
  exit 1
fi

machine="$1"

# Validate the machine using a case statement
case "$machine" in debug|perlmutter|delta|deltaai)
        echo "Hostname accepted: $machine"
    ;;
    *)
        echo "Invalid machine. Accepted types are: debug, perlmutter, delta, deltaai."
        exit 1
    ;;
esac


# Branch for each hostname type with separate logic
if [ "$machine" = "debug" ]; then
    echo "Processing debug branch..."
    # Check if test binary is provided
    if [ -z "$2" ]; then
      echo "Please provide a binary as the second argument (multcuda)."
      exit 1
    fi
    binary="$2"
    if [ "$binary" = "multcuda" ]; then
        if [ -x "$(pwd)/build/ReleaseTests/MultTimingCUDA" ]; then
            binary=$(pwd)/build/ReleaseTests/MultTimingCUDA
        else
            echo "MultTimingCUDA not compiled yet."
            exit 1
        fi

    fi
    echo "$binary"
    iter="$3"
    testtype=$4
    aname=$5
    bname=$6
    perm=$7
    func=$8
    Aname=$DLOC/$aname/$aname.mtx
    Bname=$DLOC/$bname/$bname.mtx
    echo "binary" $iter $testtype "$Aname" "$Bname" $perm $func
    cmake --build build --target MultTimingCUDA
    mpirun -np 4 $binary $iter $testtype $Aname $Bname $perm $func


elif [ "$machine" = "perlmutter" ]; then
    echo "Processing perlmutter branch..."
    # Insert perlmutter-specific logic here

elif [ "$machine" = "delta" ]; then
    echo "Processing delta branch..."
    # Insert delta-specific logic here

elif [ "$machine" = "deltaai" ]; then
    echo "Processing deltaai branch..."
    # Insert deltaai-specific logic here

else
    echo "Invalid hostname. Accepted types are: debug, perlmutter, delta, deltaai."
    exit 1
fi