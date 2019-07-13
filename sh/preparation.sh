#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

os_system=`uname`
if [ $os_system == "Darwin" ]; then
    realpath() {
        path=`eval echo "$1"`
        folder=$(dirname "$path")
        echo $(cd "$folder"; pwd)/$(basename "$path");
    }
fi

absolute_path=`pwd`/`dirname $0`
models_path=`realpath $absolute_path/../models`
if [ ! -d $models_path ]; then
    mkdir $models_path
fi

wget https://www.dropbox.com/s/d9fyj7ukhhagh4j/hopenet_robust_alpha1.pkl -P $models_path
