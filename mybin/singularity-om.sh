#!/bin/bash

. /etc/os-release
OS=$NAME
if [ "$OS" = "CentOS Linux" ]; then
    source /etc/profile.d/modules.sh
    module add openmind/singularity/3.2.0
fi

singularity "$@"
