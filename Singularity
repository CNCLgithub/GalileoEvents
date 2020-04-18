bootstrap: docker
from: ubuntu:18.04

%environment
 export PATH=$PATH:/blender/blender
 export PATH=$PATH:/miniconda/bin
 export PATH=$PATH:/julia-1.3.1/bin

%runscript
  exec bash "$@"

%post
 apt-get update
 apt-get update
 apt-get install -y software-properties-common
 add-apt-repository ppa:jonathonf/ffmpeg-4
 apt-get update --fix-missing
 apt-get install -y  build-essential \
                     graphviz \
                     git \
                     wget \
                     ffmpeg \
                     libglu1 \
                     libxi6 \
                     libc6 \
                     libgl1-mesa-dev \
                     mesa-utils \
                     xvfb \
                     gettext \
                     gettext-base \
                     libgtk-3-dev \
                     libglib2.0-dev \
                     xdg-utils
 apt-get clean

 # Install Julia
 wget "https://julialang-s3.julialang.org/bin/linux/x64/1.3/julia-1.3.1-linux-x86_64.tar.gz" \
      -O "/julia.tar.gz"
 tar xvzf "/julia.tar.gz"
 ln -s /julia-1.3.1/bin/julia /usr/bin/julia
 chmod +x /usr/bin/julia

 # Setup blender
 wget "https://yale.box.com/shared/static/8wsh0mbvvxfds04xwaef034dcocjrq5w.gz" \
      -O /blender.tar.gz
 tar xf /blender.tar.gz
 rm /blender.tar.gz
 mv blender-2.* /blender
 chmod +x blender/blender

 # Setup conda
 wget "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" -O conda.sh
 bash conda.sh -b -p /miniconda
 rm conda.sh


 # Add an sbatch workaround
 echo '#!/bin/bash\nssh -y "$HOSTNAME"  sbatch "$@"'  > /usr/bin/sbatch
 chmod +x /usr/bin/sbatch

 # Add an scancel workaround
 echo '#!/bin/bash\nssh -y "$HOSTNAME"  scancel "$@"'  > /usr/bin/scancel
 chmod +x /usr/bin/scancel

 # Add an srun workaround
 echo '#!/bin/bash\nssh -y "$HOSTNAME"  srun "$@"'  > /usr/bin/srun
 chmod +x /usr/bin/srun
