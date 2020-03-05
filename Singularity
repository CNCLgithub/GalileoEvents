bootstrap:localimage
from:base.sif

%environment
 export PATH=$PATH:/blender/blender
 export PATH=$PATH:/miniconda/bin
 export PATH=$PATH:/julia-1.3.1/bin

%runscript
  exec bash "$@"

%post
 # Add an sbatch workaround
 echo '#!/bin/bash\nssh -y "$HOSTNAME"  "cd $PWD && sbatch \"$@\""'  > /usr/bin/sbatch
 chmod +x /usr/bin/sbatch

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

