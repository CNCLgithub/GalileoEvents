# Galileo Ramp (v3)

The ramp-ball scenario for galileo

## Contributing

All team members must 

1. Create a branch based off the current master (preferably on their own fork)
2. Add commits to that new branch
3. push the new branch and submit a pull request to master

## Setup

### Config

Simple setups on local hosts should run fine with the `default.conf`.
However, if there are any issues with `singularity` the create a `user.conf`
with correct attributes.

`default.conf` reads as:
```ini
[ENV]
exec:singularity # the path to singularity binary
path:julia-cont  # the path to the singularity container
python:pyenv     # the name of the conda environment
julia_depot:.julia  # the relative path to set JULIA_DEPOT_PATH
mounts:
```
There are additional sections in `default.conf` which are using for 
project organization (`PATHS`).

```ini
[PATHS]
databases:output/databases
traces:output/traces
renders:output/renders
```

> Note: 
The content in the config changes from time to time. If you run into issues after pulling, compare your `user.conf` to `default.conf` to see if any of the keys have changed.

### Environment building

Simply run `setup.sh` in the root of this repo as follows

```bash
chmod +x setup.sh
./setup.sh cont_pull conda julia
```

You will be prompted for sudo when building the container.

`setup.sh` will then create the container at the path specified in the config (`julia-cont` by default).

> NOTE: Like many commands in this setup, variables will be bound to those specified in `user.conf` if present or `default.conf`

In the near future (yell at me if I forget), this script will, by default, attempt to download the container from a hosting service (probably dropbox). In that way, the user will not require sudo (and the container's behavior will be more consistent).

## Runtime


### Interacting with the container

After running `setup.sh`, you can now use `run.sh` to use the environment.

The synatx of `run.sh` is simply:
```bash
./run.sh <command>
```

Where `command` can be any arbitrary bash expression.

For example, you can probe the python version in the conda environment using:
```
>: ./run.sh python3 --version
No user config found, using default
INFO for ENV
        path => julia-cont
        mounts => 
        exec => singularity
        julia_depot => .julia
        python => pyenv
Python 3.6.8 :: Anaconda, Inc.

```
As you can see `./run.sh` first

1. Loads the available config
2. Reads out the config
3. Executes the command

## Interacting with Julia

Getting into the `julia` repl is simply

```
>: ./run.sh julia
```
```
No user config found, using default
INFO for ENV
        path => julia-cont
        mounts => 
        exec => singularity
        julia_depot => .julia
        python => pyenv
               _
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.1.0 (2019-01-21)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
|__/                   |

julia> 
```

Make sure that `JULIA_DEPOT_PATH` is set to that in the config (this should be taken care of by `run.sh`):

```julia
julia> DEPOT_PATH
1-element Array{String,1}:
 ".julia"

julia> 

```

Both `setup.sh` and `run.sh` use the included package info to setup Julia dependencies. Adding packages can be done normally using `Base.pkg`.

>Note:
Some Julia packages (usually stale ones) will attempt to install system level dependencies. This will NOT work in a singularity container as it is immutable. You will have to edit the definition file (`Singularity`) to include this manually.

### Running scripts / experiments

The main method of executing elements within this package are via scripts found in the (queue drum roll) `scripts` directory. If the script has a proper shebang and is executable, congrats, you just need to run:

`./run.sh scripts/my_script.`

ie
```
[galileo-ramp]$ ./run.sh scripts/ramp_profile.py --help
No user config found, using default
pybullet build time: May 15 2019 00:10:22
usage: ramp_profile.py [-h] [--table TABLE TABLE] [--table_steps TABLE_STEPS]
                       [--ramp RAMP RAMP] [--ramp_steps RAMP_STEPS]
                       [--ramp_angle RAMP_ANGLE] [--radius RADIUS]
                       [--friction FRICTION] [--n_ramp N_RAMP] [--slurm]
                       [--batch BATCH] [--debug] [--fresh]
                       mass_file

Evaluates the energy of mass ratios

positional arguments:
  mass_file             CSV file containing mass ratios

optional arguments:
  -h, --help            show this help message and exit
  --table TABLE TABLE   XY dimensions of table. (default: (35, 18))
  --table_steps TABLE_STEPS
                        Number of positions along X-axis. (default: 4)
  --ramp RAMP RAMP      XY dimensions of ramp. (default: (35, 18))
  --ramp_steps RAMP_STEPS
                        Number of positions along X-axis. (default: 4)
  --ramp_angle RAMP_ANGLE
                        ramp angle in degrees (default: 0.5235987755982988)
  --radius RADIUS       Ball radius. (default: 1.5)
  --friction FRICTION   Ball friction (default: 0.4)
  --n_ramp N_RAMP       Number of balls on ramp (default: 1)
  --slurm               Use dask distributed on SLURM. (default: False)
  --batch BATCH         Number of towers to search concurrently. (default: 1)
  --debug               Run in debug (no rejection). (default: False)
  --fresh               Ignore previous profiles (default: False)

```

## Project Layout

The experiment is formatted in the form of a pip-compliant package under `galileo_ramp`.

The package is formatted as follows:

```bash
# describes the GM
/world 
# any nn components
/models
# utilities that do not reasonably belong in previous sections
/utils
```

Each of this sections will have their own documentation.

>Note:
Please maintain hygene between scripts and modules. Any standalone executable should be within scripts. Any piece of code that is imported across several scripts should be incorporated within the project package.
