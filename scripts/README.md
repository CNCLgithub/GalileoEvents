# Workflow

Describes basic project setup


## Generating scenes

Scenes (ramp worlds) can be generating using `scripts/ramp_profile.sh`
This bash file expects a file names `mass_ratios_3.csv` located under
the `PATHS[scenes]` section of the project config (`[default|user].conf`).

An example would be:

```bash
$ cat output/scenes/mass_ratios_3.csv
1,1,1
1/2,1,2

```
## Rendering scenes

```bash
./setup.sh conda
chmod +x scripts/stimuli/create_3ball_scene.sh
./run.sh scripts/stimuli/create_3ball_scene.sh

```
### On local computer to run one trial
```bash
./run.sh python scripts/stimuli/render_stimuli.py output/databases/3ball.hdf5 --mode default --idx 0

```
### Running a batch on Milgram
```
/run.sh python scripts/stimuli/render_stimuli.py output/databases/3ball.hdf5 --mode default --run batch
```
