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

Afterwards, 
