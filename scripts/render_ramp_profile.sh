#!/bin/bash


. load_config.sh

scenes=$(find "${PATHS[scenes]}" -mindepth 1 -maxdepth 1 -type d)
singleramp=($(echo "$scenes" | grep -e "1_[0-9]" | sed 's/:.*//'))
doubleramp=($(echo "$scenes" | grep -e "2_[0-9]" | sed 's/:.*//'))
positions=(0 3 6 9 12 15 18)

single="${singleramp[0]}"
double="${doubleramp[0]}"
echo $single
echo $double
# Pick the first scene
for pos in "${positions[@]}";do
    out="profile/$(basename ${single})_$pos"
    spath="$single/${pos}_0.json"
    ./run.sh scripts/simple_render.py --src "$spath" --mode "motion" --out "$out"
    out="profile/$(basename ${double})_$pos"
    spath="${double}/${pos}_0.json"
    ./run.sh scripts/simple_render.py --src "$spath" --mode "motion" --out "$out"
done

dest="${PATHS[renders]}/profile"
echo "$dest"
find $dest -maxdepth 4 -name 0.png | \
    awk -F'/' '{print $0, $(NF-3)}' | \
    while read orig target; do cp "$orig" "${dest}/${target}.png"; done
