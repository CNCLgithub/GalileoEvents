# /stimuli

Streamlines stimuli generation

## generate_towers.py

```
usage: generate_towers.py [-h] [--out OUT] [--base BASE] n b

Renders the towers in a given directory

positional arguments:
  n            Number of towers to generate
  b            The size of each tower.

optional arguments:
  -h, --help   show this help message and exit
  --out OUT    Path to save renders.
  --base BASE  Path to base tower.

```


## generate_renders.py

```
usage: generate_renders.py [-h] [--src SRC]

Renders the towers in a given directory

optional arguments:
  -h, --help  show this help message and exit
  --src SRC   Path to tower jsons
```

## generate_alternates.py

```
usage: generate_renders.py [-h] [--src SRC]

Renders the towers in a given directory

optional arguments:
  -h, --help  show this help message and exit
  --src SRC   Path to tower jsons

```

## movies.py

```
usage: movies.py [-h] [--src SRC]

Generates movie from scene

optional arguments:
  -h, --help  show this help message and exit
  --src SRC   Path to rendered frames
```
