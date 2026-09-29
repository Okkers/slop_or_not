# Slop or not?
GML machine-text detection task.

### Setup 
create a folder for holding data (preferably call it data/, or add the name to the .gitignore)

To run the repo, install uv => https://docs.astral.sh/uv/getting-started/installation/:
```
uv sync
uv run main.py
```

or through one of the scripts (TBA).

To add packages: 
```
uv add PACKAGE_NAME
```


### Orientation
- data/ => self-explanatory
- data_factory/ => Preprocessing, loading and formatting data 
- exp/ => Model training, validation and testing here
- layers/ => model layers
- models/ => model variants
- scripts/ => drivers
- vis/ => utils for visualiation, metrics, etc.
