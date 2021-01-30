# Thai Lipreading
This toolset is built with Python.

# Subprojects

## vid2vec
- Video to vector conversion
- Output CSV

## lfa (Lip Features Analysis)
* Analyse with various methods
* Output processed CSV (-processed.csv)
* Specify method (1,2,3)


## lfa2arff
* specify frames to work
* Output ARFF

## www
- List of visemes, mouth visualisation, example

# Wait List
## v2g
- Mapping viseme to grapheme

# OBSOLETE tools
## vec2viz
- DOUBTFUL any benefit?
- Vector to viseme conversion
* Call WEKA?

## prep-vec
- Generate CSV from JSON
- BUT it only flat the json file with incompatible headers


## Create new project
Use [pyscaffold](https://pypi.org/project/PyScaffold/)
```
conda install -c conda-forge pyscaffold
putup my_project
```