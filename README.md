# Thai Lipreading
This toolset is built with Python.

# Subprojects

## vid2vec
- Video to vector conversion
- Output CSV

## lfa (Lip Features Analysis)
* Analyse with various methods
* Output processed CSV (-processed.csv)
* Output ARFF

## vec2viz
- Vector to viseme conversion
* Call WEKA?

# OBSOLETE tools
## prep-vec
- Generate CSV from JSON
- BUT it only flat the json file with incompatible headers


## Create new project
Use [pyscaffold](https://pypi.org/project/PyScaffold/)
```
conda install -c conda-forge pyscaffold
putup my_project
```