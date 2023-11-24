# Single-cell imaging of protein dynamics of paralogs reveals mechanisms of gene retention  

## Contents   
    config          Contains configurations
    modules         Contains the python modules
    notebooks       Contains the jupyter notebooks
    notebooks_post  Contains the jupyter notebooks used for the post-processing
    results         Contains the results e.g. plots, figures

## Preprocessing  

Detect protein localization shift in yeast cells with neural networks

### Requirements

```
pip install requirements.txt
```

### Usage

```
> python run.py --help
Run localization and abundance analysis [-h] [--separate_replicates] [--format {pdf,png}]
                                               [--meta_path META_PATH] [--results_path RESULTS_PATH]
                                               [--reduce {mean,median}] [--overwrite]
                                               {localization,abundance} ...

positional arguments:
  {localization,abundance}
    localization        analyze protein localization
    abundance           analyze protein abundance

optional arguments:
  -h, --help            show this help message and exit
  --separate_replicates, -s
                        separate replicates in analysis
  --format {pdf,png}, -t {pdf,png}
                        format to store output
  --meta_path META_PATH
  --results_path RESULTS_PATH
  --reduce {mean,median}
                        how to reduce features and intensity values
  --overwrite, -o       overwrite existing files
```

#### Localization
```
> python run.py localization --help
usage: Run localization and abundance analysis localization [-h] [--features_path FEATURES_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --features_path FEATURES_PATH
                        path to extracted features
```

#### Abundance
```
> python run.py abundance --help
usage: Run localization and abundance analysis abundance [-h] [--force] --mode {pairwise,relative}

optional arguments:
  -h, --help            show this help message and exit
  --force, -f           force abundance recalculation even if column is present in pair metainfo
  --mode {pairwise,relative}, -m {pairwise,relative}
                        plot pairwise abundance changes in pairs or aggregated relative abundance changes
```

## Post-processing  

### Requirements

```
pip install requirements_post.txt
```

### Steps in the analysis:   
`1  ` Pre-processed data  
`1.1` Quality filtering of the data  
`2  ` Features  
`2.1` Filtering of features  
`3  ` Single cell protein abundances  
`3.1` Quantification of single-cell protein abundance  
`3.2` Relative abundance changes  
`3.3` Measuring relative protein abundance changes  
`4  ` Redistribution  
`4.1` Calculation of redistribution scores  
`4.2` Classification of redistribution  
`5  ` Processed data  
`5.1` Combining processed data.  
`6  ` Explanatory features  
`7  ` Protein isoforms  
`7.1` Validation of Cue4 relocalization by imaging  
`7.2` Pre-processing the raw images  
`7.3` Image segmentation  
`7.4` Calculation of the single-cell abundances