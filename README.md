# yeast-protein-loc
Detect protein localization shift in yeast cells with neural networks

## Usage

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

### Localization
```
> python run.py localization --help
usage: Run localization and abundance analysis localization [-h] [--features_path FEATURES_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --features_path FEATURES_PATH
                        path to extracted features
```

### Abundance
```
> python run.py abundance --help
usage: Run localization and abundance analysis abundance [-h] [--force] --mode {pairwise,relative}

optional arguments:
  -h, --help            show this help message and exit
  --force, -f           force abundance recalculation even if column is present in pair metainfo
  --mode {pairwise,relative}, -m {pairwise,relative}
                        plot pairwise abundance changes in pairs or aggregated relative abundance changes
```