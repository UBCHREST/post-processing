# Performance Analysis
Contains scripts for scaling analysis of PETSc events utilizing csv outputs.
The directory will search the given path for csv files following the provided naming scheme.
Processes are available for strong, weak, and static scaling plots to be produced.
Performance model fitting capabilities are being developed.

### File Naming Scheme
```
filePrefix_dofSize_processNumber_problemSize.csv
```

### Example input for performance analysis
```
/usr/bin/python3.10 performanceAnalysis.py --path /path/to/scaling_tests/csv_files --name filePrefix --processes 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 --problems [105,15] [149,21] [297,42] --dof 1575 3129 12474 --events Radiation::Initialize Radiation::EvaluateGains
```
