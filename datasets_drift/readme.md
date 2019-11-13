# Concept drift datasets
This folder contains three stream datasets which simulates different concept drift for experimental purpose. 
These datasets have been generated sintetically with the MOA tool. 
As all of them have 1M instances, they are not going to be updated to the repository but the scripts to generate them can be found in this document.

## RBFi 
This is a stream generated with the Radial Basis Function (RBF) generator which simulates a stream with **incremental** drift.  

```
WriteStreamToARFFFile 
    -s (generators.RandomRBFGeneratorDrift 
        -s 0.001 
        -c 3 
        -a 200) 
    -f RBFi.arff 
    -m 1000000
```

## RTGa

This stream simulates three **abrupt** drift using the Random Tree generator.

```
WriteStreamToARFFFile 
    -s (ConceptDriftStream 
        -s (generators.RandomTreeGenerator 
            -r 1 
            -i 1 
            -c 3 
            -o 0 
            -u 200
            -d 5 
            -l 3) 
        -d (ConceptDriftStream 
            -s (generators.RandomTreeGenerator 
                -r 2 
                -i 2 
                -c 3 
                -o 0 
                -u 200 
                -d 7 
                -l 5) 
            -d (ConceptDriftStream 
                -s (generators.RandomTreeGenerator 
                    -r 1
                    -i 1
                    -c 3 
                    -o 0 
                    -u 200
                    -d 5
                    -l 3) 
                -d (generators.RandomTreeGenerator 
                    -r 2 
                    -i 2 
                    -c 3 
                    -o 0 
                    -u 200 
                    -d 7 
                    -l 5) 
                -p 250000 
                -w 1) 
            -p 250000 
            -w 1) 
        -p 250000 
        -w 1) 
    -f /path/to/your/dataset/folder/RTGa.arff 
    -m 1000000
```
## RTGa

This stream simulates three **gradual** drift using the Random Tree generator.

```
WriteStreamToARFFFile 
    -s (ConceptDriftStream 
        -s (generators.RandomTreeGenerator 
            -r 1 
            -i 1 
            -c 3 
            -o 0 
            -u 200
            -d 5 
            -l 3) 
        -d (ConceptDriftStream 
            -s (generators.RandomTreeGenerator 
                -r 2 
                -i 2 
                -c 3 
                -o 0 
                -u 200 
                -d 7 
                -l 5) 
            -d (ConceptDriftStream 
                -s (generators.RandomTreeGenerator 
                    -r 1
                    -i 1
                    -c 3 
                    -o 0 
                    -u 200
                    -d 5
                    -l 3) 
                -d (generators.RandomTreeGenerator 
                    -r 2 
                    -i 2 
                    -c 3 
                    -o 0 
                    -u 200 
                    -d 7 
                    -l 5) 
                -p 250000 
                -w 100000) 
            -p 250000 
            -w 100000) 
        -p 250000 
        -w 100000) 
    -f /path/to/your/dataset/folder/RTGg.arff 
    -m 1000000
```



