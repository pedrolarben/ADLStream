#!/bin/bash



## Incremental drift
### RBFi-slow
java -cp ~/snap/moa/lib/moa.jar moa.DoTask "
WriteStreamToARFFFile 
    -s (generators.RandomRBFGeneratorDrift 
        -s 0.0001 
        -c 3 
        -a 20) 
    -f /media/hd1/plara/datastream-minerva/datasets_drift/RBFi-slow.arff 
    -m 1000000
"

### RBFi-fast
java -cp ~/snap/moa/lib/moa.jar moa.DoTask "
WriteStreamToARFFFile 
    -s (generators.RandomRBFGeneratorDrift 
        -s 0.001 
        -c 3 
        -a 20) 
    -f /media/hd1/plara/datastream-minerva/datasets_drift/RBFi-fast.arff 
    -m 1000000
"

## Abrupt drift

### RTGa
java -cp ~/snap/moa/lib/moa.jar moa.DoTask "
WriteStreamToARFFFile 
    -s (ConceptDriftStream 
        -s (generators.RandomTreeGenerator 
            -r 1 
            -i 1 
            -c 3 
            -o 0 
            -u 20
            -d 5 
            -l 3) 
        -d (generators.RandomTreeGenerator 
                -r 2 
                -i 2 
                -c 3 
                -o 0 
                -u 20
                -d 7 
                -l 5)
        -p 500000 
        -w 1) 
    -f /media/hd1/plara/datastream-minerva/datasets_drift/RTGa.arff 
    -m 1000000
"

java -cp ~/snap/moa/lib/moa.jar moa.DoTask "
WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.RandomTreeGenerator -c 3 -o 0 -u 50 -d 20 -l 10) -d (generators.RandomTreeGenerator -r 2 -i 2 -c 3 -o 0 -u 50 -d 25 -l 8) -p 500000 -w 1) -f /media/hd1/plara/datastream-minerva/datasets_drift/RTGaD.arff -m 1000000
"

### RTGa3
java -cp ~/snap/moa/lib/moa.jar moa.DoTask "
WriteStreamToARFFFile 
    -s (ConceptDriftStream 
        -s (generators.RandomTreeGenerator 
            -r 1 
            -i 1 
            -c 3 
            -o 0 
            -u 20
            -d 5 
            -l 3) 
        -d (ConceptDriftStream 
            -s (generators.RandomTreeGenerator 
                -r 2 
                -i 2 
                -c 3 
                -o 0 
                -u 20
                -d 7 
                -l 5) 
            -d (ConceptDriftStream 
                -s (generators.RandomTreeGenerator 
                    -r 3
                    -i 3
                    -c 3 
                    -o 0 
                    -u 20
                    -d 7
                    -l 3) 
                -d (generators.RandomTreeGenerator 
                    -r 4 
                    -i 4 
                    -c 3 
                    -o 0 
                    -u 20 
                    -d 5 
                    -l 5) 
                -p 250000 
                -w 1) 
            -p 250000 
            -w 1) 
        -p 250000 
        -w 1) 
    -f /media/hd1/plara/datastream-minerva/datasets_drift/RTGa3.arff 
    -m 1000000
"

java -cp ~/snap/moa/lib/moa.jar moa.DoTask "
WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.RandomTreeGenerator -c 3 -o 0 -u 50 -d 20 -l 10) -d (ConceptDriftStream -s (generators.RandomTreeGenerator -r 2 -i 2 -c 3 -o 0 -u 50 -v 25 -d 8 -l 5) -d (ConceptDriftStream -s (generators.RandomTreeGenerator -r 3 -i 3 -c 3 -o 0 -u 50 -d 30 -l 12) -d (generators.RandomTreeGenerator -r 4 -i 4 -c 3 -o 0 -u 50 -d 20 -l 8) -p 250000 -w 1) -p 250000 -w 1) -p 250000 -w 1) -f /media/hd1/plara/datastream-minerva/datasets_drift/RTGa3D.arff -m 1000000
"

### ARGWa-F1F10
java -cp ~/snap/moa/lib/moa.jar moa.DoTask "
WriteStreamToARFFFile 
	-s (ConceptDriftStream 
		-s generators.AgrawalGenerator 
		-d (generators.AgrawalGenerator 
			-f 10) 
		-p 500000 
		-w 1) 
	-f /media/hd1/plara/datastream-minerva/datasets_drift/ARGWa-F1F10.arff 
	-m 1000000
"

### ARGWa-F2F5F8
java -cp ~/snap/moa/lib/moa.jar moa.DoTask "
WriteStreamToARFFFile 
    -s (ConceptDriftStream 
        -s (generators.AgrawalGenerator 
		-f 2)
        -d (ConceptDriftStream 
            -s (generators.AgrawalGenerator 
		-f 5)
            -d (generators.AgrawalGenerator 
			-f 8)  
            -p 330000 
            -w 1) 
        -p 330000 
        -w 1) 
    -f /media/hd1/plara/datastream-minerva/datasets_drift/ARGWa-F2F5F8.arff 
    -m 1000000
"

### ARGWa-F3F6F3F6
java -cp ~/snap/moa/lib/moa.jar moa.DoTask "
WriteStreamToARFFFile 
    -s (ConceptDriftStream 
        -s (generators.AgrawalGenerator 
		-f 3)
        -d (ConceptDriftStream 
            -s (generators.AgrawalGenerator 
		-f 6)
            -d (ConceptDriftStream
		-s (generators.AgrawalGenerator 
			-f 3)
		-d (generators.AgrawalGenerator 
			-f 6)  
		-p 250000 
		-w 1) 
            -p 250000 
            -w 1) 
        -p 250000 
        -w 1) 
    -f /media/hd1/plara/datastream-minerva/datasets_drift/ARGWa-F3F6F3F6.arff 
    -m 1000000
"

### SEAa-F2F4
java -cp ~/snap/moa/lib/moa.jar moa.DoTask "
WriteStreamToARFFFile 
	-s (ConceptDriftStream 
		-s (generators.SEAGenerator 
			-f 2) 
		-d (generators.SEAGenerator 
			-f 4) 
		-p 500000 
		-w 1) 
	-f /media/hd1/plara/datastream-minerva/datasets_drift/SEAa-F2F4.arff 
	-m 1000000
"


## Gradual drift

### RTGg
java -cp ~/snap/moa/lib/moa.jar moa.DoTask "
WriteStreamToARFFFile 
    -s (ConceptDriftStream 
        -s (generators.RandomTreeGenerator 
            -r 1 
            -i 1 
            -c 3 
            -o 0 
            -u 20
            -d 5 
            -l 3) 
        -d (generators.RandomTreeGenerator 
                -r 2 
                -i 2 
                -c 3 
                -o 0 
                -u 20
                -d 7 
                -l 5)
        -p 500000 
        -w 100000) 
    -f /media/hd1/plara/datastream-minerva/datasets_drift/RTGg.arff 
    -m 1000000
"

java -cp ~/snap/moa/lib/moa.jar moa.DoTask "
WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.RandomTreeGenerator -c 3 -o 0 -u 50 -d 20 -l 10) -d (generators.RandomTreeGenerator -r 2 -i 2 -c 3 -o 0 -u 50 -d 25 -l 8) -p 500000 -w 100000) -f /media/hd1/plara/datastream-minerva/datasets_drift/RTGgD.arff -m 1000000
"

### RTGg3
java -cp ~/snap/moa/lib/moa.jar moa.DoTask "
WriteStreamToARFFFile 
    -s (ConceptDriftStream 
        -s (generators.RandomTreeGenerator 
            -r 1 
            -i 1 
            -c 3 
            -o 0 
            -u 20
            -d 5 
            -l 3) 
        -d (ConceptDriftStream 
            -s (generators.RandomTreeGenerator 
                -r 2 
                -i 2 
                -c 3 
                -o 0 
                -u 20
                -d 7 
                -l 5) 
            -d (ConceptDriftStream 
                -s (generators.RandomTreeGenerator 
                    -r 3
                    -i 3
                    -c 3 
                    -o 0 
                    -u 20
                    -d 7
                    -l 3) 
                -d (generators.RandomTreeGenerator 
                    -r 4 
                    -i 4 
                    -c 3 
                    -o 0 
                    -u 20 
                    -d 5 
                    -l 5) 
                -p 250000 
                -w 100000) 
            -p 250000 
            -w 100000) 
        -p 250000 
        -w 100000) 
    -f /media/hd1/plara/datastream-minerva/datasets_drift/RTGg3.arff 
    -m 1000000
"
java -cp ~/snap/moa/lib/moa.jar moa.DoTask "
WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.RandomTreeGenerator -c 3 -o 0 -u 50 -d 20 -l 10) -d (ConceptDriftStream -s (generators.RandomTreeGenerator -r 2 -i 2 -c 3 -o 0 -u 50 -v 25 -d 8 -l 5) -d (ConceptDriftStream -s (generators.RandomTreeGenerator -r 3 -i 3 -c 3 -o 0 -u 50 -d 30 -l 12) -d (generators.RandomTreeGenerator -r 4 -i 4 -c 3 -o 0 -u 50 -d 20 -l 8) -p 250000 -w 100000) -p 250000 -w 100000) -p 250000 -w 100000) -f /media/hd1/plara/datastream-minerva/datasets_drift/RTGg3D.arff -m 1000000
"


### ARGWg-F1F10
java -cp ~/snap/moa/lib/moa.jar moa.DoTask "
WriteStreamToARFFFile 
	-s (ConceptDriftStream 
		-s generators.AgrawalGenerator 
		-d (generators.AgrawalGenerator 
			-f 10) 
		-p 500000 
		-w 100000) 
	-f /media/hd1/plara/datastream-minerva/datasets_drift/ARGWg-F1F10.arff 
	-m 1000000
"


### ARGWg-F2F5F8
java -cp ~/snap/moa/lib/moa.jar moa.DoTask "
WriteStreamToARFFFile 
    -s (ConceptDriftStream 
        -s (generators.AgrawalGenerator 
		-f 2)
        -d (ConceptDriftStream 
            -s (generators.AgrawalGenerator 
		-f 5)
            -d (generators.AgrawalGenerator 
			-f 8) 
            -p 330000 
            -w 100000) 
        -p 330000 
        -w 100000) 
    -f /media/hd1/plara/datastream-minerva/datasets_drift/ARGWg-F2F5F8.arff 
    -m 1000000
"

### ARGWg-F3F6F3F6
java -cp ~/snap/moa/lib/moa.jar moa.DoTask "
WriteStreamToARFFFile 
    -s (ConceptDriftStream 
        -s (generators.AgrawalGenerator 
            -f 3)
        -d (ConceptDriftStream 
            -s (generators.AgrawalGenerator 
                -f 6)
            -d (ConceptDriftStream
                -s (generators.AgrawalGenerator 
                        -f 3)
                -d (generators.AgrawalGenerator 
                        -f 6)  
                -p 250000 
                -w 100000) 
            -p 250000 
            -w 100000) 
        -p 250000 
        -w 100000) 
    -f /media/hd1/plara/datastream-minerva/datasets_drift/ARGWg-F3F6F3F6.arff 
    -m 1000000
"

### SEAg-F2F4
java -cp ~/snap/moa/lib/moa.jar moa.DoTask "
WriteStreamToARFFFile 
	-s (ConceptDriftStream 
		-s (generators.SEAGenerator 
			-f 2) 
		-d (generators.SEAGenerator 
			-f 4) 
		-p 500000 
		-w 100000) 
	-f /media/hd1/plara/datastream-minerva/datasets_drift/SEAg-F2F4.arff 
	-m 1000000
"
