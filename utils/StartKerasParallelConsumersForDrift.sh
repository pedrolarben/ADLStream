#!/bin/bash

#topics=(streams_RTGgD streams_RTGaD streams_RTGg3D streams_RTGa3D )
streams_ARGWg-F2F5F8 streams_ARGWa-F3F6F3F6 streams_ARGWa-F2F5F8 streams_ARGWg-F1F4 streams_RBFi-slow streams_RBFi-fast streams_RTGa3 streams_RTGa streams_RTGg3 streams_SEAg-F2F4 streams_SEAa-F2F4 streams_ARGWa-F1F4 streams_ARGWg-F3F6F3F6 streams_LED-4) 

for topic in ${topics[@]}; do
	echo "Consuming $topic"

	#python ../projects/kafkapy/DSClassificationKerasParallelConsumer.py --topic $topic --from_beginning --bootstrap_servers localhost:9092 --debug True --two_gpu True --batch_size 90 --num_batches_fed 60 --output_path ../../DSClassificationResults/ConceptDriftExperimentsResults/ --model cnn1
	#python ../projects/kafkapy/DSClassificationKerasParallelConsumer.py --topic $topic --from_beginning --bootstrap_servers localhost:9092 --debug True --two_gpu True --batch_size 90 --num_batches_fed 10 --output_path ../../DSClassificationResults/ConceptDriftExperimentsResults/ --model cnn1
	#python ../projects/kafkapy/DSClassificationKerasParallelConsumer.py --topic $topic --from_beginning --bootstrap_servers localhost:9092 --debug True --two_gpu True --batch_size 40 --num_batches_fed 60 --output_path ../../DSClassificationResults/ConceptDriftExperimentsResults/ --model cnn1
	#python ../projects/kafkapy/DSClassificationKerasParallelConsumer.py --topic $topic --from_beginning --bootstrap_servers localhost:9092 --debug True --two_gpu True --batch_size 40 --num_batches_fed 10 --output_path ../../DSClassificationResults/ConceptDriftExperimentsResults/ --model cnn1
	
#	python ../projects/kafkapy/DSClassificationKerasParallelConsumer.py --topic $topic --from_beginning --bootstrap_servers localhost:9092 --debug True --two_gpu True --batch_size 90 --num_batches_fed 60 --output_path ../../DSClassificationResults/ConceptDriftExperimentsResults/ --model cnn2
	#python ../projects/kafkapy/DSClassificationKerasParallelConsumer.py --topic $topic --from_beginning --bootstrap_servers localhost:9092 --debug True --two_gpu True --batch_size 90 --num_batches_fed 10 --output_path ../../DSClassificationResults/ConceptDriftExperimentsResults/ --model cnn2
	python ../projects/kafkapy/DSClassificationKerasParallelConsumer.py --topic $topic --from_beginning --bootstrap_servers localhost:9092 --debug True --two_gpu True --batch_size 40 --num_batches_fed 60 --output_path ../../DSClassificationResults/ConceptDriftExperimentsResults/ --model cnn2
	#python ../projects/kafkapy/DSClassificationKerasParallelConsumer.py --topic $topic --from_beginning --bootstrap_servers localhost:9092 --debug True --two_gpu True --batch_size 40 --num_batches_fed 10 --output_path ../../DSClassificationResults/ConceptDriftExperimentsResults/ --model cnn2

#	python ../projects/kafkapy/DSClassificationKerasParallelConsumer.py --topic $topic --from_beginning --bootstrap_servers localhost:9092 --debug True --two_gpu True --batch_size 90 --num_batches_fed 60 --output_path ../../DSClassificationResults/ConceptDriftExperimentsResults/ --model mlp1
	#python ../projects/kafkapy/DSClassificationKerasParallelConsumer.py --topic $topic --from_beginning --bootstrap_servers localhost:9092 --debug True --two_gpu True --batch_size 90 --num_batches_fed 10 --output_path ../../DSClassificationResults/ConceptDriftExperimentsResults/ --model mlp1
#	python ../projects/kafkapy/DSClassificationKerasParallelConsumer.py --topic $topic --from_beginning --bootstrap_servers localhost:9092 --debug True --two_gpu True --batch_size 40 --num_batches_fed 60 --output_path ../../DSClassificationResults/ConceptDriftExperimentsResults/ --model mlp1
	#python ../projects/kafkapy/DSClassificationKerasParallelConsumer.py --topic $topic --from_beginning --bootstrap_servers localhost:9092 --debug True --two_gpu True --batch_size 40 --num_batches_fed 10 --output_path ../../DSClassificationResults/ConceptDriftExperimentsResults/ --model mlp1

#	python ../projects/kafkapy/DSClassificationKerasParallelConsumer.py --topic $topic --from_beginning --bootstrap_servers localhost:9092 --debug True --two_gpu True --batch_size 90 --num_batches_fed 60 --output_path ../../DSClassificationResults/ConceptDriftExperimentsResults/ --model mlp2
	#python ../projects/kafkapy/DSClassificationKerasParallelConsumer.py --topic $topic --from_beginning --bootstrap_servers localhost:9092 --debug True --two_gpu True --batch_size 90 --num_batches_fed 10 --output_path ../../DSClassificationResults/ConceptDriftExperimentsResults/ --model mlp2
#	python ../projects/kafkapy/DSClassificationKerasParallelConsumer.py --topic $topic --from_beginning --bootstrap_servers localhost:9092 --debug True --two_gpu True --batch_size 40 --num_batches_fed 60 --output_path ../../DSClassificationResults/ConceptDriftExperimentsResults/ --model mlp2
	#python ../projects/kafkapy/DSClassificationKerasParallelConsumer.py --topic $topic --from_beginning --bootstrap_servers localhost:9092 --debug True --two_gpu True --batch_size 40 --num_batches_fed 10 --output_path ../../DSClassificationResults/ConceptDriftExperimentsResults/ --model mlp2
	
#	python ../projects/kafkapy/DSClassificationKerasParallelConsumer.py --topic $topic --from_beginning --bootstrap_servers localhost:9092 --debug True --two_gpu True --batch_size 90 --num_batches_fed 60 --output_path ../../DSClassificationResults/ConceptDriftExperimentsResults/ --model mlp3
	#python ../projects/kafkapy/DSClassificationKerasParallelConsumer.py --topic $topic --from_beginning --bootstrap_servers localhost:9092 --debug True --two_gpu True --batch_size 90 --num_batches_fed 10 --output_path ../../DSClassificationResults/ConceptDriftExperimentsResults/ --model mlp3
#	python ../projects/kafkapy/DSClassificationKerasParallelConsumer.py --topic $topic --from_beginning --bootstrap_servers localhost:9092 --debug True --two_gpu True --batch_size 40 --num_batches_fed 60 --output_path ../../DSClassificationResults/ConceptDriftExperimentsResults/ --model mlp3
	#python ../projects/kafkapy/DSClassificationKerasParallelConsumer.py --topic $topic --from_beginning --bootstrap_servers localhost:9092 --debug True --two_gpu True --batch_size 40 --num_batches_fed 10 --output_path ../../DSClassificationResults/ConceptDriftExperimentsResults/ --model mlp3

done

