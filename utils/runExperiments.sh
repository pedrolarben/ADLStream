#!/bin/bash

echo "DELETING STREAMS FROM KAFKA"
sh ./kafkaDeleteTopics.sh

echo "SENDING STREAMS TO KAFKA"
sh ./StartMultiKafkaProducer.sh

echo "RUN ADLStream EXPERIMENTS"
sh ./StartKerasParallelConsumersForDrift.sh

echo "RUN MOA EXPERIMENTS"
sh ./StartMOAConsumers.sh

echo "EXPERIMENTS DONE"
