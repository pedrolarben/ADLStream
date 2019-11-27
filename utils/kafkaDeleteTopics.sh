#!/bin/bash

../kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --delete --topic 'streams_.*'
echo 'Topics deleted' 
echo 'This is the list of topics currently in kafka:'
../kafka/bin/kafka-topics.sh --list --bootstrap-server localhost:9092

