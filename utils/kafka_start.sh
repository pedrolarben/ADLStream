#!/bin/bash

nohup sh /home/pedrolarben/datastream/dcos/volume0/aarcos/kafka_2.11-1.0.0/bin/zookeeper-server-start.sh /home/pedrolarben/datastream/dcos/volume0/aarcos/kafka_2.11-1.0.0/config/zookeeper.properties > $HOME/nohup_zookeeper.out &
sleep 2
nohup sh /home/pedrolarben/datastream/dcos/volume0/aarcos/kafka_2.11-1.0.0/bin/kafka-server-start.sh /home/pedrolarben/datastream/dcos/volume0/aarcos/kafka_2.11-1.0.0/config/server.properties > $HOME/nohup_kafka.out &
sleep 2
