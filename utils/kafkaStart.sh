#!/bin/bash

nohup sh ../kafka/bin/zookeeper-server-start.sh ../kafka/config/zookeeper.properties > $HOME/nohup_zookeeper.out &
sleep 2
nohup sh ../kafka/bin/kafka-server-start.sh ../kafka/config/server.properties > $HOME/nohup_kafka.out &
sleep 2

