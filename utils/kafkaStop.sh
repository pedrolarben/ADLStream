#!/bin/bash

sh ../kafka/bin/kafka-server-stop.sh 
sleep 2
sh ../kafka/bin/zookeeper-server-stop.sh 
sleep 2

