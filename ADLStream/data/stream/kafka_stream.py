"""Implements stream received from a Kafka server."""

from ADLStream.data.stream import BaseStream
from kafka import KafkaConsumer
import json


class KafkaStream(BaseStream):
    """Kafka Stream.

    Stream that consumes messages from a *Kafka* server.

    For more references check:

    * [Apache Kafka](https://kafka.apache.org/)
    * [kafka-python library](https://github.com/dpkp/kafka-python)

    Arguments:
        topic (str): Optional list of topics to subscribe to.
        group_id (str, optional): The name of the consumer group to join for dynamic
            partition assignment (if enabled), and to use for fetching and committing
            offsets. If None, auto-partition assignment (via group coordinator) and
            offset commits are disabled.
            Defaults to None.
        bootstrap_servers (str, optional): 'host[:port]' string (or list of
            'host[:port]' strings) that the consumer should contact to bootstrap initial
            cluster metadata. This does not have to be the full node list. It just needs
            to have at least one broker that will respond to a Metadata API Request.
            Default port is 9092.
            Defaults to `"localhost:9092"`.
        auto_offset_reset (str, optional): A policy for resetting offsets on
            OffsetOutOfRange errors: "earliest" will move to the oldest available
            message, "latest" will move to the most recent. Any other value will raise
            an exception.
            Defaults to "latest".
        value_deserializer (callable): Any callable that takes a raw message value and
            returns a deserialized value.
            Defaults to json ascii decoder.
        timeout (int): number of milliseconds to block during message iteration before
            raising StopIteration (i.e., ending the iterator). In order to block forever
            use `float('inf')`.
            Defaults to 30000.

    """

    def __init__(
        self,
        topic,
        group_id=None,
        bootstrap_servers="localhost:9092",
        outo_offset_reset="latest",
        value_deserializer=lambda m: json.loads(m.decode("ascii")),
        timeout=30000,
        **kwargs
    ):
        super(timeout=timeout, **kwargs)
        self.topic = (topic,)
        self.group_id = group_id
        self.bootstrap_servers = (bootstrap_servers,)
        self.auto_offset_reset = outo_offset_reset
        self.value_deserializer = value_deserializer

        self.kafka_consumer = None

    def start(self):
        super().start()
        self.kafka_consumer = KafkaConsumer(
            self.topic,
            group_id=self.group_id,
            bootstrap_servers=self.bootstrap_servers,
            auto_offset_reset=self.auto_offset_reset,
            consumer_timeout_ms=self.timeout,
            value_deserializer=self.value_deserializer,
        )

    def get_message(self):
        message = next(self.kafka_consumer).value
        return message
