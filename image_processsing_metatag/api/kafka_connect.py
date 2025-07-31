#!/usr/bin/env python

from random import choice
import json
from confluent_kafka import Producer, KafkaError, KafkaException # type: ignore
from helper import getConfigInfo

config = {
    # User-specific properties that you must set
    'bootstrap.servers': getConfigInfo('kafka_cluster1.server1')+","+getConfigInfo('kafka_cluster1.server2')+","+getConfigInfo('kafka_cluster1.server3'),
    'socket.timeout.ms': 60000, # 60 seconds timeout for connection

    # Fixed properties
    'acks': 'all'
}
class KafkaConnect:
    def __init__(self):
        pass
    
    @staticmethod
    def callback(err, event):
        if err:
            print(f'Produce to topic {event.topic()} failed for event: {event.key()}')
            return False
        else:
            val = event.value().decode('utf8')
            print(f'{val} sent to partition {event.partition()}.')
            return True

    @staticmethod
    def pushToKafka(data):
        key = data['key']
        print(data)
        try:
            producer = Producer(config)
            value = json.dumps(data["message"])
            topic = data["topic"]
            if producer.produce(topic, value, key, on_delivery=KafkaConnect.callback):
                print(f"Produced event to topic {topic}: key = {key:12} value = {value:12}")
                producer.poll(10000)  # Wait for events to be delivered
                producer.flush()
                return True
            else:
                print(f"Failed to produce event to topic {topic}: key = {key:12} value = {value:12}")
                return False
        except KafkaException as exception:
            kafka_error = exception.args[0]
            if kafka_error.code() == KafkaError._UNKNOWN_PARTITION:
                print("Kafka Topic/Partition Does Not Exist!!")
                return False
            else:
                print(f"Kafka Error: {kafka_error}")
                return False
        except Exception as e:
            print(f"An error occurred: {e}")
            return False
        finally:
            try:
                producer.flush()
                return True
            except Exception as e:
                print(f"Error flushing producer: {e}")
                return False