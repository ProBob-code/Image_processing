#!/usr/bin/env python
import os
import pika
import sys
from src.libs.helpers import checkKeyExists


class RabbitMQ:
    def __init__(self):
        pass

    def createConnection(self, queue_obj):
        try:
            server = queue_obj['server'] if checkKeyExists(queue_obj, 'server') else ''
            username = queue_obj['username'] if checkKeyExists(queue_obj, 'username') else ''
            password = queue_obj['password'] if checkKeyExists(queue_obj, 'password') else ''
            port = queue_obj['port'] if checkKeyExists(queue_obj, 'port') else ''
            
            heartbeat = int(queue_obj['heartbeat']) if checkKeyExists(queue_obj, 'heartbeat') else None
            blocked_connection_timeout = int(queue_obj['blocked_connection_timeout']) if checkKeyExists(queue_obj, 'blocked_connection_timeout') else None

            credentials = pika.PlainCredentials(username, password)
            if heartbeat != None and blocked_connection_timeout != None:
                return pika.BlockingConnection(pika.ConnectionParameters(server, port, queue_obj['host'] if checkKeyExists(queue_obj, 'host') else '/', credentials, heartbeat=heartbeat, blocked_connection_timeout=blocked_connection_timeout))
            else:    
                return pika.BlockingConnection(pika.ConnectionParameters(server, port, queue_obj['host'] if checkKeyExists(queue_obj, 'host') else '/', credentials))
        except Exception as e:
            print("Something went wrong while connecting to rabbitMQ server!")
            print(str(e))
            exit(1)

    def postQueue(self, queue_obj):
        try:
            rabbit_mq_connection = self.createConnection(queue_obj['credentials'])
            channel = rabbit_mq_connection.channel()
            channel.queue_declare(queue=queue_obj['queue_name'], passive=False, durable=True, exclusive=False, auto_delete=False)
            

            channel.basic_publish(
                exchange='',
                routing_key=queue_obj['queue_name'],
                body=queue_obj['message'],
                properties=pika.BasicProperties(
                    delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE
                ))
            channel.close()
            rabbit_mq_connection.close()
            return True
        except Exception as e:
            print("Something went wrong while pushing data to queue!")
            print(str(e))
            return False
            exit(1)

