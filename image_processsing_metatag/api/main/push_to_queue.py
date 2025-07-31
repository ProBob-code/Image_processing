import time
from flask import Blueprint, request
from rabbitmq import RabbitMQ
from kafka_connect import KafkaConnect
from classes.stream_video import StreamVideo
from helper import getConfigInfo
from common import generateRandom

# Define Blueprint
push_to_queue_bp = Blueprint('push_to_queue_bp', __name__, url_prefix='/cp/api')

@push_to_queue_bp.route('/rabbit_push', methods=['POST'])
def rabbit_mq_push():
    if request.method == 'POST':
        post_data = request.get_json()
        queueData = dict()
        queueHost = dict()

        rabbitMq = RabbitMQ()
        if "data" in post_data:
            if "credentials" in post_data["data"]:
                serverInfo = dict()
                serverInfo = post_data["data"]["credentials"]
                queueHost["server"] = serverInfo["server"] if "server" in serverInfo else "192.168.131.114"
                queueHost["port"] = serverInfo["port"] if "port" in serverInfo else "5672"
                queueHost["username"] = serverInfo["username"] if "username" in serverInfo else "admin"
                queueHost["password"] = serverInfo["password"] if "password" in serverInfo else "admin"
                queueHost["host"] = serverInfo["host"] if "host" in serverInfo else "/"
                queueData["credentials"] = queueHost
            else:
                queueHost["server"] = "192.168.131.114"
                queueHost["port"] = "5672"
                queueHost["username"] = "admin"
                queueHost["password"] = "admin"
                queueHost["host"] = "/"
                queueData["credentials"] = queueHost

            if "message" in post_data["data"]:
                queueData["message"] = post_data["data"]["message"]
            else:
                return {"error_code": 1, "status": "ERROR", "message": "No message passsed"}, 500
            
            if "queue_name" in post_data["data"] and post_data["data"]["queue_name"] != "":
                queueData["queue_name"] = post_data["data"]["queue_name"]
            else:
                return {"error_code": 1, "status": "ERROR", "message": "No queue_name passsed"}, 500
        else:
            return {"error_code": 1, "status": "ERROR", "message": "Not valid format passsed"}, 500

        respo = rabbitMq.postQueue(queueData)
        print("Bluprint called")
        if respo:
            return {"error_code": 0, "status": "SUCCESS", "message": respo}, 200
        else:
            return {"error_code": 1, "status": "ERROR", "message": respo}, 500
    else:
        return {"message": "Above request method is not allowd"}, 400

@push_to_queue_bp.route('/v1/lg_video', methods= ['GET'])
def push_to_queue():
    if request.method == 'GET':
        print("#### CALLING push_to_queue ####")
        if 'limit' in request.args and request.args['limit'] != '':
            limit = request.args['limit']
        else:
            limit = 1000
        
        stream_video = StreamVideo
        res = stream_video.PushVideoInQueue(limit)
        return {"error_code": 0, "status": "SUCCESS", "message": res}, 200

@push_to_queue_bp.route('/hello', methods=['GET'])
def hello_w():
    print("Hello World")
    return {"error_code": 0, "status": "HELLO", "message": "Hello DEV"}, 200

@push_to_queue_bp.route('/kafka/push', methods=['POST','GET'])
def push_to_kafka():
    if request.method == 'POST':
        post_data = request.get_json()
        queueData1 = dict()
        if "data" in post_data:
            if "key" in post_data["data"] and post_data["data"]["key"] != "":
                queueData1["key"] = post_data["data"]["key"]

            if "message" in post_data["data"]:
                queueData1["message"] = post_data["data"]["message"]
            else:
                return {"error_code": 1, "status": "ERROR", "message": "No message passed"}, 500
            
            if "topic" in post_data["data"] and post_data["data"]["topic"] != "":
                queueData1["topic"] = post_data["data"]["topic"]
            else:
                return {"error_code": 1, "status": "ERROR", "message": "No topic passed"}, 500
        else:
            return {"error_code": 1, "status": "ERROR", "message": "Not valid format passed"}, 500

        respo = KafkaConnect.pushToKafka(queueData1)
        if respo:
            return {"error_code": 0, "status": "SUCCESS", "message": respo}, 200
        else:
            return {"error_code": 1, "status": "ERROR", "message": "Error in pushing to Kafka", "exc": respo}, 500
    else:
        return {"error_code": 1, "status": "FAILED", "message": "Method not allowed"}, 400

@push_to_queue_bp.route('/v1/video-stream-process', methods= ['POST'])
def push_to_video_init():
    if request.method == 'POST':
        post_data = request.form.to_dict()
        
        rabbit_mq = RabbitMQ()
        queueData = dict()
        queueHost = getConfigInfo('rabbitmq_server1')
        NFS_PATH2 = getConfigInfo('NFS_path.video_input')
        
        files_found = url_found = kafka_push = False
        if request.files:
            files_list = request.files.getlist('files[]')
            # print(files_list)
            # print(len(files_list))
            uploaded_files = []
            for file_path in files_list:
                unique_name = generateRandom(15) + "_" + generateRandom(6) + "_" + str(int(time.time()))
                file_path.filename = unique_name + "." + file_path.filename.split(".")[-1]
                file_path.path = NFS_PATH2 + "/"+ file_path.filename
                file_path.save(file_path.path)
                uploaded_files.append(file_path.path)
                files_found = True
                # print(file_path.path)

            post_data["files"] = uploaded_files
        
        if "url" in post_data and post_data["url"] != "":
            url_found = True
            post_data["video_url"] = post_data["url"]
            
        if "video_url" in post_data and post_data["video_url"] != "":
            url_found = True
            post_data["video_url"] = post_data["video_url"]
            
        if "insta" in post_data and post_data["insta"] == "0":
            # push in `liv` video queue
            queueHost["host"] = "liv"
        elif "insta" in post_data and post_data["insta"] == "2":
            queueHost["host"] = "gen"
        else:
            # push in `content_processing` video queue
            queueHost["host"] = "content_processing"
        
        if "kafka" in post_data and post_data["kafka"] == "1":
            kafka_push = True
        # print(queueHost)
        print(post_data)
        queueData["credentials"] = queueHost
        queueData["message"] = post_data
        queueData["queue_name"] = "video_initiate"
        if kafka_push:
            queueData["topic"] = "video_initiate"
            queueData["key"] = queueHost["host"]
            res = KafkaConnect.pushToKafka(queueData)
            if res:
                return {"error_code": 0, "status": res, "message": "Successfully Pushed in kafka topic"}, 200
            else:
                return {"error_code": 1, "status": "ERROR", "message": "Error in pushing to Kafka"}, 500
        elif url_found or files_found:
            res = rabbit_mq.postQueue(queueData)
            return {"error_code": 0, "status": res, "message": "Successfully Pushed in queue"}, 200
        else:
            return {"error_code": 1, "status": "ERROR", "message": "Invalid Parameters"}, 400
    else:
        return {"error_code": 1, "status": "ERROR", "message": "Method not allowed"}, 400