import os
import redis
from flask import Flask, Blueprint, request, jsonify

redis_bp = Blueprint('redis_bp', __name__, url_prefix='/cp/redis-api')


# --- Connection Setup ---
# The hostname 'my-dragonfly-cluster.dragonfly-operator-system.svc.cluster.local' is the internal
# Kubernetes DNS address for your Dragonfly service. We get this from an environment variable.
# The default is provided for local testing if the environment variable isn't set.
dragonfly_host = os.environ.get('DRAGONFLY_HOST', 'my-dragonfly-cluster.dragonfly-operator-system.svc.cluster.local')

try:
    # It's recommended to use the Redis client with decode_responses=True
    # to get strings back from Dragonfly instead of bytes.
    r = redis.Redis(host=dragonfly_host, port=6379, decode_responses=True)
    # Ping the server to ensure a connection is established.
    r.ping()
    print(f"Successfully connected to Dragonfly at {dragonfly_host}")
except redis.exceptions.ConnectionError as e:
    print(f"Error connecting to Dragonfly: {e}")
    # In a real app, you might want to handle this more gracefully,
    # for now, we'll print the error and the app might fail to start properly.
    r = None

# --- API Endpoints ---

@redis_bp.route('/set', methods=['POST'])
def set_value():
    """
    Sets a key-value pair in Dragonfly.
    Expects a JSON payload with 'key' and 'value'.
    e.g., curl -X POST -H "Content-Type: application/json" -d '{"key": "mykey", "value": "hello"}' http://<Node_IP>:<Node_Port>/set
    """
    if not r:
        return jsonify({"error": "No connection to Dragonfly"}), 500

    try:
        data = request.get_json()
        key = data.get('key')
        value = data.get('value')

        if not key or value is None:
            return jsonify({"error": "Missing 'key' or 'value' in request body"}), 400

        r.set(key, value)
        return jsonify({"status": "success", "key": key, "value": value}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@redis_bp.route('/get/<string:key>', methods=['GET'])
def get_value(key):
    """
    Gets a value by its key from Dragonfly.
    e.g., curl http://<Node_IP>:<Node_Port>/get/mykey
    """
    if not r:
        return jsonify({"error": "No connection to Dragonfly"}), 500

    try:
        value = r.get(key)
        if value is None:
            return jsonify({"error": "Key not found"}), 404
        return jsonify({"key": key, "value": value})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@redis_bp.route('/purge', methods=['DELETE'])
def purge_all():
    """
    Deletes all keys from the current database. Use with caution!
    e.g., curl -X DELETE http://<Node_IP>:<Node_Port>/purge
    """
    if not r:
        return jsonify({"error": "No connection to Dragonfly"}), 500

    try:
        r.flushdb()
        return jsonify({"status": "success", "message": "All keys have been purged."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500