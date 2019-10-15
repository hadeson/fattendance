import configparser, json, requests
import time
from datetime import datetime

config = configparser.ConfigParser()
config.read("config.ini")
server_ip = "http://localhost:9999"
ir_sensor = server_ip + config['REST_IN']['Ir_sensor']
print(ir_sensor)

def human_detected():
    body = {
        'time': '1234',
    }

    header = {"Content-type": "application/x-www-form-urlencoded",
            "Accept": "text/plain"} 
    body_json = json.dumps(body)
    r = requests.post(ir_sensor, data=body_json, headers=header)
    print(r.json())
    
human_detected()