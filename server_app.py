from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
import configparser
import json
from app import FaceRecognition
from sys import argv

class S(BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.config = configparser.ConfigParser()
        self.config.read("config.ini")
        self.ir_sensor = self.config['REST_IN']['Ir_sensor']
        # self.face_recog = FaceRecognition(self.config)
        super().__init__(*args, **kwargs)

    def do_OPTIONS(self):           
        self.send_response(200, "ok")       
        self.send_header('Access-Control-Allow-Origin', '*')                
        self.send_header("Access-Control-Allow-Credentials", "true")                
        self.send_header('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,PATCH,OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "Content-Range, Content-Disposition, Authorizaion, Access-Control-Allow-Headers, Origin, Accept, X-Requested-With, Content-Type, Access-Control-Request-Method, Access-Control-Request-Headers") 
        self.end_headers()
        response = {
            'error_code': 0
        }
        response_js = json.dumps(response)
        self.wfile.write(response_js.encode('utf-8'))

    def _set_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

    def start_face_recog(self, data_dict):
        response = {
            'error_code': 0
        }
        response_js = json.dumps(response)
        self._set_response()
        self.wfile.write(response_js.encode('utf-8'))
        global face_recog
        ret = face_recog.serve()
        if ret == 2:
            print("terminated by user")
        elif ret == 1:
            print("camera error")

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        # print(post_data)
        data_dict = json.loads(post_data.decode('utf-8'))
        # data_dict = json.loads(post_data) # use this if no utf encode is used
        print(content_length)
        if self.path == self.ir_sensor:
            self.start_face_recog(data_dict)

def run(server_class=HTTPServer, handler_class=S, port=9999):
    logging.basicConfig(level=logging.INFO)
    server_address = ('localhost', port)
    print(server_address)
    global face_recog
    face_recog = FaceRecognition()
    httpd = server_class(server_address, handler_class)
    logging.info('Starting httpd...\n')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    logging.info('Stopping httpd...\n')


if __name__ == '__main__':
    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()