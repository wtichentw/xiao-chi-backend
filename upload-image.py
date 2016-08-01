from os import curdir
from os.path import join as pjoin

from http.server import BaseHTTPRequestHandler, HTTPServer

class StoreHandler(BaseHTTPRequestHandler):
    image_path = pjoin(curdir, 'pred.jpg')

    def do_GET(self):
        if self.path == '/pred.jpg':
            with open(self.store_path) as fh:
                self.send_response(200)
                self.send_header('Content-type', 'text/json')
                self.end_headers()
                self.wfile.write(fh.read().encode())

    def do_POST(self):
        if self.path == '/pred.jpg':
            length = self.headers['content-length']
            data = self.rfile.read(int(length))

            with open(self.store_path, 'w') as fh:
                fh.write(data.decode())

            self.send_response(200)


server = HTTPServer(('', 8080), StoreHandler)
server.serve_forever()
