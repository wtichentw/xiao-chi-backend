from prediction import *
from os import curdir
from os.path import join as pjoin
from http.server import BaseHTTPRequestHandler, HTTPServer

class StoreHandler(BaseHTTPRequestHandler):
    image_path = pjoin(curdir, 'pred.jpg')

    def do_POST(self):
        if self.path == '/pred':
            content_type = self.headers['content-type']
            if not content_type:
                return (False, "Content-Type header doesn't contain boundary")
            boundary = content_type.split("=")[1].encode()

            remainbytes = int(self.headers['content-length'])

            line = self.rfile.readline()
            remainbytes -= len(line)
            if not boundary in line:
                return (False, "Content NOT begin with boundary")
            line = self.rfile.readline()
            remainbytes -= len(line)

            line = self.rfile.readline()
            remainbytes -= len(line)
            line = self.rfile.readline()
            remainbytes -= len(line)
            try:
                out = open(self.image_path, 'wb')
            except IOError:
                return (False, "Can't create file to write, do you have permission to write?")
                    
            preline = self.rfile.readline()
            remainbytes -= len(preline)
            while remainbytes > 0:
                line = self.rfile.readline()
                remainbytes -= len(line)
                if boundary in line:
                    preline = preline[0:-1]
                    if preline.endswith(b'\r'):
                        preline = preline[0:-1]
                    out.write(preline)
                    out.close()
                else:
                    out.write(preline)
                    preline = line

            print (run_inference_on_image(self.image_path))


if __name__ == '__main__':
    server = HTTPServer(('', 8080), StoreHandler)
    print ('Start Server at port', 8080)
    server.serve_forever()

