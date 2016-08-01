import os
from prediction import *
from http.server import BaseHTTPRequestHandler, HTTPServer

class StoreHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/pred':
            # Set image path
            image_path = os.path.join(os.curdir, 'pred.jpg')

            # Check content type valid
            content_type = self.headers['content-type']
            if not content_type:
                return (False, "Content-Type header doesn't contain boundary")

            # Get boundary
            boundary = content_type.split("=")[1].encode()

            # Get length
            remainbytes = int(self.headers['content-length'])

            # Get 1st line, shoule be boundary
            line = self.rfile.readline()
            remainbytes -= len(line)
            if not boundary in line:
                return (False, "Content NOT begin with boundary")

            # Get 2nd line, shoule be content-disposition
            line = self.rfile.readline()
            remainbytes -= len(line)

            # Get 3rd line, shoule be content-type
            line = self.rfile.readline()
            remainbytes -= len(line)

            # Get 4th line, shoule be /r/n
            line = self.rfile.readline()
            remainbytes -= len(line)

            # Open file
            try:
                out = open(image_path, 'wb')
            except IOError:
                return (False, "Can't create file to write, do you have permission to write?")
                    
            # Save to file 
            # May judge by boundary or remain byte
            while remainbytes > 0:
                line = self.rfile.readline()
                remainbytes -= len(line)
                out.write(line)
            out.close()

            print (run_inference_on_image(image_path))

if __name__ == '__main__':
    server = HTTPServer(('', 8080), StoreHandler)
    print ('Start Server at port', 8080)
    server.serve_forever()

