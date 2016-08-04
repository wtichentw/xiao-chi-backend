import os
from prediction import *
from http.server import BaseHTTPRequestHandler, HTTPServer

html =  """
        <html>
            <head>
                <title> Hello World </title> 
            </head> 
            <body>
                <h1> Hello World </h1>
            </body>
        </html>
        """

class StoreHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(bytes(html, 'UTF-8'))
        
    def do_POST(self):
        if self.path == '/xiao-chi-prediction':
            print ("Get Request")
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
            #print ("-----boundary")
            #print (line)
            remainbytes -= len(line)
            if not boundary in line:
                return (False, "Content NOT begin with boundary")

            # Get 2nd line, shoule be content-disposition
            line = self.rfile.readline()
            #print ("-----disposition")
            #print (line)
            remainbytes -= len(line)

            # Get 3rd line, shoule be content-type
            line = self.rfile.readline()
            #print ("-----type")
            #print (line)
            remainbytes -= len(line)
            
            # GEt 4th line, should be content-length
            line = self.rfile.readline()
            remainbytes -= len(line)

            # Get 4th line, shoule be /r/n
            line = self.rfile.readline()
            #print ("-----rn")
            #print (line)
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

            print ("Ready to predict")
            result = run_inference_on_image(image_path)
            print (result)
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(result.encode())

def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(conf.output_graph, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

def run_inference_on_image(imagePath):
    answer = None

    image_data = tf.gfile.FastGFile(imagePath, 'rb').read()
    #image_data = tf.gfile.FastGFile('/home/wtichen/codespace/xiao-chi-backend/pred.jpg', 'rb').read()


    with tf.Session() as sess:

        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)
        top_k = predictions.argsort()[-5:][::-1]  # Getting top 5 predictions
        f = open(conf.output_labels, 'r')
        lines = f.readlines()
        labels = [str(w).replace("\n", "").replace(" ", "-") for w in lines]
        for node_id in top_k:
            human_string = labels[node_id]
            score = predictions[node_id]
            #print('%s (score = %.5f)' % (human_string, score))

        answer = labels[top_k[0]]
        return answer

if __name__ == '__main__':
    print ("Creating graph")
    create_graph()

    print ('Start Server at port', 8080)
    server = HTTPServer(('', 8080), StoreHandler)
    server.serve_forever()

