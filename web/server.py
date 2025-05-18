import json
from http.server import SimpleHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

class Handler(SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/prever':
            length = int(self.headers.get('content-length', 0))
            data = json.loads(self.rfile.read(length))
            # TODO: integrar com modelo predito
            resultado = {'resultado': 'N/A'}
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(resultado).encode())
        else:
            self.send_error(404)

def run(server_class=HTTPServer, handler_class=Handler):
    server_address = ('', 8000)
    httpd = server_class(server_address, handler_class)
    print('Servidor iniciado em http://localhost:8000')
    httpd.serve_forever()

if __name__ == '__main__':
    run()
