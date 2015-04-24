# Test barista server
import socket
import struct
import barista


sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('127.0.0.1', 50001))

msg = barista.GRAD_UPDATE
totalsent = 0
while totalsent < barista.MSG_LENGTH:
    sent = sock.send(msg[totalsent:])
    if sent == 0:
        raise RuntimeError("socket connection broken")
    totalsent = totalsent + sent

response = ""
while True:
    chunk = sock.recv(1024)
    if not chunk:
        break
    response += chunk

sock.close()
print response
