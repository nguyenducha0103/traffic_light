import socket

s = socket.socket()
s.connect(("10.70.39.40", 5691)) 

# 1024 là số bytes mà client có thể nhận được trong 1 lần
# Phần tin nhắn đầu tiên
while True:
    msg = s.recv(1024)
    print("Recvied ", msg.decode())
    
s.close()
