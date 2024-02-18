import socketio
from aiohttp import web
import uvicorn
import fastapi
# create a Socket.IO server
sio = socketio.AsyncServer()

# wrap with a WSGI application
app = fastapi.FastAPI()
sio.attach(app)

@sio.event
def connect(sid, environ, auth):
    print('connect', sid)

@sio.event
def disconnect(sid):
    print('disconnect ', sid)

# @sio.event
# async def msg(data):
#     # print('Server recieve data', data)
#     await sio.emit('message', {'event1':'data'})


if __name__ == '__main__':
    
	# start a thread that will perform motion detection
    uvicorn.run(app, host='0.0.0.0', port=5691, access_log=False)