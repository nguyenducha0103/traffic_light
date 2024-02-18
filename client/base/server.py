import socketio


sio_server = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins=[]
)

sio_app = socketio.ASGIApp(
    socketio_server=sio_server,
    socketio_path='sockets'
)


@sio_server.event
def connect(sid, environ, auth):
    print(f'{sid}: connected')

@sio_server.event
async def event(sid, data):
    await sio_server.emit('message', data)

@sio_server.event
def disconnect(sid):
    print(f'{sid}: disconnected')
