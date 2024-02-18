import asyncio
import socketio

sio = socketio.AsyncClient()

@sio.event
async def connect():
    print('connection established')


@sio.event
async def disconnect():
    print('disconnected from server')


@sio.on("message")
async def recv2(msg):
    print("da nhan dc r th cho:", msg)

async def main():
    await sio.connect('http://0.0.0.0:5691')
    await sio.wait()


asyncio.run(main())