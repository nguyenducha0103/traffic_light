import asyncio
import socketio

sio = socketio.AsyncClient()

@sio.event
async def connect():
    print('client 1 connection established')

@sio.event
async def message(data):
    print('message received with ', data)

@sio.event
async def disconnect():
    print('disconnected from server')

async def main():
    await sio.connect('http://0.0.0.0:5691')
    await sio.wait()

asyncio.run(main())