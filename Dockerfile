FROM traffic_light:23.08
WORKDIR /traffic_light

CMD python3 main.py --host 0.0.0.0 --port 10000