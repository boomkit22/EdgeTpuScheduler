from asyncore import write
import os
import time
import random
from urllib import request

from client1 import FromServer, ToServer
import threading



############Initialize#############
def Initialize():
    Model = ['EfficientNet_S', 'MobileNetV3', 'ResNet']
    pid = os.getpid()
    model = 'EfficientNet_S'
    path = './Pipe/listenPipe'
    msg = '{} {}'.format(str(pid), model).encode()

    listenPipe = os.open(path, os.O_WRONLY)
    os.write(listenPipe, msg)
    os.close(listenPipe)
    

    ToServer = './Pipe/' + str(pid) + 'ToServer_Pp'
    FromServer = './Pipe/' + 'ServerTo'+str(pid) + '_Pp'
    # Open Pipe until Server made Pipe
    while True:
        try:
            FromServer = os.open(FromServer, os.O_RDONLY)
            ToServer = os.open(ToServer, os.O_WRONLY)
            print('succeed')
            return
        except FileNotFoundError:
            print('failed')
            time.sleep(0.000001)
    ############Initialize#############

############## Send Random Image To Server ################

def ReadWrite():
    pid = os.getpid()
    num = 0
    #sec = time.perf_counter()
    while True:
        img = random.choice(os.listdir("./imageDir"))
        request_time = time.perf_counter()
        args = '{} {} {}'.format(str(pid), img, str(request_time)).encode()
        os.write(ToServer, args)
        num = num + 1
        # if time.perf_counter() - sec > 1:
        #     print(num)
        #     num = 0
        #     sec = time.perf_counter()
        print('why')
        readmsg = (os.read(FromServer, 1024)).decode()
        time.sleep(10)


if __name__ == '__main__':
    Initialize()
    ReadWriteThread = threading.Thread(target=ReadWrite,args=())
    ReadWriteThread.start()
    ReadWriteThread.join()




