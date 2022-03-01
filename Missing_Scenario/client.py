from asyncore import write
from operator import mod
import os
import time
import random
import threading
from multiprocessing import Process
from urllib import request

# Model = ['EfficientNet_L', 'EfficientNet_M',EfficientNet_S,'MobileNet_v1']

# Global Variable

#


class Client:
    def __init__(self,model_name,period):
        self.FromServer = 0
        self.ToServer = 0
        self.pid = 0
        self.model = model_name
        self.period = period

    def Initialize(self):

        self.pid = os.getpid()
        
        path = './Pipe/listenPipe'
        msg = '{} {}'.format(str(self.pid), self.model).encode()

        listenPipe = os.open(path, os.O_WRONLY)
        os.write(listenPipe, msg)
        os.close(listenPipe)

        ToServerPath = './Pipe/' + str(self.pid) + 'ToServer_Pp'
        FromServerPath = './Pipe/' + 'ServerTo'+str(self.pid) + '_Pp'

        while True:
            try:
                self.FromServer = os.open(FromServerPath, os.O_RDONLY)
                self.ToServer = os.open(ToServerPath, os.O_WRONLY)
                print('succeed')
                return
            except FileNotFoundError:
                print('failed')
                time.sleep(0.000001)

    def Write(self):
        # 1초에 몇번 보내는지 출력
        num = 0
        sendPerSec = time.perf_counter()
        ##############
        start = time.perf_counter()
        iter = 0
        segment_num = 3
        while True:
            # 주기적으로 서버로 write
            time.sleep(1e-9)
            if time.perf_counter() - start > self.period * iter:
                # img = random.choice(os.listdir("./imageDir"))
                img = "n0153282900000036.jpg"
                request_time = time.perf_counter()
                args = '{} {} {} {}'.format(str(self.pid), img, str(
                    request_time), str(self.period)).encode()
                os.write(self.ToServer, args)
                # print(time.time()* 1000)
                num = num + 1
                iter = iter + 1

            # DEBUG #초당 몇번 서버로 write 하는지 print
            if time.perf_counter() - sendPerSec > 1:
                # print(num)
                num = 0
                sendPerSec = time.perf_counter()

    def Read(self):
        # print('Read From Server {}'.format(FromServer))
        init_msg =  (os.read(self.FromServer, 1024)).decode()
        if init_msg == 'complete':
              th_Write = threading.Thread(target=self.Write, args=())
              th_Write.start()
        
        print(time.perf_counter())
        while True:
            readMsg = (os.read(self.FromServer, 1024)).decode()
            # if readMsg:
                # print(readMsg)

def ClientProcess(model_name, period):
    client = Client(model_name, period)
    client.Initialize()
    th_Read = threading.Thread(target=client.Read, args=())   
    th_Read.start()
    th_Read.join()
    
    
# model = 'EfficientNet_M'
if __name__ == '__main__':
     EFNet_M = Process(target=ClientProcess, args=('EfficientNet_M',0.20))
     EFNet_S = Process(target=ClientProcess, args =('EfficientNet_S',0.08))
     EFNet_M.start()
     EFNet_S.start()
     EFNet_M.join()
     EFNet_S.join()
        

    #os.system('rm ./Pipe/*')
    # listenProcess = Process(target=Listen, args=())
    # scheduler = Process(target=ScheduleAndRun, args=())
    # th_Write.start()
    # th_Read.start()
