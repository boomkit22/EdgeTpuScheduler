# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Example using PyCoral to classify a given image using an Edge TPU.

To run this code, you must attach an Edge TPU attached to the host and
install the Edge TPU runtime (`libedgetpu.so`) and `tflite_runtime`. For
device setup instructions, see coral.ai/docs/setup.

Example usage:
```
bash examples/install_requirements.sh classify_image.py

python3 examples/classify_image.py \
  --model test_data/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite  \
  --labels test_data/inat_bird_labels.txt \
  --input test_data/parrot.jpg
```
"""

import argparse
from concurrent.futures import process, thread
from inspect import ArgSpec
from sched import scheduler
from sre_constants import SUCCESS
import time
from urllib.request import Request
import numpy as np
import os
from PIL import Image
from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
import threading


lock = threading.Lock()


class NamedPipe:
    request_list = []
    process_dict = {}

    def __init__(self):
        self.path = './Pipe/' + 'listenPipe'

    def make_listen_pipe(self):
        if os.path.exists(self.path):
            os.remove(self.path)
        os.mkfifo(self.path)

        self.listenPipe = os.open(self.path, os.O_RDONLY)

    def make_client_pipe(self):
        i = 0
        client_num = 2
        client_thread_list = []
        # os.system('ls')
        while True:
            readmsg = (os.read(self.listenPipe, 100)).decode()
            if readmsg:
                i = i + 1
                pid, modelName = readmsg.split(' ')
                ServerWritePath = './Pipe/' + 'ServerTo' + str(pid) + '_Pp'
                if os.path.exists(ServerWritePath):
                    os.remove(ServerWritePath)
                os.mkfifo(ServerWritePath)
                ClientWritePath = './Pipe/' + str(pid) + 'ToServer_Pp'
                if os.path.exists(ClientWritePath):
                    os.remove(ClientWritePath)
                os.mkfifo(ClientWritePath)
                client_thread_list.append(threading.Thread(target=self.read_and_write_to_queue, args=(
                    pid, modelName, ServerWritePath, ClientWritePath)))
                if i == client_num:
                    for thread in client_thread_list:
                        thread.start()
                    break

    def read_and_write_to_queue(self, pid, model_name, server_write_path, client_write_path):
        to_client = os.open(server_write_path, os.O_WRONLY)
        from_client = os.open(client_write_path, os.O_RDONLY)
        pid = int(pid)
        NamedPipe.process_dict[pid] = (model_name, to_client)
        
        init_msg = 'complete'.encode()
        os.write(to_client ,init_msg)
        
        while True:
            msg = os.read(from_client, 100).decode()
            if msg:
                try:
                    pid, image_path, request_time, period = msg.split(' ')
                except(ValueError):
                    print(msg)
                pid = int(pid)
                period = float(period)
                request_time = float(request_time)
                lock.acquire()
                # print('pid = {} now = {}'.format(pid,time.perf_counter() * 1000))
                NamedPipe.request_list.append(
                    (pid, image_path, request_time, period))
                lock.release()

    def run(self):
        self.make_listen_pipe()
        self.listen_thread = threading.Thread(
            target=self.make_client_pipe, args=())
        self.listen_thread.start()
        # self.listen_thread.join()


class Interpreter:
    def __init__(self):
        self.model_interpreter_dict = {}
        self.model_path_list = []

    def make_model_path_list(self):
        self.model_path_list = []
        # model = '/home/hun/WorkSpace/coral/pycoral/models/result/SM_1/efficientnet-edgetpu-L_quant_edgetpu.tflite'
        # modelName = 'EfficientNet_L'
        # self.model_path_list.append((modelName, model))

        model = '/home/hun/WorkSpace/coral/pycoral/models/result/SM_Origin/efficientnet-edgetpu-M_quant_edgetpu.tflite'
        modelName = 'EfficientNet_M'
        self.model_path_list.append((modelName, model))

        model = '/home/hun/WorkSpace/coral/pycoral/models/result/SM_Origin/efficientnet-edgetpu-S_quant_edgetpu.tflite'
        modelName = 'EfficientNet_S'
        self.model_path_list.append((modelName, model))

        # model = '/home/hun/WorkSpace/coral/pycoral/models/result/SM_1/mobilenet_v1_0.25_128_quant_edgetpu.tflite'
        # modelName = 'MobileNet_V1'
        # self.model_path_list.append((modelName, model))

    def initialize_dict(self):
        self.make_model_path_list()
        for model in self.model_path_list:
            Interpreter = make_interpreter(model[1])
            Interpreter.allocate_tensors()
            self.model_interpreter_dict[model[0]] = Interpreter

    def initialize_model(self):
        imagePath = './imageDir/n0153282900000005.jpg'

        for model in self.model_interpreter_dict.values():
            size = common.input_size(model)
            image = Image.open(imagePath).convert(
                'RGB').resize(size, Image.ANTIALIAS)

            params = common.input_details(
                model, 'quantization_parameters')
            scale = params['scales']
            zero_point = params['zero_points']
            mean = 128.0
            std = 128.0

            if abs(scale * std - 1) < 1e-5 and abs(mean - zero_point) < 1e-5:
                # Input data does not require preprocessing.
                common.set_input(model, image)
            else:
                # Input data requires preprocessing
                normalized_input = (np.asarray(image) - mean) / \
                    (std * scale) + zero_point
                np.clip(normalized_input, 0, 255, out=normalized_input)
                common.set_input(
                    model, normalized_input.astype(np.uint8))

            model.invoke()
 


            


class Analyzer:
    def __init__(self, Scheduler):
        self.scheduler = Scheduler
        self.prev_request_num = 0
        self.request_num = 0
        self.request_per_sec = 0

    def CountRequest(self):
        while True:
            self.request_num = Scheduler.success + Scheduler.fail
            self.request_per_sec = self.request_num - self.prev_request_num
            self.prev_request_num = self.request_num
            print('SUCCESS : {} FAIL : {}  ReqPerSec : {}'.format(Scheduler.success,
                                                                  Scheduler.fail, self.request_per_sec))
            time.sleep(1)

    def run(self):
        analyzer_thread = threading.Thread(target=self.CountRequest, args=())
        analyzer_thread.start()


class Scheduler:
    success = 0
    fail = 0

    def __init__(self, Interpreter):
        self.interpreter = Interpreter
        self.labels = read_label_file(
            '/home/hun/WorkSpace/coral/pycoral/test_data/labels.txt')

    def schedule(self):
        M = open('Efficient_M', 'w')
        S = open('Efficient_S', 'w')
        while True:
            time.sleep(1e-9)
            while_start = time.perf_counter()
            if len(NamedPipe.request_list) != 0:
                # print(len(request_list))
                ##FIFO##
                lock.acquire()
                NamedPipe.request_list.sort(key=lambda request: request[3])
                who = NamedPipe.request_list.pop(0)
                lock.release()
                ########
                pid = who[0]
                imagePath = './imageDir/' + who[1]
                request_time = who[2]
                period = who[3]
                # print(period)
                deadline = period

                modelName = NamedPipe.process_dict[pid][0]
                ToClient = NamedPipe.process_dict[pid][1]
                # 요청을 받으면 모델과 interpreter를 만들어서
                assigned_interpreter = interpreter.model_interpreter_dict[modelName]

                count = 5
                threshold = 0.0
                top_k = 3

                if common.input_details(assigned_interpreter, 'dtype') != np.uint8:
                    raise ValueError('Only support uint8 input type.')

                size = common.input_size(assigned_interpreter)

                image = Image.open(imagePath).convert(
                    'RGB').resize(size, Image.ANTIALIAS)

                params = common.input_details(
                    assigned_interpreter, 'quantization_parameters')
                scale = params['scales']
                zero_point = params['zero_points']

                mean = 128.0
                std = 128.0

                if abs(scale * std - 1) < 1e-5 and abs(mean - zero_point) < 1e-5:
                    # Input data does not require preprocessing.
                    common.set_input(assigned_interpreter, image)
                else:
                    # Input data requires preprocessing
                    normalized_input = (np.asarray(image) - mean) / \
                        (std * scale) + zero_point
                    np.clip(normalized_input, 0, 255, out=normalized_input)
                    common.set_input(
                        assigned_interpreter, normalized_input.astype(np.uint8))

                for _ in range(1):
                    assigned_interpreter.invoke()
                    classes = classify.get_classes(
                        assigned_interpreter, top_k, threshold)

                msg = ''
                for c in classes:
                    msg = msg + ('%s: %.5f\n' %
                                 (self.labels.get(c.id, c.id), c.score))

                os.write(ToClient, msg.encode())
                response_time = time.perf_counter() - request_time
                # while_end = time.perf_counter() - while_start
                # print(while_end * 1000)
                
                if modelName == 'EfficientNet_M':
                    # print(response_time)
                    if Scheduler.success + Scheduler.fail < 2000:
                        M.write(str(response_time * 1000) + '\n')
                    else:
                        M.close()
                elif modelName == 'EfficientNet_S':
                    if Scheduler.success + Scheduler.fail < 2000:
                        S.write(str(response_time * 1000) + '\n')
                    else:
                        S.close()

                # print('model = {} rp_time = {}'.format(modelName,response_time * 1000))

                if time.perf_counter() - request_time < deadline:
                    # print((time.perf_counter() - request_time) * 1000)
                    # print('loop time = {:.3f}ms'.format(
                    # (time.perf_counter() - loopStartTime)*1000))
                    Scheduler.success = Scheduler.success+1
                else:
                    Scheduler.fail = Scheduler.fail + 1

                # print(tpuInvokeTime * 1000)
             # if preprocessingFlag:
                #     print('requires preprocessing')
                # else:
                #     print('non preprocessing')

                # print('setInputTime = {:.3f}ms'.format((setInputTime)*1000))
                # print('getClassesTime = {:.3f}ms'.format((getClassesTime)*1000))

                # print('loop time = {:.3f}ms'.format(
                #     (time.perf_counter() - loopStartTime)*1000))
                # # print('schedulingOverheadTime = {:.3f}ms'.format(
                #     (schedulingOverheadTime)*1000))
                # print('imageResizeTime = {:.3f}ms'.format((imageResizeTime)*1000))
                # print('tpuInvokeTime = {:.3f}ms'.format((tpuInvokeTime)*1000))
                # print('overHeadTime = {:.3f}ms'.format(
                #     (setInputTime + getClassesTime)*1000))
                # print('remain time = {:.3f}ms'.format((time.perf_counter() - loopStartTime - schedulingOverheadTime - imageResizeTime
                #                                     - tpuInvokeTime - setInputTime)*1000))

    def run(self):
        self.schedule_thread = threading.Thread(target=self.schedule, args=())
        self.schedule_thread.start()
        # self.schedule_thread.join()


if __name__ == '__main__':
    os.system('rm ./Pipe/*')

    interpreter = Interpreter()
    interpreter.initialize_dict()
    interpreter.initialize_model()

    named_pipe = NamedPipe()
    named_pipe.run()

    scheduler = Scheduler(interpreter)
    scheduler.run()

    analyzer = Analyzer(scheduler)
    analyzer.run()
