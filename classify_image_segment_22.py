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
from statistics import mode
import time
from urllib import request
from urllib.request import Request
import numpy as np
import os
from PIL import Image
from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
import threading
from datetime import datetime

lock = threading.Lock()
preempt_lock = threading.Lock()


class NamedPipe:
    request_list = []
    process_dict = {}
    current_task = {"model_name": 'Default',
                    "priority": 0}
    preempt_flag = False

    def __init__(self):
        self.path = './Pipe/' + 'listenPipe'

    def make_listen_pipe(self):
        if os.path.exists(self.path):
            os.remove(self.path)
        os.mkfifo(self.path)
        self.listenPipe = os.open(self.path, os.O_RDONLY)

    def make_client_pipe(self):
        while True:
            readmsg = (os.read(self.listenPipe, 100)).decode()
            if readmsg:
                pid, modelName = readmsg.split(' ')
                ServerWritePath = './Pipe/' + 'ServerTo' + str(pid) + '_Pp'
                if os.path.exists(ServerWritePath):
                    os.remove(ServerWritePath)
                os.mkfifo(ServerWritePath)
                ClientWritePath = './Pipe/' + str(pid) + 'ToServer_Pp'
                if os.path.exists(ClientWritePath):
                    os.remove(ClientWritePath)
                os.mkfifo(ClientWritePath)
                ReadWritePipe = threading.Thread(target=self.read_and_write_to_queue, args=(
                    pid, modelName, ServerWritePath, ClientWritePath))
                ReadWritePipe.start()
                # break

    def read_and_write_to_queue(self, pid, model_name, server_write_path, client_write_path):
        to_client = os.open(server_write_path, os.O_WRONLY)
        from_client = os.open(client_write_path, os.O_RDONLY)
        pid = int(pid)
        NamedPipe.process_dict[pid] = (model_name, to_client)
        prev_read = 0
        while True:
            before_read = time.perf_counter()
            msg = os.read(from_client, 100).decode()
            # print('read = {}'.format( (time.perf_counter() - before_read) * 1000 ))
            
            # queue_start = time.perf_counter()
            try:
                pid, image_path, request_time, period, segment_num = msg.split(
                    ' ')
            except(ValueError):
                print(msg)
            pid = int(pid)
            period = float(period)
            
            request_time = float(request_time)
            until_server_read_message_end = time.perf_counter()
            until_server_read_message_time = (time.perf_counter() - request_time)
            print('------------------------------------------------------')
            print('until server read mesaage = {}'.format(until_server_read_message_time * 1000))
            
            # 지금 들어온 request가 현재 실행중인 task의 segment보다 priority가 높으면
            # preempt_flag를 True로
            # preempt_flag를 False로는 언제 바꾸지?
            
            ####################################################3
            preempt_lock_start = time.perf_counter()
            preempt_lock.acquire()
            priority = 1/period

            if priority > NamedPipe.current_task["priority"] and NamedPipe.current_task["priority"] != 0:
                NamedPipe.preempt_flag = True
            else:
                NamedPipe.preempt_flag = False

            preempt_lock.release()

            preempt_lock_release = time.perf_counter()
            print('acquring preempt lock = {} '.format((preempt_lock_release - preempt_lock_start) * 1000))
            ########################################################
            
            #####################################################
            request_lock = time.perf_counter()
            segment_num = int(segment_num)
            if msg:
                request_list_lock_acquire_start = time.perf_counter()
                lock.acquire()
                NamedPipe.request_list.append(
                    (pid, image_path, request_time, period, segment_num))
                
                lock.release()
                request_list_lock_acquire_time = time.perf_counter() - request_list_lock_acquire_start
            request_lock_release = time.perf_counter()
            print('acquirng request lock = {}'.format((request_lock_release - request_lock) * 1000))
            #####################################################
            
            
            print('until server_read_message to put request = {}'.format( (time.perf_counter() -preempt_lock_start) * 1000 ))
                # print('request_list_lock_acquire_time = {}'.format(request_list_lock_acquire_time* 1000))
                
            # queue_overhead_time = time.perf_counter() - queue_start
            # print('queue overhead_time = {}'.format(queue_overhead_time * 1000))

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
        self.model_path_dict = {}
        # todo
        # 요청을 받을때 model path를 list에 추가함으로써
        model_1 = '/home/hun/WorkSpace/coral/pycoral/model/segment/Efficient_L/efficientnet-edgetpu-L_quant_segment_0_of_3_edgetpu.tflite'
        model_2 = '/home/hun/WorkSpace/coral/pycoral/model/segment/Efficient_L/efficientnet-edgetpu-L_quant_segment_1_of_3_edgetpu.tflite'
        model_3 = '/home/hun/WorkSpace/coral/pycoral/model/segment/Efficient_L/efficientnet-edgetpu-L_quant_segment_2_of_3_edgetpu.tflite'
        modelName = 'EfficientNet_L'
        self.model_path_dict[modelName] = [model_1, model_2, model_3]

        model_1 = '/home/hun/WorkSpace/coral/pycoral/model/segment/Efficient_M/efficientnet-edgetpu-M_quant_segment_0_of_3_edgetpu.tflite'
        model_2 = '/home/hun/WorkSpace/coral/pycoral/model/segment/Efficient_M/efficientnet-edgetpu-M_quant_segment_1_of_3_edgetpu.tflite'
        model_3 = '/home/hun/WorkSpace/coral/pycoral/model/segment/Efficient_M/efficientnet-edgetpu-M_quant_segment_2_of_3_edgetpu.tflite'
        modelName = 'EfficientNet_M'
        self.model_path_dict[modelName] = [model_1, model_2, model_3]

        model_1 = '/home/hun/WorkSpace/coral/pycoral/model/segment/Efficient_S/efficientnet-edgetpu-S_quant_segment_0_of_3_edgetpu.tflite'
        model_2 = '/home/hun/WorkSpace/coral/pycoral/model/segment/Efficient_S/efficientnet-edgetpu-S_quant_segment_1_of_3_edgetpu.tflite'
        model_3 = '/home/hun/WorkSpace/coral/pycoral/model/segment/Efficient_S/efficientnet-edgetpu-S_quant_segment_2_of_3_edgetpu.tflite'
        modelName = 'EfficientNet_S'
        self.model_path_dict[modelName] = [model_1, model_2, model_3]

        model_1 = '/home/hun/WorkSpace/coral/pycoral/model/segment//Mobilenet_V1/mobilenet_v1_1.0_224_quant_segment_0_of_3_edgetpu.tflite'
        model_2 = '/home/hun/WorkSpace/coral/pycoral/model/segment//Mobilenet_V1/mobilenet_v1_1.0_224_quant_segment_1_of_3_edgetpu.tflite'
        model_3 = '/home/hun/WorkSpace/coral/pycoral/model/segment//Mobilenet_V1/mobilenet_v1_1.0_224_quant_segment_2_of_3_edgetpu.tflite'
        modelName = 'MobileNet_V1'
        self.model_path_dict[modelName] = [model_1, model_2, model_3]

    def initialize_dict(self):
        self.make_model_path_list()

        for model_name in self.model_path_dict:
            interpreter_segment_list = []
            for model_path in self.model_path_dict[model_name]:
                interpreter_segment = make_interpreter(model_path)
                interpreter_segment.allocate_tensors()
                interpreter_segment_list.append(interpreter_segment)
            self.model_interpreter_dict[model_name] = interpreter_segment_list


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

            print('O : {} X : {} in Queue: {} ReqPerSec : {}'.format(Scheduler.success,
            Scheduler.fail,len(NamedPipe.request_list),self.request_per_sec))

            time.sleep(1)

    def run(self):
        analyzer_thread = threading.Thread(target=self.CountRequest, args=())
        analyzer_thread.start()


# 1. 유동적인 Segment개수에 따른 스케줄링
# -- 어떻게 하드코딩된 세그먼트를 처리할 것인가 ( clear )
# 2. Preempt가능한 스케줄링
# -- 어떻게 intermediate output을 저장할 것인가
# -- 어떻게 preempt를 지원할 것인가  (Preempt flag?)
# ---- Namedpipe에서 request list에 다음 task 넣을 때 현재 스케줄링 중인 task보다 priority 높으면 Preempt flag를 setting 한다
# ---- segment를 invoke 할때 Preempt flag를 보고
# ------ Preempt Flag FALSE --> 다음 segment 이어서 실행
# ------ Preempt Flag TRUE  --> intermediate output과 다음 실행할 segment index를 저장
# ---- 그럼 또 preempt flag도 lock을 잡고 read write을 해야함
# -- 유동적인 Segment개수에 따른 다음 Segment invoke를 어떻게 실행할 것인가 (intermediate output에 다음 실행 index와 같이 저장)
# 3. Analyzer 구현
# -- execution time : segement time 모두 이어서 실행했을 때 걸리는 시간
# -- blocking time : 가장 실행시간 긴 한개 segment의 exection time
# -- interference time : priority 높은 애들이 자신의 period 동안 얼마나 들어와서 얼마나 실행이 되나

 # 4. Segmented 되어있는

#  {"model_name" : 'Default',
#                                      "pid" : 0,
#                                      "priority" : 0,
#                                      "intermediate_output" : 0,
#                                      "next_index" : 0,
#                                      "segment_num" : 0}

# 5. segment 되어 있는 task 들의 rm scheduling
# 1. waiting queue에서 priority 제일 높은 task를 가져온다
# -- 첫번째 segment를 실행시킨다
# -- preempt flag를 확인한 후
# ---- preempt flag 가 true면 다음 output을 저장
# ------ 마지막 segment이면 ? preempt flag 상관 없이
# ---- preempt flag 가 false면 이어서 실행
# segment를 실행하고 preemp0.035008388105779886
# 6. preempt된 task부터 있는지 확인?
# 현재 코드 상으로는 preempt된 task가 있는지 먼저 확인을 하는데
# 이러면 현재 request queue에 priority 더 높은 task 보다 단지 preempt 되었다는 이유로 preempt된 task를 먼저 실행한다
# 어떻게 수정?
# 현재 request queue와  , intermediate_output을 동시에 확인해야해야하는데


class Scheduler:
    success = 0
    fail = 0

    def __init__(self, Interpreter):
        self.interpreter = Interpreter
        self.labels = read_label_file(
            '/home/hun/WorkSpace/coral/pycoral/test_data/labels.txt')

        self.intermediate_output = []

    def schedule(self):
        while True:
            # preempt된 task부터 있는지 확인
            # intermediate_output을 저장해둔것이 있으면 preempt된 task가 있는 것
            execute_segment_flag = False
            execute_request_flag = False
            
            lock_start = time.perf_counter()
            lock.acquire()
            
            if len(self.intermediate_output) != 0 and len(NamedPipe.request_list) != 0:
                execute_segment_flag = True
                segment_max_priority = -1
                for i in range(len(self.intermediate_output)):
                    if self.intermediate_output[i]["priority"] > segment_max_priority:
                        segment_max_priority = self.intermediate_output[i]["priority"]
                        # print(self.intermediate_output[i]["priority"])
                for i in range(len(NamedPipe.request_list)):
                    task_priority = 1 / NamedPipe.request_list[i][3]
                    # print('task_priority = {}, segment_max_priority = {}'.format(task_priority, segment_max_priority))
                    if task_priority > segment_max_priority:
                        execute_segment_flag = False
                        execute_request_flag = True
                        break
                    
            elif len(self.intermediate_output) != 0  and len(NamedPipe.request_list) == 0:
                execute_segment_flag = True
                execute_request_flag = False
                
            elif len(self.intermediate_output) == 0 and len(NamedPipe.request_list) != 0:
              
                while_start = time.perf_counter()
                execute_segment_flag = False
                execute_request_flag = True
                        
            lock.release()    
            lock_time = (time.perf_counter() - lock_start) * 1000
            
            

            # intermediate_output and request_list are not empty

            if execute_segment_flag:
                # 있으면 priority 제일 높은 segment부터 실행
                self.intermediate_output.sort(key=lambda x: x["priority"], reverse = True)
                next_task = self.intermediate_output.pop(0)
                pid = next_task["pid"]
                segment_intermediate_output = next_task["intermediate_output"]
                next_index = next_task["next_index"]
                segment_num = next_task["segment_num"]
                modelName = next_task["model_name"]
                priority = next_task["priority"]
                
                ###################################
                # print('executing preempted segment model = {}  ,index = {}'.format(modelName, next_index))

                lock.acquire()
                NamedPipe.current_task["model_name"] = modelName
                NamedPipe.current_task["priority"] = priority
                lock.release()
                preempt_lock.acquire()
                NamedPipe.preempt_flag = False
                preempt_lock.release()

                for i in range(next_index, segment_num):
                    # executing last segment
                    if i == segment_num - 1:

                        assigned_interpreter = interpreter.model_interpreter_dict[modelName][i]
                        common.set_input(assigned_interpreter,
                                         segment_intermediate_output)
                        assigned_interpreter.invoke()
                        tensor_index = assigned_interpreter.get_output_details()[
                            0]['index']
                        output = assigned_interpreter.get_tensor(tensor_index)
                        classes = classify.get_classes(
                            assigned_interpreter, top_k, threshold)

                        msg = ''
                        for c in classes:
                            msg = msg + ('%s: %.5f\n' %
                                         (self.labels.get(c.id, c.id), c.score))
                        os.write(ToClient, msg.encode())
                        # print(msg)

                        if time.perf_counter() - request_time < deadline:
                            Scheduler.success = Scheduler.success+1
                        else:
                            Scheduler.fail = Scheduler.fail + 1

                    # executing middle segment
                    else:
                        # print('i is not segment_num - 1   , i == {}'.format(i))
                        assigned_interpreter = interpreter.model_interpreter_dict[modelName][i]
                        common.set_input(assigned_interpreter,
                                         segment_intermediate_output)
                        assigned_interpreter.invoke()
                        tensor_index = assigned_interpreter.get_output_details()[
                            0]['index']
                        segment_intermediate_output = assigned_interpreter.get_tensor(
                            tensor_index)

                        preempt_lock.acquire()
                        if NamedPipe.preempt_flag == True:
                            # print('****************************preempt in preempt')
                            self.intermediate_output.append({"model_name": modelName,
                                                             "pid": pid,
                                                             "priority": priority,
                                                             "intermediate_output": segment_intermediate_output,
                                                             "next_index": i+1,
                                                             "segment_num": segment_num})
                            preempt_lock.release()
                            break
                        else:
                            preempt_lock.release()

            elif execute_request_flag:
                loopStartTime = time.perf_counter()
                print('request')
                # print(len(request_list))
                ##FIFO##
                lock.acquire()

                NamedPipe.request_list.sort(key=lambda request: request[3])
                who = NamedPipe.request_list.pop(0)
                pid = who[0]
                period = who[3]
                modelName = NamedPipe.process_dict[pid][0]
                ToClient = NamedPipe.process_dict[pid][1]
                NamedPipe.current_task["model_name"] = modelName
                NamedPipe.current_task["priority"] = 1/period

                #################33
                # print('exectue request model Name = {}'.format(modelName))
                lock.release()
                ########
                preempt_lock.acquire()
                NamedPipe.preempt_flag = False
                preempt_lock.release()

                imagePath = './imageDir/' + who[1]
                request_time = who[2]
                
                print('1 {} '.format((time.perf_counter() - lock_start) * 1000))
                until_while_start = (time.perf_counter() - request_time) * 1000
                print('until_while_start = {}'.format(until_while_start))

                segment_num = who[4]
                # print(period)
                deadline = period

                # 요청을 받으면 모델과 interpreter를 만들어서

                count = 5
                threshold = 0.0
                top_k = 3

                schedulingOverheadTime = time.perf_counter() - loopStartTime
                imageResizeStart = time.perf_counter()

                for i in range(segment_num):
                    # executing first segment
                    if i == 0:
                        assigned_interpreter = interpreter.model_interpreter_dict[modelName][0]
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
                            np.clip(normalized_input, 0, 255,
                                    out=normalized_input)
                            common.set_input(
                                assigned_interpreter, normalized_input.astype(np.uint8))
                        assigned_interpreter.invoke()
                        tensor_index = assigned_interpreter.get_output_details()[
                            0]['index']
                        output = assigned_interpreter.get_tensor(tensor_index)

                        preempt_lock.acquire()
                        if NamedPipe.preempt_flag == True:
                            # print('{} is preempted! next index = {}'.format(modelName,i+1))
                            self.intermediate_output.append({"model_name": modelName,
                                                             "pid": pid,
                                                             "priority": 1 / period,
                                                             "intermediate_output": output,
                                                             "next_index": i+1,
                                                             "segment_num": segment_num})
                            preempt_lock.release()
                            break
                        else:
                            preempt_lock.release()

                    # executing last segment
                    elif i == segment_num - 1:
                        assigned_interpreter = interpreter.model_interpreter_dict[modelName][i]
                        common.set_input(assigned_interpreter, output)
                        assigned_interpreter.invoke()
                        tensor_index = assigned_interpreter.get_output_details()[
                            0]['index']
                        output = assigned_interpreter.get_tensor(tensor_index)

                        classes = classify.get_classes(
                            assigned_interpreter, top_k, threshold)
                        msg = ''
                        for c in classes:
                            msg = msg + ('%s: %.5f\n' %
                                         (self.labels.get(c.id, c.id), c.score))
                        os.write(ToClient, msg.encode())
                        while_end = (time.perf_counter() - while_start)
                        print('while end  = {}'.format(while_end * 1000))
                        print(' plus two = {}'.format(until_while_start + while_end))
                        # print(msg)

                        # if (time.perf_counter() - request_time) < deadline:
                        if while_end < deadline: 
                            print('full executioned time = {}'.format((time.perf_counter() - request_time) *1000))
                            print('------------------------------------------')
                            Scheduler.success = Scheduler.success+1
                        else:
                            print('full executioned time = {}'.format((time.perf_counter() - request_time)* 1000))
                            print('------------------------------------------')
                            Scheduler.fail = Scheduler.fail + 1

                    # executing middle segment
                    else:
                        assigned_interpreter = interpreter.model_interpreter_dict[modelName][i]
                        common.set_input(assigned_interpreter, output)
                        assigned_interpreter.invoke()
                        tensor_index = assigned_interpreter.get_output_details()[
                            0]['index']
                        output = assigned_interpreter.get_tensor(tensor_index)

                        preempt_lock.acquire()
                        if NamedPipe.preempt_flag == True:
                            # print('{} is preempted! next index = {}'.format(modelName,i+1))
                            self.intermediate_output.append({"model_name": modelName,
                                                             "pid": pid,
                                                             "priority": 1 / period,
                                                             "intermediate_output": output,
                                                             "next_index": i+1,
                                                             "segment_num": segment_num})
                            preempt_lock.release()
                            break
                        else:
                            preempt_lock.release()

            # if Scheduler.preempt_flag == False:
            #     msg = ''
            #     for c in classes:
            #         msg = msg + ('%s: %.5f\n' % (self.labels.get(c.id, c.id), c.score))

            #     # print(msg)

            #     os.write(ToClient, msg.encode())

            #     if time.perf_counter() - request_time < deadline:
            #         Scheduler.success = Scheduler.success+1
            #     else:
            #         Scheduler.fail = Scheduler.fail + 1

    def run(self):
        self.schedule_thread = threading.Thread(target=self.schedule, args=())
        self.schedule_thread.start()
        # self.schedule_thread.join()


if __name__ == '__main__':

    os.system('rm ./Pipe/*')

    named_pipe = NamedPipe()
    named_pipe.run()

    interpreter = Interpreter()
    interpreter.initialize_dict()

    scheduler = Scheduler(interpreter)
    scheduler.run()

    analyzer = Analyzer(scheduler)
    analyzer.run()
