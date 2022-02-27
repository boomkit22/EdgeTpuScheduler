
import time
from datetime import datetime

print(time.time())
print(type(time.time()))
# print(time.perf_counter())
# intermediate_output = [{"model_name" : 'Default',
#                                           "priority" : 1,
#                                           "intermediate_output" : 0,
#                                           "next_index" : 0}]



# intermediate_output.append({"model_name" : 'Efficient', "priority" : 2, "intermediate_output":1,"next_index" : 1})


# print(intermediate_output)


# intermediate_output.sort(key = lambda x : x["priority"], reverse=True)

# print(intermediate_output)
# current_task = {"model_name" :'Default'
#         , "priority" : 3.0}

# print(current_task["model_name"])
# print(current_task["priority"])
# model_path_dict = {}
# model_1 = '/home/hun/WorkSpace/coral/pycoral/model/segment/Efficient_L/efficientnet-edgetpu-L_quant_segment_0_of_3_edgetpu.tflite'
# model_2 = '/home/hun/WorkSpace/coral/pycoral/model/segment/Efficient_L/efficientnet-edgetpu-L_quant_segment_1_of_3_edgetpu.tflite'
# model_3 = '/home/hun/WorkSpace/coral/pycoral/model/segment/Efficient_L/efficientnet-edgetpu-L_quant_segment_2_of_3_edgetpu.tflite'
# modelName = 'EfficientNet_L'
# model_path_dict[modelName] = [model_1, model_2, model_3]

# model_1 = '/home/hun/WorkSpace/coral/pycoral/model/segment/Efficient_M/efficientnet-edgetpu-M_quant_segment_0_of_3_edgetpu.tflite'
# model_2 = '/home/hun/WorkSpace/coral/pycoral/model/segment/Efficient_M/efficientnet-edgetpu-M_quant_segment_1_of_3_edgetpu.tflite'
# model_3 = '/home/hun/WorkSpace/coral/pycoral/model/segment/Efficient_M/efficientnet-edgetpu-M_quant_segment_2_of_3_edgetpu.tflite'
# modelName = 'EfficientNet_M'
# model_path_dict[modelName] = [model_1, model_2, model_3]

# model_1 = '/home/hun/WorkSpace/coral/pycoral/model/segment/Efficient_S/efficientnet-edgetpu-S_quant_segment_0_of_3_edgetpu.tflite'
# model_2 = '/home/hun/WorkSpace/coral/pycoral/model/segment/Efficient_S/efficientnet-edgetpu-S_quant_segment_1_of_3_edgetpu.tflite'
# model_3 = '/home/hun/WorkSpace/coral/pycoral/model/segment/Efficient_S/efficientnet-edgetpu-S_quant_segment_2_of_3_edgetpu.tflite'
# modelName = 'EfficientNet_S'
# model_path_dict[modelName] = [model_1, model_2, model_3]

# model_1 = '/home/hun/WorkSpace/coral/pycoral/model/segment//Mobilenet_V1/mobilenet_v1_1.0_224_quant_segment_0_of_3_edgetpu.tflite'
# model_2 = '/home/hun/WorkSpace/coral/pycoral/model/segment//Mobilenet_V1/mobilenet_v1_1.0_224_quant_segment_1_of_3_edgetpu.tflite'
# model_3 = '/home/hun/WorkSpace/coral/pycoral/model/segment//Mobilenet_V1/mobilenet_v1_1.0_224_quant_segment_2_of_3_edgetpu.tflite'
# modelName = 'MobileNet_v1'
# model_path_dict[modelName] = [model_1, model_2, model_3]

# model_interpreter_dict = {}

# for model_name in model_path_dict:
#     interpreter_segment = []
#     for model_path in model_path_dict[model_name]:
#         interpreter_segment.append(model_path)
    
#     model_interpreter_dict[model_name] = interpreter_segment
        
# print(model_interpreter_dict[model_name][0])
# print(model_interpreter_dict[model_name][1])
# print(model_interpreter_dict[model_name][2])