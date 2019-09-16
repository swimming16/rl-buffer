import os
import json
import  matplotlib.pyplot as plt

# COOKED_TRACE_FOLDER = './cooked_traces/'
# COOKED_TRACE_FILE = '/network.json'
COOKED_TRACE_FOLDER = './network/'

#load a file
# def load_trace(cooked_trace_file=COOKED_TRACE_FILE):
#     cooked_dir = os.getcwd()
#     cooked_trace_file=cooked_dir+cooked_trace_file
#     cooked_time = []
#     cooked_bw = []
#     print(cooked_trace_file)
#     with open(cooked_trace_file, 'r') as f:
#         manifset=json.load(f)
#     for i in manifset:
#         cooked_time.append(i[])

#load a dir
def load_trace(cooked_trace_folder=COOKED_TRACE_FOLDER):
    cooked_files = os.listdir(cooked_trace_folder)
    all_cooked_time = []
    all_cooked_bw = []
    all_file_names = []
    for cooked_file in cooked_files:
        file_path = cooked_trace_folder + cooked_file
        cooked_time = []
        cooked_bw = []
        # print file_path
        with open(file_path, 'rb') as f:
            for line in f:
                parse = line.split()
                cooked_time.append(float(parse[0]))
                cooked_bw.append(float(parse[1]))
        all_cooked_time.append(cooked_time)
        all_cooked_bw.append(cooked_bw)
        all_file_names.append(cooked_file)
        print(all_cooked_bw)
        return all_cooked_time, all_cooked_bw, all_file_names

def plot_bandwidth(all_time,all_bandwidth,all_filename):
    for i,j,k in zip(all_time,all_bandwidth,all_filename):
        plt.plot(i,j,label=k)
        plt.show()


