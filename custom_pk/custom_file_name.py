#-v1 2024.01.15 KONG SEONUI
 # -add csv_file_name()
 
#-v2 2024.01.21 KONG SEONUI
# -devide main file_name()
# -add h5_file_name()

import time as tm
def file_name(path, filename, format):
    ltm = tm.localtime(tm.time())
    save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}" 
    return f"{path}{filename}{save_time}.{format}"

def h5_file_name(path, filename):
    return file_name(path, filename, 'h5')

def csv_file_name(path, filename):
    return file_name(path, filename, 'csv')