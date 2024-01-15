#-v1 2024.01.15 KONG SEONUI
import time as tm
def csv_file_name(path, filename = None):
    ltm = tm.localtime(tm.time())
    save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}" 
    return path + f"{filename}{save_time}.csv"