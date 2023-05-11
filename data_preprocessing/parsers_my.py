import gzip
import shutil
import numpy as np
import os
import csv
from datetime import datetime
from general.signal_processing import instant_power

import time

#new_path = (r'C:\Users\micki\nauka Pytonga\aMasterPytong\from_sel\DATA_FOLDER_LOCATION\20210930_00h_04h')

def get_file_names(new_path, termination=None):
    """
    Get the names of all files existent in the .env DATA_XXX_LOCATION variable.
    If termination has a value, get only files with a specific termination

    Parameters
    ----------
    path : string
        string with folder location of which we want to get the files names
    termination : string
        string with specific termination type (e.g., .csv, ".txt.gz")

    Returns
    ----------
    filenames : list
        list with relevant files
    """
    filenames = []
    dirList = os.listdir(new_path)
    if termination is None:
        filenames = dirList
    else:
        for filename in dirList:
            if filename.endswith(termination):
                filenames.append(filename)
    filenames.sort()
    return filenames


# load CSV File data (timestamp, sampleN, voltage, current, instantPower)
def readCSV(new_path, filename):
    timestamp = []
    sampleN = []
    voltage = []
    current = []
    power = []
    if filename == "NULL":
        filename = get_file_names(new_path, ".csv")[0]
    with open(new_path + filename, newline="\n") as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=",")
        line_count = 0
        for row in csv_reader:
            if line_count != 0:
                timestamp.append(row[0])
                sampleN.append(int(line_count))
                voltage.append(float(row[1]))
                current.append(float(row[2]))
            line_count += 1
    return timestamp, sampleN, voltage, current


# writes the read raw data in a .csv format
def writeCSV(new_path, t, voltage, current, power, header):
    csvFileName, date = create_csv_file_name(t)
    with open(new_path + '\\'+csvFileName, "w", encoding="UTF8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        utctime = datetime.strptime(date[:23], "%Y/%m/%d %H:%M:%S")
        date = utctime.timestamp()
        for i in range(len(voltage)):
            data = [date, round(voltage[i], 3), round(current[i], 3), round(power[i], 3)]
            date = ""
            writer.writerow(data)


def create_csv_file_name(t):
    """ """
    timestamp = t[0]
    year = timestamp[0:4]
    month = timestamp[5:7]
    day = timestamp[8:10]
    hour = timestamp[11:13]
    minute = timestamp[14:16]
    second = timestamp[17:19]
    date = (str)(year + month + day + "_" + hour + minute + second)
    csvFileName = date + ".csv"
    date = (str)(year + "/" + month + "/" + day + " " + hour + ":" + minute + ":" + second)
    return csvFileName, date


# writes obtained metrics to csv format file
def writeMetrics(new_path, t, sampleN, meteringData, header):
    timestamp = (float)(t[0])
    date = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S")
    csvFileName = new_path + date + ".csv"
    with open(csvFileName, "w", encoding="UTF8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        print(len(meteringData))
        for k in range(len(meteringData)):
            for i in range(7500):
                timestamp = t[i]
                data = [timestamp, i, meteringData[i]]
                writer.writerow(data)


def read_txt(new_path, file2read):
    """
    Uncompresses .gz files and read the txt
    Computes Real voltage and current from raw V and I
    Returns list arrays with values for date, V, I and a boolean to check if file is valid

    Parameters
    ----------
    path : string
        string with folder location of which we want to get the files names
    file2read : string
        string with file name (including extension .gz)

    Returns
    ----------
    time :
        ...
    voltage :
        ...
    current :
        ...
    power : :
        ...
    """
    each_line = []
    time_t = []
    voltage = []
    current = []
    with gzip.open(new_path + '\\' + file2read, "rb") as f_in:
        print(file2read.replace(".gz", ""))
        with open(new_path + file2read.replace(".gz", ""), "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    with open(new_path + file2read.replace(".gz", ""), "r", encoding="utf-8") as f:
        for line in f:
            each_line.append(line.split("/"))
    flat_list = flatten(each_line)
    os.remove(new_path + file2read.replace(".gz", ""))
    for i in range(0, len(flat_list) - 1, 3):
        time_t.append(flat_list[i])
        voltage.append(flat_list[i + 1].split(","))
        current.append(flat_list[i + 2].split(","))
    voltage = flatten(voltage)
    current = flatten(current)
    voltage = voltage[:-1]  # to remove last ","
    current = current[:-1]  # to remove last ","
    voltage = np.array(voltage, dtype=np.int32)
    current = np.array(current, dtype=np.int32)
    voltage = np.divide(voltage, 2 ** 23)  # normalize to 24 bits
    current = np.divide(current, 2 ** 23)
    offsetVoltage = 0.469  # np.mean(voltage)  # for DB it was used 0.473 for JW it was used 0.469
    # print(offsetVoltage)
    offsetCurrent = 0.496  # np.mean(current)  # for DB it was used 0.494 for JW it was used 0.496
    # print(offsetCurrent)
    voltage = np.subtract(voltage, offsetVoltage)  # offset correction voltage channel
    current = np.subtract(current, offsetCurrent)  # offset correction current channel
    # in case of JW box a R of 2.2kOm was used the correction factor should be increased to 1.354
    # for DB voltage correction R of 4.7kOhm cte 1.167
    voltage = np.multiply(voltage, 2 * 2 ** 0.5 * 230 * 1.354)  # headroom was left to not saturate ADC input in case of peak voltage
    current = np.multiply(current, 2 * 2 ** 0.5 * 30 / 1.391586)  # amplitude correction factor due to CT burden resistor
    power = instant_power(voltage=voltage, current=current)
    return time_t, voltage, current, power


def flatten(t):
    return [item for sublist in t for item in sublist]


# def get_file_names(path, filenames):
# 	with open(path+filenamesFile) as f:
# 		filenames = f.read().splitlines()
# 	return filenames

# unkown chars on JWA filenames (had to rename them all)
def renameFiles(new_path):
    filenames = os.listdir(new_path)
    for filename in filenames:
        # print(filename)
        date = filename[0:8]
        hour = filename[9:11]
        # print(hour)
        minute = filename[12:14]
        # print(minute)
        second = filename[15:17]
        # print(second)
        newfilename = (str)(date + "_" + hour + ":" + minute + ":" + second + ".txt.gz")
        # print(newfilename)
        os.rename(new_path + filename, new_path + newfilename)
