from general.data_validation import (
    check_outliers,
    check_rms_voltage,
    check_rms_current,
    check_dimension,
    check_timestamps,
    check_data_continuity,
    check_zero_cross_freq,
    check_zero_cross_total,
)
from general.parsers import readCSV
from general.utils import get_env_setting, get_file_names, copy_files, remove_file
from os import path, makedirs

path_to_file = get_env_setting("PREPROCESS_FOLDER_LOCATION")
filenames = get_file_names(path_to_file, termination=".csv")
path_problems = get_env_setting("PROBLEM_FILES")
valid_file_path = get_env_setting("VALID_FILES")

validations = ["outliers", "rms_voltage", "rms_current", "dimensions", "zero_cross_freq", "zero_cross_total", "timestamps", "data_continuity"]
validation_path = [path.join(path_problems, validation) for validation in validations]

# Create folders for errors
for val_path in validation_path:
    makedirs(val_path) if not path.exists(val_path) else None

# timestamps
faulty_minutes, flag = check_timestamps(filenames)

if flag:
    for faulty_minute in faulty_minutes:
        copy_files(datafilespath=path_to_file, new_path=validation_path[6], file2read=faulty_minute)
        valid = False

voltage1 = []
errors = []
for file in filenames:
    timestamp, sampleN, voltage, current = readCSV(path=path_to_file, filename=file)

    valid = True
    # outliers
    value, index, flag, value_diff = check_outliers(L=voltage, threshold=50)
    if flag:
        copy_files(datafilespath=path_to_file, new_path=validation_path[0], file2read=file)
        valid = False
        errors.append([file, validation_path[0]])

    # rms_voltage
    outlier, index, flag = check_rms_voltage(L=voltage)
    if flag:
        copy_files(datafilespath=path_to_file, new_path=validation_path[1], file2read=file)
        valid = False
        errors.append([file, validation_path[1]])

    # rms_current
    outlier, index, flag = check_rms_current(L=current)
    if flag:
        copy_files(datafilespath=path_to_file, new_path=validation_path[2], file2read=file)
        valid = False
        errors.append([file, validation_path[2]])

    # dimension
    l1_size, l2_size, flag = check_dimension(L1=current, L2=voltage)
    if flag:
        copy_files(datafilespath=path_to_file, new_path=validation_path[3], file2read=file)
        valid = False
        errors.append([file, validation_path[3]])

    # # zero_cross_freq
    # index_max, index_min, max_sample, min_sample, flag = check_zero_cross_freq(L=voltage)
    # if flag:
    #     copy_files(datafilespath=path_to_file, new_path=validation_path[4], file2read=file)
    #     valid = False
    #     errors.append([file, validation_path[4]])

    # zero_cross_total
    number, flag = check_zero_cross_total(L=voltage)
    if flag:
        copy_files(datafilespath=path_to_file, new_path=validation_path[5], file2read=file)
        valid = False
        errors.append([file, validation_path[5]])

    # data_continuity
    diff, flag = check_data_continuity(L1=voltage1, L2=voltage)
    if flag:
        copy_files(datafilespath=path_to_file, new_path=validation_path[7], file2read=file)
        valid = False
        errors.append([file, validation_path[7]])

    voltage1 = voltage

    if valid:
        copy_files(datafilespath=path_to_file, new_path=valid_file_path, file2read=file)

    # remove_file(path_to_file, file)

print(len(errors))
print(errors)
