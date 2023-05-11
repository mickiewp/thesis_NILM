from general.parsers import read_txt, writeCSV, create_csv_file_name
from general.utils import get_env_setting, get_file_names, copy_files, remove_file



datafilespath = get_env_setting("DATA_FOLDER_LOCATION")
processedfilespath = get_env_setting("PREPROCESS_FOLDER_LOCATION")
filewerrors = get_env_setting("PROBLEM_FILES")
valid_file_path = get_env_setting("VALID_FILES")

input_path = datafilespath
output_path = processedfilespath
check_valid = valid_file_path


def txt2csv():
    """
    Convert zipped files from the datalogger (txt) to csv
    After converting the file, the new one (csv) is saved to the specified location above
    When the number of values for voltage and current does not match the original txt.gz file is saved to a different folder

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    filenames = get_file_names(path=input_path, termination=".txt.gz")
    all_files = input("Do your want to read all files [Y/N]: ")
    if all_files == "y" or all_files == "Y":
        initial_file_n = 1
        end_file_n = len(filenames)
    else:
        multiple_files = input("Do your want to read multiple files [Y/N]: ")
        if multiple_files == "y" or multiple_files == "Y":
            initial_file_n = int(input("Enter the initial file number to read: "))
            end_file_n = int(input("Enter the final file number to read: "))
        else:
            initial_file_n = int(input("Enter the file number you want to read: "))
            end_file_n = initial_file_n
    for i in range(initial_file_n - 1, end_file_n):
        file2read = filenames[i]
        t, voltage, current, power = read_txt(input_path, file2read)
        csvFileName, _ = create_csv_file_name(t)
        # get already processed file names
        valid_filenames = get_file_names(check_valid)
        if csvFileName not in valid_filenames:
            fieldnames = ["timestamp", "voltage", "current", "power"]
            try:
                writeCSV(output_path, t, voltage, current, power, fieldnames)
            except:
                copy_files(input_path, filewerrors, file2read)
        else:
            print("Already validated")
        # remove_file(input_path, file2read)


if __name__ == "__main__":
    txt2csv()
