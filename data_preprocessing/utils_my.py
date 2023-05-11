from os import environ, listdir, path, remove
from dotenv import load_dotenv
import shutil
import logging
import traceback


# def get_env_setting(setting):
#     # Get environment variable
#     try:
#         return environ[setting]
#     except KeyError:
#         raise KeyError(f"Set the {setting} env variable")

new_path = (r'C:\Users\micki\nauka Pytonga\aMasterPytong\from_sel\DATA_FOLDER_LOCATION\20211011_20h_24h')

def get_file_names(new_path=None, termination='.txt.gz'):
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
    new_path='C:\\Users\\micki\\nauka Pytonga\\aMasterPytong\\from_sel\\DATA_FOLDER_LOCATION\\20211011_20h_24h\\'
    dirList = listdir(new_path)
    
    if termination is None:
        filenames = dirList
    else:
        for filename in dirList:
            if filename.endswith(termination):
                filenames.append(filename)
    filenames.sort()
    return filenames


def copy_files(datafilespath=None, new_path=None, file2read=None):
    """copy files to new location"""
    source = new_path.join(datafilespath, file2read)
    destination = new_path.join(new_path, file2read)
    try:
        shutil.copy(source, destination)
        print("File copied successfully.")

    # If source and destination are same
    except shutil.SameFileError:
        print("Source and destination represents the same file.")

    # If there is any permission issue
    except PermissionError:
        print("Permission denied.")

    # For other errors
    except Exception as e:
        logging.error(traceback.format_exc())


def remove_file(datafilespath=None, file2read=None):
    """
    delete file from path
    """
    file_path = new_path.join(datafilespath, file2read)
    try:
        remove(file_path)
    except Exception as e:
        logging.error(traceback.format_exc())


def zip_file(datafilespath=None, file2read=None, new_path=None):
    # source = path.join(datafilespath, file2read)
    dest = new_path.join(new_path, file2read)
    try:
        shutil.make_archive(dest, "zip", datafilespath, file2read)
    except Exception as e:
        logging.error(traceback.format_exc())


load_dotenv()
