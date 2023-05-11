## Guide to file Validation and txt2csv convertion
NILM processing software and firmware tools

### Create .env
Create .env file in root directory of project and place the following 
```
DATA_FOLDER_LOCATION = ..\nilm\
PREPROCESS_FOLDER_LOCATION = ..\nilm\PREPROCESS\
VALID_FILES=..\nilm\VALID\
PROBLEM_FILES=..\nilm\FILE_ERRORS\
```
Feel free to set parameteters.
DATA_FOLDER_LOCATION - Where to place the txt.gzip
PREPROCESS_FOLDER_LOCATION - Destination of new csv files
VALID_FILES - Validated files path
PROBLEM_FILES - Files with problems directory in txt 2 csv convertion path, also csv files that did not pass validation

###  Create virtual environment and install requirements
After creating virtual environment, do a pip install using the command below:
```
pip install -r requirements.txt
```

### txt2csv
Execute the txt2csv.py file for csv convertion.
You can either run convert all zip files or choose a subset.
When the validation steps have already been done, change the output path of th csv to "VALID_FILES"

### validate
If needed, run the validate.py file to validate created csv (flagged csv already sent to you in an excel file)
This automatically creates many subfolders inder PROBLEM_FILES, where the flagged csv are saved to the relevant problem folder location
Only when the file passes all tests, it is saved in the VALID_FILES 
