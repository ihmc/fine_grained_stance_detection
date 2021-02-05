# Installation
Clone the repository to your system. This event detection software uses python 3.7 or newer.
Inside the directory that is cloned you will need to run the following to install the necessary dependencies:
```bash
pip install -r requirements.txt
```

# Usage
Once the install has finished you will need to start up a python interactive environment:
```bash
python
```

Once the interactive environment is running you will need to:
```python
import detection_app
```

This will take a little while to load but once it does you can proceed to run stance detection on data in a number of ways (Please note for the stance processing from file functions, any large input files will take quite a while to process):

1. The first way is the simplest but will only run on one line of text. It is mean just to test specific sentences or small text to see what stance they will produce it can be run by the following:
```python
detection_app.test("This is a sentence to run")
```

The next 3 ways are a bit more involved and make use of files of data to do stance detection. Each of the functions described below will output a file to a directory named "user_provided_stance_output" which the software will create if it does not already exist. The output files will have a .jsonl extension because each line in the output file will be the stances for each line in the input file.

2. **Detecting stances from a text file**  
The .txt file provided must have a single text per line. For example one line could read "I wear masks to protect me." (Note the quotations do not have to be present in the .txt file).
    ```python
    detection_app.text_to_stances("path/to/input/file", "data_description", 0)
    ```
    * The second parameter will be used to build the output files name so the user can provided a description that helps keep track of the output files and the data they are created from. If the data description is not specified the output file will be called user_provided_text_stances.jsonl. NOTE: That if your run this function more than once without specifying the description it will always overwrite the previous file named user_provided_text_stances.jsonl and fill it with the most recent output.
    * The last parameter is the amount of lines in the input file that the user wants to process. If the user specifies 0 or does not provide the last parameter the function will default to processing the entirety of the data in the input file.
</br>

3. **Detect stances from a json file**  
User provides a path to a json file for stance processing. The extension does not have to be json, but each line must be a single json structure that has an attribute for the text to process, some kind of author identifier, a timestamp, and some kind of document identifier. For example if the json represented a tweet there should be the author id, timestamp of the tweet, and the id of the tweet. User must provide the name of each of these attributes that is found in each json structure.
	```python
	detection_app.json_to_stances("path/to/input/file", "text_attribute", "author_attribute", "timestamp_attribute", "doc_id_attribute", "data_description", 0)
	```
    * NOTE: If one of the attributes is nested (i.e. author attribute is {"user": {"id": 12345678}} then supply the attibute, in appropriate order, comma separated, "user,id".
    * Data description appends the provided text to the output file name as described above. NOTE: the same warning as above applies here.
    * Again, the last parameter is the number of lines to process, providing 0 or not having that parameter at all will process the whole input data file.
</br>

4. **Detect stances from a json file**. 
User provides a path to a csv file for stance processing. The file must have a header row with a label for the text to process, some kind of author identifier, a timestamp, and some kind of document identifier. For example if each csv line represented a tweet there would be headers for text of the tweet, the author id, timestamp of the tweet, and the id of the tweet. User must provide the name of each of these labels that is found in the csv file.
	```python
	detection_app.csv_to_stances("path/to/input/file", "text_label", "author_label", "timestamp_label", "doc_id_label", "data_description", 0)
	```
    * NOTE: If one of the items requires the combinations of two csv columns (i.e. the timestamp is made up from a date column and a time column) then supply both labels, in appropriate order pipe (|) separated, "date|time".
    * Data description appends the provided text to the output file name as described above. NOTE: the same warning as above applies here.
    * Again, the last parameter is the number of lines to process, providing 0 or not having that parameter at all will process the whole input data file.

