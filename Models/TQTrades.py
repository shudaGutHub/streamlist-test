from dataclasses import dataclass
import json
import pandas as pd
import pprint as pp
import csv

TQ_TRADE_TYPES = {}

from dx import *

def print_nested_dicts(nested_dict, indent_level=0):
    """This function prints a nested dict object
    Args:
        nested_dict (dict): the dictionary to be printed
        indent_level (int): the indentation level for nesting
    Returns:
        None

    """

    for key, val in nested_dict.items():
        if isinstance(val, dict):
            print("{0} : ".format(key))
            print_nested_dicts(val, indent_level=indent_level + 1)
        elif isinstance(val, list):
            print("{0} : ".format(key))
            for rec in val:
                print_nested_dicts(rec, indent_level=indent_level + 1)
        else:
            print("{0}{1} : {2}".format("\t" * indent_level, key, val))


def extract_json(file_name, do_print=True):
    """This function extracts and prints json content from a given file
    Args:
        file_name (str): file path to be read
        do_print (bool): boolean flag to print file contents or not

    Returns:
        None

    """
    try:
        json_filedata = open(file_name).read()
        json_data = json.loads(json_filedata)

        if do_print:
            print_nested_dicts(json_data)
    except IOError:
        raise IOError("File path incorrect/ File not found")
    except ValueError:
        ValueError("JSON file has errors")
    except Exception:
        raise


def extract_pandas_json(file_name, orientation="records", do_print=True):
    """This function extracts and prints json content from a file using pandas
       This is useful when json data represents tabular, series information
    Args:
        file_name (str): file path to be read
        orientation (str): orientation of json file. Defaults to records
        do_print (bool): boolean flag to print file contents or not

    Returns:
        None

    """
    try:
        df = pd.read_json(file_name, orient=orientation)

        if do_print:
            print(df)
    except IOError:
        raise IOError("File path incorrect/ File not found")
    except ValueError:
        ValueError("JSON file has errors")
    except Exception:
        raise


def print_basic_csv(file_name, delimiter=','):
    """This function extracts and prints csv content from given filename
       Details: https://docs.python.org/2/library/csv.html
    Args:
        file_name (str): file path to be read
        delimiter (str): delimiter used in csv. Default is comma (',')

    Returns:
        None

    """
    csv_rows = list()
    csv_attr_dict = dict()
    csv_reader = None

    # read csv
    csv_reader = csv.reader(open(file_name, 'r'), delimiter=delimiter)

    # iterate and extract data
    for row in csv_reader:
        print(row)
        csv_rows.append(row)

    # prepare attribute lists
    for col in csv_rows[0]:
        csv_attr_dict[col] = list()

    # iterate and add data to attribute lists
    for row in csv_rows[1:]:
        csv_attr_dict['sno'].append(row[0])
        csv_attr_dict['fruit'].append(row[1])
        csv_attr_dict['color'].append(row[2])
        csv_attr_dict['price'].append(row[3])

    # print the result
    print("\n\n")
    print("CSV Attributes::")
    pprint(csv_attr_dict)


def print_tabular_data(file_name, delimiter=","):
    """This function extracts and prints tabular csv content from given filename
       Details: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
    Args:
        file_name (str): file path to be read
        delimiter (str): delimiter used in csv. Default is comma ('\t')

    Returns:
        None

    """
    df = pd.read_csv(file_name, sep=delimiter)
    print(df)


# --- GET VALUES FROM EXCEL
# xw.Book.caller() References the calling book
# when the Python function is called from Excel via RunPython.
wb = xw.Book.caller()
sht = wb.sheets("Portfolio")
show_msgbox = wb.macro("modMsgBox.ShowMsgBox")
TARGET_CURRENCY = sht.range("TARGET_CURRENCY").value
START_ROW = sht.range("TICKER").row + 1  # Plus one row after the heading
LAST_ROW = sht.range(sht.cells.last_cell.row, Column.ticker.value).end("up").row
sht.range("TIMESTAMP").value = timestamp()
tickers = (
    sht.range(START_ROW, Column.ticker.value).options(expand="down", numbers=str).value
)
