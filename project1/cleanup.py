import xlrd
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

"""
def get_exel_sheet_names(input_file_name):
    sheet_objects = xlrd.open_workbook(input_file_name, on_demand=True)
    sheet_list = sheet_objects.sheet_names()
    return sheet_list


def get_data_frame_from_excel(input_file_name, sheet_name):
    df = pd.read_excel(input_file_name, sheet_name=sheet_name, index_col=0)
    return df
"""

