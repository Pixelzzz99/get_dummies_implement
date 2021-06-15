import numpy as np
import pandas as pd
from pandas.core.algorithms import unique


class Series_List_Utils_Processing:
    def get_unique_headers_list(self, data):
        return list(unique(data))

    '''*********Data convert Dummy style Table*********'''
    def create_get_dummies_style_table(self, data, dtype):
        headers_list = unique(data)

        column_counts = self.__get_column_count(data)
        row_counts = self.__get_row_count(data)

        new_table_once = self.__get_zeros_table(
            row_counts, column_counts, dtype)
        return self.__dummy_processing(new_table_once, data, headers_list)

    def __get_column_count(self, data):
        return len(unique(data))

    def __get_row_count(self, data):
        return data.shape[0] if isinstance(data, pd.Series) else len(data)

    def __get_zeros_table(self, row_count, columns_count, dtype):
        return np.zeros(shape=(row_count, columns_count), dtype=dtype)

    def __dummy_processing(self, new_table_once, data, headers_list):
        row = 0
        for elem_data in data:
            for column,  elem_header in enumerate(headers_list):
                if self.__compare_elements(elem_data, elem_header):
                    new_table_once = self.__put_the_unit_in_table(
                        new_table_once, row, column)
                    row = self.__next_row(row)
                    break
        return new_table_once

    def __compare_elements(self, element1, element2):
        return str(element1) == str(element2)

    def __put_the_unit_in_table(self, new_table_once, row, column):
        new_table_once[row, column] = 1
        return new_table_once

    def __next_row(self, row):
        row += 1
        return row
    '''*************************************************'''


    '''******* Drop Nan Column From Table (is default) ****************'''
    def drop_nan_column_from_dummy_table(self, data, new_table_once, headers_list):
        nan_column = self.__get_nan_column_of_data_legth(data)

        for index, head in enumerate(headers_list):
            if self.__is_nan_column(head):
                new_table_once, nan_column = self.__drop_nan_column_from_table(
                    new_table_once, index, headers_list)
        headers_list = self.__remove_nan_header_from_header_list(headers_list)
        return nan_column, new_table_once

    def __get_nan_column_of_data_legth(self, data):
        return [0 for i in range(len(data))]

    def __is_nan_column(self, head):
        return str('nan') == str(head)

    def __drop_nan_column_from_table(self, new_table_once, index, headers_list):
        left_table = self.__get_left_table_from_nan_column(
            new_table_once, index)
        nan_column = self.__get_nan_column_from_table(new_table_once, index)

        if self.__is_nan_column_last_header(index, headers_list):
            return left_table.copy(), nan_column

        right_table = self.__get_right_table_from_nan_column(
            new_table_once, index)

        return self.__concat_two_tables(left_table, right_table), nan_column

    def __get_left_table_from_nan_column(self, new_table_once, index):
        return np.array(new_table_once[..., 0:index:])

    def __get_nan_column_from_table(self, new_table_once, index):
        return np.array(new_table_once[..., index])

    def __is_nan_column_last_header(self, index, headers_list):
        return index == len(headers_list)-1

    def __get_right_table_from_nan_column(self, new_table_once, index):
        return np.array(new_table_once[..., index+1::])

    def __remove_nan_header_from_header_list(self, headers_list):
        return headers_list.remove(np.nan) if np.nan in headers_list else headers_list

    def __concat_two_tables(self, left_table, right_table):
        return np.concatenate((left_table, right_table), axis=1)
    '''*************************************************'''


    '''****** Add Nan Column ********************************'''
    def add_nan_column(self, new_table_once, headers_list, nan_column):
        self.__add_nan_head_in_header_list(headers_list)
        nan_column = self.__add_dims_to_correct_concat(nan_column)
        new_table_once = self.__concat_two_tables(new_table_once, nan_column)
        return new_table_once, headers_list

    def __add_nan_head_in_header_list(self, headers_list):
        headers_list.append(np.nan)

    def __add_dims_to_correct_concat(self, nan_column):
        return np.expand_dims(nan_column, axis=1)

    def drop_first_column(self, new_table_once, headers_list):
        new_table_once = self.__overwrite_table_without_the_first_column(
            new_table_once)
        self.__remove_first_head_from_header_list(headers_list)
        return new_table_once, headers_list

    def __overwrite_table_without_the_first_column(self, new_table_once):
        return np.array(new_table_once[..., 1::])

    def __remove_first_head_from_header_list(self, headers_list):
        headers_list.pop(0)
    '''*************************************************'''


    '''***** Add Prefix and PrefixSep Methods ******'''
    def add_prefix(self, headers_list, prefix, prefix_sep):
        consist_char = self.__is_have_char_element_in_header(headers_list)
        if consist_char:
            return self.__add_default_prefix(headers_list, prefix, prefix_sep)
        return self.__add_prefix_only_numbers_headers(headers_list, prefix, prefix_sep)        

    def __is_have_char_element_in_header(self, headers_list):
        for head in headers_list:
            if isinstance(head, str):
                return True
        return False

    def __add_default_prefix(self, headers_list, prefix, prefix_sep):
        return [prefix + prefix_sep + str(head) for head in headers_list]

    def __add_prefix_only_numbers_headers(self, headers_list, prefix, prefix_sep):
        return [prefix + prefix_sep + str(float(head)) for head in headers_list]
    '''*************************************************'''

    
    '''****** Sorted Data (headers and table by headers) *****'''
    def sorted_data(self, headers_list, new_table_once):        
        headers_list, index_headers_list_sort = self.__get_sorted_headers_list_and_index(headers_list)
        new_table_once = self.__sort_table_by_headers_indexes(new_table_once, index_headers_list_sort)
        self.__nan_column_corrector(headers_list)
        return headers_list, new_table_once
        
    def __get_sorted_headers_list_and_index(self, headers_list):
        temp_headers_list = [str(head) for head in headers_list]
        index_headers_list_sort = [i[0] for i in sorted(enumerate(temp_headers_list), key=lambda x:x[1])]
        headers_list = sorted(temp_headers_list)
        return headers_list, index_headers_list_sort
    
    def __sort_table_by_headers_indexes(self, new_table_once, index_headers_list_sort):
        transpose_table = new_table_once.T
        sorted_new_table = self.__sort_table_by_index(index_headers_list_sort, transpose_table)
        return np.array(sorted_new_table).T

    def __sort_table_by_index(self, index_headers_list_sort, table):
        sorted_table = list()
        for index_sorted in index_headers_list_sort:
            for index, column in enumerate(table):
                if index_sorted == index:
                    sorted_table.append(column)
        return sorted_table

    def __nan_column_corrector(self, headers_list):
        is_nan = self.__last_column_is_nan(headers_list)
        return self.__change_str_nan_to_float_nan(headers_list) if is_nan == 'nan' else headers_list

    def __last_column_is_nan(self, headers_list):
        return headers_list[len(headers_list)-1]
        
    def __change_str_nan_to_float_nan(self, headers_list):
        headers_list.pop()
        headers_list.append(np.nan)
        return headers_list
    '''*************************************************'''


class DataFrame_Utils_Processing:

    '''****************Method if have column param*******************'''
    def columns_procesing(self, new_table_once, columns_list, headers_list, prefix,
                          prefix_sep, dummy_na_status,drop_first_status):

        for column in columns_list:
            for index, head in enumerate(headers_list):
                if self.__compare_column_and_header(column, head):
                    prefix_decomposition_coloumn = self.__prefix_determinant(headers_list, index, prefix)

                    decomposition_column = get_dummies(new_table_once[column],
                                                       prefix=prefix_decomposition_coloumn,
                                                       prefix_sep=prefix_sep,
                                                       dummy_na=dummy_na_status,
                                                       drop_first=drop_first_status)

                    new_table_once = self.__remove_found_column_from_the_DataFrame(new_table_once, headers_list, index)
                    new_table_once = self.__concat_two_tables(new_table_once, decomposition_column)

        return new_table_once

    def __compare_column_and_header(self, column, head):
        return column == head

    def __prefix_determinant(self, headers_list, index, prefix):
        return prefix if prefix is not None else headers_list[index]

    def __remove_found_column_from_the_DataFrame(self, new_table_once, headers_list, index):
        return new_table_once.drop(str(headers_list[index]), axis=1)

    def __concat_two_tables(self, left_table, second_table):
        list_tables = [left_table, second_table]
        return pd.concat(list_tables, axis=1)
    '''************************************************'''

    '''*************Method if there is no Columns parameter************'''
    def default_DataFrame_Procesing(self, new_table_once, data_values, headers_list, prefix,
                                    prefix_sep, dummy_na_status, drop_first_status):

        for index, column in enumerate(data_values):
            for elem in column:
                if isinstance(elem, str):
                    decomposition_column = list(column)
                    prefix_decomposition_coloumn = self.__prefix_determinant(headers_list, index, prefix)

                    decomposition_column = get_dummies(decomposition_column,
                                                       prefix=prefix_decomposition_coloumn,
                                                       prefix_sep=prefix_sep,
                                                       dummy_na=dummy_na_status,
                                                       drop_first=drop_first_status)

                    new_table_once = self.__remove_found_column_from_the_DataFrame(new_table_once, headers_list, index)
                    new_table_once = self.__concat_two_tables(new_table_once, decomposition_column)
                    break
        return new_table_once
    '''************************************************'''

def get_dummies(data,
                prefix=None,
                prefix_sep='_',
                dummy_na=False,
                columns=None,
                drop_first=False,
                dtype=np.uint8):

    def empty(data):
        return len(data) == 0

    def data_processing(data):
        utils = Series_List_Utils_Processing()

        if empty(data):
            return 'Data not have elements'

        headers_list = utils.get_unique_headers_list(data)

        new_table_once = utils.create_get_dummies_style_table(data, dtype)

        nan_column, new_table_once = utils.drop_nan_column_from_dummy_table(
            data, new_table_once, headers_list)

        if dummy_na:
            new_table_once, headers_list = utils.add_nan_column(
                new_table_once, headers_list, nan_column)

        if drop_first:
            new_table_once, headers_list = utils.drop_first_column(
                new_table_once, headers_list)

        if prefix is not None:
            headers_list = utils.add_prefix(headers_list, prefix, prefix_sep)

        headers_list, new_table_once = utils.sorted_data(headers_list, new_table_once)

        return pd.DataFrame(new_table_once, columns=headers_list)

    def Series_Processing(data: pd.Series):
        return data_processing(data)

    def List_Processing(data: list):
        return data_processing(data)

    def Numpy_Array_Processing(data: np.ndarray):
        if data.ndim > 1:
            return 'Data must be 1-dimensional'
        return data_processing(data)

    def DataFrame_Processing(data: pd.DataFrame):
        utils = DataFrame_Utils_Processing()

        if empty(data):
            return 'Data not have elements'

        new_table_once = pd.DataFrame(data)
        headers_list = list(unique(data.columns))
        data_from_DataFrame = data.values
        data_from_DataFrame = data_from_DataFrame.T

        if columns is not None:
            return utils.columns_procesing(new_table_once,
                                           columns,
                                           headers_list,
                                           prefix,
                                           prefix_sep,
                                           dummy_na,
                                           drop_first_status=drop_first)

        return utils.default_DataFrame_Procesing(new_table_once,
                                                 data_from_DataFrame,
                                                 headers_list,
                                                 prefix,
                                                 prefix_sep,
                                                 dummy_na,
                                                 drop_first_status=drop_first)


    if isinstance(data, pd.Series):
        return Series_Processing(data)
    if isinstance(data, list):
        return List_Processing(data)
    if isinstance(data, pd.DataFrame):
        return DataFrame_Processing(data)
    if isinstance(data, np.ndarray):
        return Numpy_Array_Processing(data)
