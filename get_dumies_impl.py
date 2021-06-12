import numpy as np
import pandas as pd
from pandas.core.algorithms import unique
import numbers


def list_unique(list1):
    list_set = set(list1)
    unique_list = (list(list_set))
    return unique_list


def get_dummies(data,
                prefix=None,
                prefix_sep='_',
                dummy_na=False,
                columns=None,
                sparse=False,
                drop_first=False,
                dtype=np.uint8):

    def Series_Procesing(data: pd.Series):
        if len(data) == 0:
            return 'Data not have elements'
        headers_list = unique(data)
        unique_counts = len(headers_list)
        new_table_once = np.zeros(
            shape=(data.shape[0], unique_counts), dtype=dtype)
        i = 0
        for elem_data in data:
            for index,  elem_header in enumerate(headers_list):
                if elem_data == elem_header:
                    new_table_once[i, index] = 1
                    i += 1

        nan_column = [0 for i in range(len(data))]
        for index, head in enumerate(headers_list):
            if str('nan') == str(head):
                temp1 = np.array(new_table_once[..., 0:index:])
                nan_column = np.array(new_table_once[..., index])

                if index == len(headers_list)-1:
                    new_table_once = temp1.copy()
                    break
                temp2 = np.array(new_table_once[..., index+1::])
                new_table_once = np.concatenate((temp1, temp2), axis=1)

        headers_list = list(headers_list)
        if np.nan in headers_list:
            headers_list.remove(np.nan)

        if dummy_na:
            headers_list.append(np.nan)
            nan_column = np.expand_dims(nan_column, axis=1)
            new_table_once = np.concatenate(
                (new_table_once, nan_column), axis=1)

        if drop_first:
            new_table_once = np.array(new_table_once[..., 1::])
            headers_list = list(headers_list)
            headers_list.pop(0)

        if prefix is not None:
            headers_list = [
                prefix + prefix_sep + str(float(head) if isinstance(head, numbers.Number) else head) for head in headers_list]

        result_Data_Frame = pd.DataFrame(new_table_once, columns=headers_list)
        return result_Data_Frame

    def List_Procesing(data: list):
        if len(data) == 0:
            return 'Data not have elements'

        headers_list = unique(data)

        unique_counts = len(headers_list)

        new_table_once = np.zeros(
            shape=(len(data), unique_counts), dtype=dtype)
        i = 0
        for elem_data in data:
            for index,  elem_header in enumerate(headers_list):
                if str(elem_data) == str(elem_header):
                    new_table_once[i, index] = 1
                    i += 1
                    break

        nan_column = [0 for i in range(len(data))]
        for index, head in enumerate(headers_list):
            if str('nan') == str(head):
                temp1 = np.array(new_table_once[..., 0:index:])
                nan_column = np.array(new_table_once[..., index])

                if index == len(headers_list)-1:
                    new_table_once = temp1.copy()
                    break
                temp2 = np.array(new_table_once[..., index+1::])
                new_table_once = np.concatenate((temp1, temp2), axis=1)

        headers_list = list(headers_list)
        if np.nan in headers_list:
            headers_list.remove(np.nan)

        if dummy_na:
            headers_list.append(np.nan)
            nan_column = np.expand_dims(nan_column, axis=1)
            new_table_once = np.concatenate(
                (new_table_once, nan_column), axis=1)

        if drop_first:
            new_table_once = np.array(new_table_once[..., 1::])
            headers_list = list(headers_list)
            headers_list.pop(0)

        if prefix is not None:
            headers_list = [
                prefix + prefix_sep + str(float(head) if isinstance(head, numbers.Number) else head) for head in headers_list]

        result_Data_Frame = pd.DataFrame(new_table_once, columns=headers_list)
        return result_Data_Frame

    def DataFrame_Procesing(data: pd.DataFrame):
        if len(data) == 0:
            return 'Data not have elements'
        new_table_once = pd.DataFrame(data)
        headers_list = list(unique(data.columns))
        unique_counts = len(headers_list)
        data_from_DataFrame = data.values
        decomposition_column = None
        data_from_DataFrame = data_from_DataFrame.T
        prefix_decomposition_coloumn = None

        if columns is not None:
            for index, head in enumerate(headers_list):
                for column in columns:
                    if column == head:
                        prefix_decomposition_coloumn = headers_list[index]

                        if prefix is not None:
                            prefix_decomposition_coloumn = prefix

                        decomposition_column = get_dummies(new_table_once[column],
                                                           prefix=prefix_decomposition_coloumn,
                                                           prefix_sep=prefix_sep,
                                                           dummy_na=dummy_na)
                        new_table_once = new_table_once.drop(
                            str(headers_list[index]), axis=1)
                        new_table_once = pd.concat(
                            [new_table_once, decomposition_column], axis=1)
        else:
            for index, column in enumerate(data_from_DataFrame):
                for elem in column:
                    if isinstance(elem, str):
                        decomposition_column = list(column)
                        prefix_decomposition_coloumn = headers_list[index]

                        if prefix is not None:
                            prefix_decomposition_coloumn = prefix

                        decomposition_column = get_dummies(decomposition_column,
                                                           prefix=prefix_decomposition_coloumn,
                                                           prefix_sep=prefix_sep,
                                                           dummy_na=dummy_na)

                        new_table_once = new_table_once.drop(
                            str(headers_list[index]), axis=1)
                        new_table_once = pd.concat(
                            [new_table_once, decomposition_column], axis=1)
                        break

        return new_table_once

    if isinstance(data, pd.Series):
        return Series_Procesing(data)
    if isinstance(data, list):
        return List_Procesing(data)
    if isinstance(data, pd.DataFrame):
        return DataFrame_Procesing(data)
