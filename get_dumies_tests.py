import pandas as pd
import unittest
import get_dumies_impl  

class TestGetDumies:
    def __init__(self):
        pass

    def should_data_none(self):
        s = pd.Series(list(''))
        get_dumies_impl.get_dummies(s)

    def simple_test(self):
        s = pd.Series(list(''))
        s.count
        pd.get_dummies(s)
