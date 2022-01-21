import pandas as pd


class Data:

    @staticmethod
    def get_data():

        return pd.read_csv("res/Position_Salaries.csv")
