import pandas

class Email_DataFrame():
    def __init__(self,data_dir):
        self.data_dir = data_dir
        self.df = pandas.read_csv(data_dir,encoding='latin-1')
        self.df = self.df.drop(columns = ['S. No.'])

    def get_df(self):
        return self.df

    def info(self):
        print(self.df.info())