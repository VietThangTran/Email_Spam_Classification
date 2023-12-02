import torch

class Email_Dataset(torch.utils.data.Dataset):
  def __init__(self,df):
    self.df = df
    self.label2id = {label : id for id, label in enumerate(df['Label'].unique())}
    self.data = self._get_data()

  def _get_data(self):
    data = []
    for row in self.df.iloc:
      email = row['Message_body']
      label = row['Label']
      label = self.label2id[label]
      label = [1. if x == label else 0. for x in range(2)]
      data.append((email,label))
    return data

  def __len__(self):
    return len(self.data)
  
  def __getitem__(self,idx):
    email, label = self.data[idx]
    return email, label