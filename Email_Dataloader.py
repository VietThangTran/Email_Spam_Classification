import torch

class Email_Dataloader():
    def __init__(self,dataset,batch_size = 16, shuffle = True):
        self.dataset = dataset
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size, shuffle) 

    def get_dataloader(self):
        return self.dataloader
