import Email_Dataset, Email_DataFrame, Email_Dataloader, Email_Spam_Model
import torch

if __name__ == '__main__':
    data_dir = 'data/SMS_test.csv'

    df = Email_DataFrame.Email_DataFrame(data_dir)
    dataset = Email_Dataset.Email_Dataset(df.get_df())
    dataloader = Email_Dataloader.Email_Dataloader(dataset,batch_size=16, shuffle= True).get_dataloader()
    pretrained_model = torch.load('spam_classification.pth')
    model = Email_Spam_Model.Classification_Model()
    model.load_state_dict(pretrained_model['model'])

    confusion_matrix = model.confusion_matrix(dataset)
    print(confusion_matrix)