import transformers
import torch

class Classification_Model(transformers.DistilBertForSequenceClassification):
  def __init__(self):
    pretrained_model = transformers.DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
    config = pretrained_model.config
    super(Classification_Model, self).__init__(config)
    state_dict = pretrained_model.state_dict()
    self.tokenizer  = transformers.DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    self.load_state_dict(state_dict)

    for p in self.distilbert.parameters():
      p.requires_grad = False
    for p in self.pre_classifier.parameters():
      p.requires_grad = False
    
    self.criterion = torch.nn.BCELoss()
    self.optimizer = torch.optim.Adam(self.parameters(), lr = 1e-3)
    self.softmax = torch.nn.Softmax(dim = -1)
  
  def forward(self,X):
    o = super(Classification_Model, self).forward(**X)
    o = self.softmax(o.logits)
    return o
  
  def trainer(self,epochs, dataloader):
    for epoch in range(epochs):
      for num, (email, label) in enumerate(dataloader):
        input = self.tokenizer(email, truncation= True, padding = True, return_tensors = 'pt')
        target = torch.cat((label[0].unsqueeze(0), label[1].unsqueeze(0)), dim = 0).permute(1,0).to(torch.float32)
        o = self.forward(input)

        loss = self.criterion(o,target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    return None
  
  def evaluate(self,val_dataset):
    pred = []
    target = []
    for email, label in val_dataset:
      inputs = self.tokenizer(email, return_tensors="pt")
      with torch.no_grad():
          logits = self.forward(inputs)

      predicted_class_id = logits.argmax().item()
      pred.append(predicted_class_id)
      target.append(label)

    return pred, target
  
  def confusion_matrix(self,val_dataset):
    pred, target = self.evaluate(val_dataset)
    confusion_matrix = [[0 for _ in range(2)] for _ in range(2)]
    for p, t in zip(pred, target):
      t = t.index(1.0)
      confusion_matrix[p][t] += 1
    return confusion_matrix
