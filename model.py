import torch
import torch.nn as nn

class IdiographicClassifier(nn.Module):
    def __init__(self, params_loaded):
        super().__init__()
        
        # self.pretrained_model = torch.hub.load('pytorch/vision:v0.10.0', params_loaded['model']['pretrained'], pretrained=True)
        # self.feature_extractor = torch.nn.Sequential(*list(self.pretrained_model.children())[:-1*params_loaded['model']['layer']])
        self.loss = nn.MSELoss()
        self.classifier = nn.Sequential(nn.Linear(1000,50),nn.LayerNorm(50),nn.LeakyReLU(inplace=True),nn.Linear(50,1))

        self._init_weights()
    
    def init_weights(self, l):
        if type(l) == nn.Linear:
          nn.init.normal_(l.weight, mean=0.0, std=0.01)
          if l.bias != None:
            l.bias.data.fill_(0)

    def _init_weights(self):
        self.classifier.apply(self.init_weights)
    
    def compute_loss(self,pred, target):
       return self.loss(pred, target)
    
    def forward(self,img):
    #    out = self.feature_extractor(img)
    #    feats = img
    #    img = torch.squeeze(img, (2,3))
       out = self.classifier(img)
       return out
