from torch.optim import lr_scheduler
import torch
from optimizer import optimizer
import json
from monai.data import DataLoader
from dataset import brats
from model import unet
from loss import loss
from utils import preprocess
from train import Train_Val
from earlystopping import earlystopping

with open('config.json') as user_file:
    """ 
        loading the json file and writing into variable.
    """
    data = json.load(user_file)
    
class main:
    def __init__(self):
        """
            functions the all the function from the dataset preparation to model training,checkpoints and early stopping.
        """
        criteria = loss(data['loss'],)(loss)
        model=unet(data['out_channels']).to(preprocess.device)
        train_data=brats(path=data['train_path'],transform=preprocess.train_transform)
        train_loader=DataLoader(train_data,batch_size=data['batch_size'],shuffle=True)
        val_data=brats(path=data['val_path'],transform=preprocess.val_trans)
        val_loader=DataLoader(val_data,batch_size=data['batch_size'],shuffle=False)
        optimi = optimizer(data['optimizer'])(optimizer,data['l_r'],data['weight_decay'],model.parameters())
        early_stop=earlystopping(data['earlystopping']['is_true'],data['earlystopping']['patience'],data['earlystopping']['min_del'])
        exp_lr_scheduler=lr_scheduler.StepLR(optimi,step_size=10,gamma=0.1,verbose=True)
        Train_Val(data,model,train_loader,val_loader,optimi,criteria,early_stop,exp_lr_scheduler).train()

            

    
if __name__ == '__main__':
    main()   
    