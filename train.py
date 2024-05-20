from tqdm import tqdm
import torch
from utils import preprocess
from checkpoint import checkpoint
import mlflow
class Train_Val:
    def __init__(self,data ,model,train_loader,val_loader,optimzer,loss,early_stop,exp_lr_scheduler) -> None:
        """This implements the intializing of the parameters that is required for training and validating the data.

        Args:
            data (_type_): contains the whole config data in a dictionary form.
            model (_type_): contains the model
            train_loader (_type_): contains the trainloader
            val_loader (_type_): contains the trainloader
            optimzer (_type_): contains the optimzer depending upon the config.
            loss (_type_): contains the loss depending upon the config.
            early_stop (_type_): contains the early stopping parameters from the main file.
            exp_lr_scheduler (_type_): learning rate scheduler
        """
        self.data = data
        self.epochs=data['Epochs']
        self.checkpoints=data["checkpoint"]
        self.model=model
        self.train_loader=train_loader
        self.val_loader=val_loader
        self.optimizer=optimzer
        self.dice=loss
        self.early_stopping=early_stop
        self.exp_lr_scheduler=exp_lr_scheduler
        
    def train(self):
        """ Trains the dataset depending upon the config file.
        """
        self.model.train()
        for epoch in range(self.epochs):
            iter_loss=0.0
            iter=0
            for img,mask in tqdm(self.train_loader):
                self.optimizer.zero_grad()
                img=img.to(preprocess.device)
                mask=mask.to(preprocess.device)
                predict=self.model(img)
                loss=self.dice(predict,mask)
                loss.backward()
                self.optimizer.step()
                iter_loss+=loss.item()
                iter+=img.shape[0]
            self.exp_lr_scheduler.step()
            total_loss=iter_loss/iter
            print(f"epoch/epochs => {epoch+1}/{self.epochs} ,loss_item=>{loss.item():.4f} ,loss => {total_loss:.4f}, Dice_score=> {(1-total_loss):.4f}")
            value=self.val()
            if self.data['checkpoint']['is_true']:
                checkpoint(self.checkpoints,value,self.model.state_dict())
            status=self.early_stopping(value['val_loss'])
            if status:
                    print("Model is overfitting and best model is already saved")
    def val(self):
        """ validates the dataset depending upon the config..

        Returns:
            _type_: returns the loss value in a form of dictionary for early stopping and checkpoint.
        """
        self.model.eval()
        val_loss=0.0
        loop=0
        with torch.no_grad():
            for img,mask in tqdm(self.val_loader):
                img=img.float().to(preprocess.device)
                mask=mask.float().to(preprocess.device)
                predict=self.model(img)
                loss_val=self.dice(predict,mask)
                val_loss+=loss_val.item()
                loop+=img.shape[0]
            total_Val_loss=val_loss/loop
            print(f"loss item => {loss_val.item():.4f},val_loss => {total_Val_loss:.4f}, val_dice_score : {(1-total_Val_loss):.4f}")
        return {"val_loss" : total_Val_loss}