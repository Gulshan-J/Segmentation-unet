import torch
import numpy as np

class  checkpoint:
    max = 0.0
    min=np.Inf
    def __init__(self, checkpoint_dict : dict, value : dict, modelstate : any) -> None:
        """intializes the parameters used for checkpoint.

        Args:
            checkpoint_dict (dict): contains the dictionary of the checkpoints parameters.
            value (dict): Dict contains the loss or accuracy or both.
            modelstate (any): model weights to be saved
        """
        self.value = value
        self.modelstate = modelstate
        self.checkpoint_dict = checkpoint_dict
        self.check()

    def check(self) -> None:
        """checks validation accuracy or loss depending upon the config.

        Raises:
            NameError: if not found rasises the name error
        """

        if self.checkpoint_dict['monitor'] == 'val_accuracy':
            self.val_accuracy()
        elif self.checkpoint_dict['monitor'] == 'val_loss':
            self.val_loss()
        else:
            raise NameError
 
    def save_model(self) -> None:
        """
            saves the model in the path(save_path) given by the config
        """
        save_path = self.checkpoint_dict['save_path']+'/'+f"_{self.checkpoint_dict['monitor']}_{self.checkpoint_dict['mode']}" +'.pth.par'
        print('saving the model.....')
        torch.save(self.modelstate, save_path)

    def val_accuracy(self) -> None:
        """ implemets the validation accuracy

        Raises:
            NameError: if not found raises the name error
        """
        print(checkpoint.max)
        if self.checkpoint_dict['mode'] == 'max':
            if checkpoint.max < self.value['val_accuracy']:
                checkpoint.max = self.value['val_accuracy']
                self.save_model()
        elif self.checkpoint_dict['mode'] == 'min':
            if self.value['val_accuracy'] < checkpoint.min:
                self.save_model()
                checkpoint.min = self.value['val_accuracy']
        else:
            raise NameError 
    
    def val_loss(self) -> None:
        """ implements the validation loss

        Raises:
            NameError: if not found raises the name error
        """
        if self.checkpoint_dict['mode'] == 'min':
            if  self.value['val_loss'] < checkpoint.min:
                checkpoint.min = self.value['val_loss']
                self.save_model()
                print(f" minimum loss value {checkpoint.min}")

        elif self.checkpoint_dict['mode'] == 'max':
            if checkpoint.max < self.value['val_loss']:
                checkpoint.max=self.value['val_loss']
                self.save_model()
        else:
            raise NameError
            

        
               
        
