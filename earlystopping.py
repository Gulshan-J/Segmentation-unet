import numpy as np

class earlystopping:
    def __init__(self,is_true=False,tolerance=None, min_delta=None) -> None :
        """ implements the early stopping function.
            init : intializes the parameters

        Args:
            is_true (bool, optional): either True or False. Defaults to False.
            tolerance (_type_, optional): check number of time with no improvement after which training will be stopped. Defaults to None.
            min_delta (_type_, optional): small change in the monitored quantity to qualify as an improvement. Defaults to None.
        """
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.min_validation_loss = np.inf
        self.counter = 0
        self.is_true=is_true
        self.setvalue()
    
    def setvalue(self) -> None:
        """
            sets the value if no tolerance or min_delta is provided.
        """
        if self.tolerance is None:
            self.tolerance=5
        if self.min_delta is None:
            self.min_delta=0
    
    def __call__(self,validation_loss : float) -> bool:
        """ implements the early stopping
        Args:
            validation_loss (float): validation loss
        Returns:
            bool: returns the message whether it is true or false.
        """
        if self.is_true:
            if validation_loss < self.min_validation_loss:
                self.min_validation_loss = validation_loss
                self.counter = 0
            elif validation_loss > (self.min_validation_loss + self.min_delta):
                self.counter += 1
                if self.counter >= self.tolerance:
                    return True
            return False
        
