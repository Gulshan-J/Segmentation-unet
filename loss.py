from monai.losses import dice

class loss:
    """
        This class implements the selection of loss function 
    """
    def __new__(cls , name : str):
        """ This magic method is used for the selection of the loss function

        Args:
            name (str): name of the loss function

        Returns:
            str: returns the name of the loss function found in the class
        """
        return getattr(cls,name)
   
    def Diceloss(self):
        return dice.DiceLoss(softmax=True)
    def Dice_focal(self):
        return dice.DiceFocalLoss(softmax=True)
    def Dice_CE(self):
        return dice.DiceCELoss(softmax=True)
    
