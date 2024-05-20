from torch import optim

class optimizer:
    """This class implements the selection of optimizer
    """
    def __new__(cls , name : str):
        """ This magic method is used for the selection of the optimizer function

        Args:
            name (str): name of the optimizer function

        Returns:
            str: returns the name of the optimizer function found in the class
        """
        return getattr(cls,name)

    def adam(self,l_r : float,w_d:float,parameter : any) -> any:
        return optim.Adam(parameter, lr = l_r,weight_decay=w_d)
    def SGD(self,l_r : float,w_d:float, parameter : any) -> any:
        return optim.SGD(parameter, lr = l_r, momentum = 0.09,weight_decay=w_d)
