from glob import glob
from monai.data import Dataset
import nibabel as nb
import numpy as np

class brats(Dataset):
    """Implements the dataset loading and returns the image and label.

    Args:
        Dataset (_type_): inheriting the dataset class from Monai Dataset.
        path (_type_): The path of the image and label data
        transform (_type_, optional): to apply transform for the dataset. Defaults to None.
    """
    def __init__(self,path : str,transform=None) -> None:

        self.flair_list=sorted(glob(f'{path}/*/*flair.nii.gz'))
        self.t1ce_list=sorted(glob(f'{path}/*/*t1ce.nii.gz'))
        self.t1_list=sorted(glob(f'{path}/*/*t1.nii.gz'))
        self.t2_list=sorted(glob(f'{path}/*/*t2.nii.gz'))
        self.mask_list=sorted(glob(f'{path}/*/*seg.nii.gz'))
        self.transform=transform
    
    def __len__(self) -> int:
        """ Implements the length of the dataset

        Returns:
            int: returns the length of the dataset.
        """
        return (len(self.flair_list))

    def __getitem__(self,idx):
        """ To get the image and label.

        Args:
            idx (_type_): index 

        Returns:
            _type_: image and label.
        """
        flair=nb.load(self.flair_list[idx]).get_fdata().astype(np.float32)
        flair=flair[56:184,56:184,13:141]
        t1ce=nb.load(self.t1ce_list[idx]).get_fdata().astype(np.float32)
        t1ce=t1ce[56:184,56:184,13:141]
        t1=nb.load(self.t1_list[idx]).get_fdata().astype(np.float32)
        t1=t1[56:184,56:184,13:141]
        t2=nb.load(self.t2_list[idx]).get_fdata().astype(np.float32)
        t2=t2[56:184,56:184,13:141]
        mask=nb.load(self.mask_list[idx]).get_fdata().astype(np.float32)
        mask=mask[56:184,56:184,13:141]
        if self.transform:
            train_files=self.transform({'flair': flair,'t1':t1,'t1ce':t1ce,'t2':t2,'mask': mask})
        image,label=train_files['image'],train_files['mask']
        return image,label
