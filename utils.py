import torch 
import numpy as np
from monai.transforms import(Compose,RandGaussianNoised,RandGaussianSmoothd,MapTransform,AddChanneld,EnsureTyped,
                            NormalizeIntensityd,RandAdjustContrastd,RandFlipd,RandShiftIntensityd,RandScaleIntensityd,Orientationd,
                            RandRotated,ToTensord,EnsureChannelFirstd,ConcatItemsd,DeleteItemsd)

class Normalization:#scales b/w [0,1] or [-1,1]
    def __call__(self,image ):
        """This class implements the normalization of the dataset ranging between 0 and 1.

        Args:
            img (_type_): normalized value of the numpy array of the image.
        """
        max=np.max(image)
        min=np.min(image)
        image=(image-min)/(max-min)

class standardize(MapTransform):#scales b/w depending on data
    def __call__(self, data):
        """This class implements the standardization of the dataset depending the dataset. This inherits the class from Maptransform from Monai.

        Args:
            data (_type_): image in the numpy array.

        Returns:
            _type_: standardized the numerical values of the image numpy array.
        """
        d=dict(data)
        for key in self.key_iterator(d):
            mask=d[key]>0
            low=np.percentile(d[key][mask], 0.2)
            up=np.percentile(d[key][mask], 99.9)
            d[key][mask & (d[key]<low)]=low
            d[key][mask & (d[key]>up)]=up 
            y=d[key][mask]
            d[key]-=y.mean()
            d[key]/=y.std()
        return d
    
class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    def __call__(self, data):
        """ converts the mask or label into categorical depending upon the classes .

        Args:
            data (_type_): image in the numpy array.

        Returns:
            _type_: converted mask or label into classes.
        """
        d = dict(data)
        for key in self.keys:
            result = []
            result.append(np.logical_or(d[key] == 2, d[key] == 3))
            result.append(np.logical_or(np.logical_or(d[key] == 2, d[key] == 3), d[key] == 1))
            result.append(d[key] == 2)
            d[key] = np.stack(result, axis=0).astype(np.uint8)
        return d
    
class _preprocess:
    def __init__(self):
        """ This class conatins the preprocessing parameters for the model.
            It conatins the Device and Transform.
        """
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.train_transform=Compose([
                        AddChanneld(keys=['flair','t1','t1ce','t2']),
                        standardize(keys=['flair','t1','t1ce','t2']),
                        ConcatItemsd (keys=['flair','t1','t1ce','t2'],name='image',dim=0),
                        DeleteItemsd(keys=['flair','t1','t1ce','t2']),
                        ConvertToMultiChannelBasedOnBratsClassesd(keys='mask'),
                        RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0),
                        RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1),
                        RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=2),
                        RandGaussianNoised(keys='image',prob=0.25,mean=0,std=0.33),
                        RandGaussianSmoothd(keys='image',prob=0.25,sigma_x=(0.5,1.5),sigma_y=(0.5,1.5),sigma_z=(0.5,1.5)),
                        RandScaleIntensityd(keys=['image'], factors=0.1, prob=1.0),
                        RandShiftIntensityd(keys=['image'], offsets=0.1, prob=1.0),
                        RandAdjustContrastd(keys='image', prob=0.25,gamma=(0.7,0.13)),
                        EnsureTyped(keys=['image', 'mask']),])
        self.val_trans=Compose([
                        AddChanneld(keys=['flair','t1','t1ce','t2']),
                        standardize(keys=['flair','t1','t1ce','t2']),
                        ConcatItemsd (keys=['flair','t1','t1ce','t2'],name='image',dim=0),
                        DeleteItemsd(keys=['flair','t1','t1ce','t2']),
                        ConvertToMultiChannelBasedOnBratsClassesd(keys='mask'),
                        RandAdjustContrastd(keys='image', prob=0.25,gamma=(0.7,0.13)),
                        EnsureTyped(keys=['image', 'mask']),])
        

preprocess=_preprocess()