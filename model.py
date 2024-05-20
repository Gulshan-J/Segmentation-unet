import torch
import torch.nn as nn

def double_conv(in_c,out_c):
    """ implements the double convolution on the image

    Args:
        in_c (_type_): in_channels
        out_c (_type_): out_channls

    Returns:
        _type_: the whole convoled image.
    """
    double_con=nn.Sequential(nn.Conv3d(in_c,out_c,kernel_size=(3,3,3),padding=1,bias=False), # changed the padding
                    nn.BatchNorm3d(out_c),
                    nn.Dropout3d(0.25),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(out_c,out_c,kernel_size=(3,3,3),padding=1,bias=False),
                    nn.BatchNorm3d(out_c),
                    nn.ReLU(inplace=True),
                    nn.Dropout3d(0.25),
                    )
    return double_con
def transpose(in_c,out_c):
    """ implements the tranpose (a decoder) of the image

    Args:
        in_c (_type_): in_channels
        out_c (_type_): out_channels

    Returns:
        _type_: the whole transposed image.
    """
    return nn.ConvTranspose3d(in_c,out_c,kernel_size=(2,2,2),stride=2)

def alter_img_dim(target,input):
    """ implements the alteration of the image so that can used for adding

    Args:
        target (_type_): target
        input (_type_): input is the get altered using the target

    Returns:
        _type_: altered image .
    """
    target_size=target.size()[2]
    input_size=input.size()[2]
    val=input_size-target_size
    val=val//2
    return input[: ,: ,val:input_size-val,val:input_size-val,val:input_size-val]

class unet(nn.Module):
    def __init__(self,out_channels : int ):
        """ implements the unet model.

        Args:
            out_channels (int): no.of.channels for the output
        """
        super(unet,self).__init__()
        #encode
        self.encode_1= double_conv(4,64)
        self.encode_2=double_conv(64,128)
        self.encode_3=double_conv(128,256)
        self.encode_4=double_conv(256,512)
        self.encode_5=double_conv(512,1024)
        #maxpool
        self.maxpool=nn.MaxPool3d(kernel_size=(2,2,2),stride=2)
        #decode
        self.decode_trans_1= transpose(1024,512)
        self.decode_1=double_conv(1024,512)
        self.decode_trans_2= transpose(512,256)
        self.decode_2=double_conv(512,256)
        self.decode_trans_3= transpose(256,128)
        self.decode_3=double_conv(256,128)
        self.decode_trans_4= transpose(128,64)
        self.decode_4=double_conv(128,64)
        #output
        self.output=nn.Sequential(nn.Conv3d(64,out_channels,(1,1,1)),nn.Dropout3d(0.25))

    def forward(self,x):
        """ perfoms the forward function of the model.

        Args:
            x (tensor image): image 

        Returns:
            _type_: prediction of the model
        """
        out_d_1=self.encode_1(x) # to add
        out_d=self.maxpool(out_d_1)
        out_d_2=self.encode_2(out_d) # to add
        out_d=self.maxpool(out_d_2)
        out_d_3=self.encode_3(out_d)# to add
        out_d=self.maxpool(out_d_3)
        out_d_4=self.encode_4(out_d)# to add
        out_d=self.maxpool(out_d_4)
        out_d_5=self.encode_5(out_d)
        # up in unet
        out_up=self.decode_trans_1(out_d_5)
        alter= alter_img_dim(out_up,out_d_4)
        out_up=self.decode_1(torch.cat([out_up,alter],1))

        out_up=self.decode_trans_2(out_up)
        alter= alter_img_dim(out_up,out_d_3)
        out_up=self.decode_2(torch.cat([out_up,alter],1))

        out_up=self.decode_trans_3(out_up)
        alter= alter_img_dim(out_up,out_d_2)
        out_up=self.decode_3(torch.cat([out_up,alter],1))

        out_up=self.decode_trans_4(out_up)
        alter= alter_img_dim(out_up,out_d_1)
        out_up=self.decode_4(torch.cat([out_up,alter],1))

        final_out=self.output(out_up)
        return final_out