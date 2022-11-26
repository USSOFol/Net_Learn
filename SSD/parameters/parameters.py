import torch
class Config:
    def __init__(self,random_seed=42,lr=1e-3,batch_size=64,epoch=5,img_size=32,n_classes=10,device="cuda" if torch.cuda.is_available() else "cpu"
                 ,loss_fn=torch.nn.CrossEntropyLoss()):
        self.my_random_seed=random_seed
        self.my_lr=lr
        self.my_batch_size=batch_size
        self.my_epoch=epoch
        self.my_img_size=img_size
        self.my_classes=n_classes
        self.my_device=device
        self.my_loss=loss_fn
    # if torch.cuda.is_available() else "cpu"
    def random_seed(self):
        return self.my_random_seed
    def lr(self):
        return self.my_lr
    def batch_size(self):
        return self.my_batch_size
    def epochs(self):
        return self.my_epoch
    def img_size(self):
        return self.my_img_size
    def n_classes(self):
        return self.my_classes
    def device(self):
        return self.my_device
    def loss_func(self):
        return self.my_loss
