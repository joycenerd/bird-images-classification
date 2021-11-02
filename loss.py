import torch
import torch.nn as nn
import matplotlib.pyplot as plt

                                                                                   
                                                                                   
class NLL_OHEM(torch.nn.NLLLoss):                                                     
    """ Online hard example mining. 
    Needs input from nn.LogSotmax() """                                             
                                                                                   
    def __init__(self, ratio,device,total_ep):      
        super(NLL_OHEM, self).__init__(None, True)                                 
        self.ratio = ratio
        self.device=device
        self.total_ep=total_ep                                                         
                                                                                   
    def forward(self, x, y, epoch,sched_ratio=True):                                                                                              
        num_inst = x.size(0)
        if sched_ratio:
            self.ratio_sched(epoch)
        else:
            self.ratio=1
        # print(self.ratio)                                                      
        num_hns = int(self.ratio * num_inst)
        if num_hns>0:                                       
            x_ = x.clone()                                                       
            inst_losses = torch.autograd.Variable(torch.zeros(num_inst).to(self.device))       
            for idx, label in enumerate(y.data):                                       
                inst_losses[idx] = -x_.data[idx, label]                                                                                 
            _, idxs = inst_losses.topk(num_hns)                                        
            x_hn = x.index_select(0, idxs)                                             
            y_hn = y.index_select(0, idxs)
            loss=torch.nn.functional.nll_loss(x_hn, y_hn,reduction='mean')
        else:
            loss=torch.nn.functional.nll_loss(x,y,reduction='mean')                                        
        return loss  

    """def ratio_sched(self,epoch):
        half=int(self.total_ep/2)
        max_range=int(half*0.2)
        if epoch<half:
            if epoch<max_range:
                self.ratio=1.0
            else:
                self.ratio=(half-epoch)/float(half-max_range)
        
        else:
            if epoch<(half+max_range):
                self.ratio=0.5
            else: 
                self.ratio=0.5*(self.total_ep-epoch)/float(half-max_range)"""
        
    """def ratio_sched(self,epoch):
        if epoch<40:
            self.ratio=1
        elif epoch>=40 and epoch<60:
            self.ratio=0.9
        elif epoch>=60 and epoch<90:
            self.ratio=0.8
        elif epoch>=90 and epoch<130:
            self.ratio=0.7
        elif epoch>=130 and epoch<170:
            self.ratio=0.6
        elif epoch>=170:
            self.ratio=0.5"""
    
    def ratio_sched(self,epoch):
        if epoch<40:
            self.ratio=1
        elif epoch>=40 and epoch<50:
            self.ratio=0.4
        elif epoch>=50 and epoch<70:
            self.ratio=0.5
        elif epoch>=70 and epoch<110:
            self.ratio=0.6
        elif epoch>=110 and epoch<150:
            self.ratio=0.7
        elif epoch>=150 and epoch<180:
            self.ratio=0.8
        elif epoch>=180:
            self.ratio=0.9



if __name__=='__main__':
    ratio_list=[]
    i_list=[]
    for i in range(0,200):
        half=100
        max_range=int(half*0.2)
        if i<half:
            if i<max_range:
                ratio=1.0
            else:
                ratio=(half-i)/float(half-max_range)
        
        else:
            if i<(half+max_range):
                ratio=0.5
            else: 
                ratio=0.5*(200-i)/float(half-max_range)
        ratio_list.append(ratio)
        i_list.append(i)

    plt.rcParams["font.family"] = "serif"
    plt.plot(i_list,ratio_list)
    plt.xlabel('epoch')
    plt.ylabel('hard mining ratio (k)')
    plt.savefig('ohem_ratio.png')
