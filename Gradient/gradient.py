import torch
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F

########################
## w_tri ## triplet weight
########################
def weight_triplet_con(Pos, Neg, *args):
    return torch.ones_like(Pos).unsqueeze(1)

def weight_triplet_nca(Pos, Neg, tau=10, *args):
    PN = Pos-Neg # similarity difference
    return (1/(1+torch.exp(tau * PN))).unsqueeze(1)

def weight_triplet_cir(Pos, Neg, tau=10, *args):
    PN = Pos*(2-Pos)-Neg*Neg-0.5 # similarity difference #0.74284
    return (1/(1+torch.exp(tau * PN))).unsqueeze(1)

########################
## w_pair ## Pos/Neg pairs weight
########################
def weight_pair_con(Pos, Neg, param_dict):
    wp = torch.ones_like(Pos)
    wn = torch.ones_like(Neg)
    return wp.unsqueeze(1), wn.unsqueeze(1)

def weight_pair_lin(Pos, Neg, param_dict):
    wp = (1-Pos)
    wn = Neg
    return wp.unsqueeze(1), wn.unsqueeze(1)

def weight_pair_lin_ms(Pos, Neg, Pos_rs, Neg_rs, param_dict):
    wp = (1-Pos_rs)*(1-Pos)
    wn = (1+Neg_rs)*Neg
    return wp.unsqueeze(1), wn.unsqueeze(1)

def weight_pair_sig(Pos, Neg, param_dict):
    wp = 1/( ( param_dict['ms_a']*(Pos-param_dict['ms_l'])).exp() + 1 )
    wn = 1/( (-param_dict['ms_b']*(Neg-param_dict['ms_l'])).exp() + 1 )
    return wp.unsqueeze(1), wn.unsqueeze(1)

def weight_pair_sig_ms(Pos, Neg, Pos_rs, Neg_rs, param_dict):
    wp = 1/( ( param_dict['ms_a']*(Pos-param_dict['ms_l'])).exp() + Pos_rs )
    wn = 1/( (-param_dict['ms_b']*(Neg-param_dict['ms_l'])).exp() + Neg_rs )
    return wp.unsqueeze(1), wn.unsqueeze(1)

def sig_ms_term(N, Sim, V_pos_max, V_neg_max, I_neg_max):
    
    Pos_mat = V_pos_max.repeat(N,1)
    
    th_pos = V_pos_max.min().expand_as(V_pos_max).unsqueeze(1)-0.1
    th_neg = V_neg_max.unsqueeze(1)+0.1
    
    # relative positive/negative set values
    Pos_set_val = ( ms_a*(V_pos_max.unsqueeze(1)-Pos_mat)).exp()
    Neg_set_val = (-ms_b*(V_neg_max.unsqueeze(1)-Sim)).exp()
        
    # select valid positive/negative set
    Pos_set_val[Pos_mat>th_neg]=0
    Pos_set_val.fill_diagonal_(0)
    
    Neg_set_val[Sim<th_pos]=0
    Neg_set_val.fill_diagonal_(0)
    Neg_set_val[torch.arange(0,N), I_neg_max]=0
    
    # count valid positive/negative set
    Pos_set_size = (Pos_set_val!=0).float().sum(1)
    Neg_set_size = (Neg_set_val>0).float().sum(1)
    
    # average relative positive/negative similarity
    Pos_rs = Pos_set_val.sum(1)/Pos_set_size
    Pos_rs[Pos_set_size==0]=0
    
    Neg_rs = Neg_set_val.sum(1)/Neg_set_size
    Neg_rs[Neg_set_size==0]=0
    
    return Pos_rs, Neg_rs

def lin_ms_term(N, Sim, V_pos_max, V_neg_max, I_neg_max):
    
    Pos_mat = V_pos_max.repeat(N,1)
    
    th_pos = V_pos_max.min().expand_as(V_pos_max).unsqueeze(1)-0.1
    th_neg = V_neg_max.unsqueeze(1)+0.1
    
    # relative positive/negative set values
    Pos_set_val = V_pos_max.unsqueeze(1)-Pos_mat
    Neg_set_val = V_neg_max.unsqueeze(1)-Sim#>0
    
    # select valid positive/negative set
    Pos_set_val[Pos_mat>th_neg]=0
    Pos_set_val.fill_diagonal_(0)
    
    Neg_set_val[Sim<th_pos]=0
    Neg_set_val.fill_diagonal_(0)
    Neg_set_val[torch.arange(0,N), I_neg_max]=0
    
    # count valid positive/negative set
    Pos_set_size = (Pos_set_val!=0).float().sum(1)
    Neg_set_size = (Neg_set_val>0).float().sum(1)
    
    # average relative positive/negative similarity
    Pos_rs = Pos_set_val.sum(1)/Pos_set_size
    Pos_rs[Pos_set_size==0]=0
    
    Neg_rs = Neg_set_val.sum(1)/Neg_set_size
    Neg_rs[Neg_set_size==0]=0
            
    return Pos_rs, Neg_rs

########################
# gradient direction
def grad_dir_cos(fvec_img, fvec_txt, Neg_img_idx, Neg_txt_idx, wp_it, wn_it, wp_ti, wn_ti, wt_it, wt_ti):
    G_img = wt_it * (wn_it*fvec_txt[Neg_txt_idx,:] - wp_it*fvec_txt ) - wt_ti * ( wp_ti*fvec_txt ) # G_anc_img = txt + txt feature = txt feautre
    G_txt = wt_ti * (wn_ti*fvec_img[Neg_img_idx,:] - wp_ti*fvec_img ) - wt_it * ( wp_it*fvec_img ) # G_anc_txt = img + img feature = img feautre
    
    N = fvec_img.size(0)
    
    for i in range(N):           
        index_neg = Neg_img_idx[i]
        G_img[index_neg,:] += wt_ti[i]*wn_ti[i]*fvec_txt[i,:]
        
    for i in range(N):           
        index_neg = Neg_txt_idx[i]
        G_txt[index_neg,:] += wt_it[i]*wn_it[i]*fvec_img[i,:]

    return G_img, G_txt

########################
# gradient element selection 
def weight_triplet(mode):
    if mode=='con': 
        return weight_triplet_con
    elif mode=='nca':
        return weight_triplet_nca
    elif mode=='cir':
        return weight_triplet_cir
    
def weight_pair(mode):
    if mode=='con':
        return weight_pair_con
    elif mode=='lin':
        return weight_pair_lin
    elif mode=='lin_ms':
        return weight_pair_lin_ms
    elif mode=='sig':
        return weight_pair_sig
    elif mode=='sig_ms':
        return weight_pair_sig_ms

def grad_dir(mode):
    # gradient distance
    if mode=='cos':
        return grad_dir_cos
        
class TripletGrad(Module):
        
    def __init__(self, gmode, tau=1, ms_a=2, ms_b=10, ms_l=0.5):
        super(TripletGrad, self).__init__()
        # gradient mode
        mode_tp, mode_pr, mode_gd = gmode

        # rs: relative similarity
        if mode_pr=='sig_ms':
            fun_rs = sig_ms_term
        elif mode_pr=='lin_ms':
            fun_rs = lin_ms_term
        else:
            fun_rs = None
        
        self.fun_dict = {'wt': weight_triplet(mode_tp),
                         'wp': weight_pair(mode_pr),
                         'rs': fun_rs
                         'gd': grad_dir(mode_gd),
                         }
        
        self.param_dict = {'tau': tau,
                           'ms_a': ms_a,
                           'ms_b': ms_b,
                           'ms_l': ms_l,
                           }

        self.gradfun = TripletGradFun()

    def forward(self, input_img, input_txt):
        return self.gradfun.apply(input_img, input_txt, self.fun_dict, self.param_dict)

class TripletGradFun(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, fvec_img, fvec_txt, fun_dict, param_dict):
 
        ############################################
        # preprocessing
        N = fvec_img.size(0)
        
        # Pos value
        Pos = torch.mul(fvec_img, fvec_txt).sum(1)

        # Similarity Matrix
        scores = torch.mm(fvec_img, fvec_txt.t()).fill_diagonal_(-1)

        # Neg value
        Neg_txt, Neg_txt_idx = scores.max(1)
        Neg_img, Neg_img_idx = scores.max(0)

        ############################################
        # triplet weight for image
        wt_it = fun_dict['wt'](Pos, Neg_txt, param_dict['tau'])
        wt_ti = fun_dict['wt'](Pos, Neg_img, param_dict['tau'])

        # pair weight for image
        if fun_rs:
            # input: N, Sim, V_pos_max, V_neg_max, I_neg_max
            Pos_it_rs, Neg_it_rs = fun_dict['rs'](N, scores,     Pos, Neg_txt, Neg_txt_idx)
            Pos_ti_rs, Neg_ti_rs = fun_dict['rs'](N, scores.t(), Pos, Neg_img, Neg_img_idx)
            wp_it, wn_it = fun_dict['wp'](Pos, Neg_txt, Pos_it_rs, Neg_it_rs, param_dict)
            wp_ti, wn_ti = fun_dict['wp'](Pos, Neg_img, Pos_ti_rs, Neg_ti_rs, param_dict)
        else:
            wp_it, wn_it = fun_dict['wp'](Pos, Neg_txt, param_dict)
            wp_ti, wn_ti = fun_dict['wp'](Pos, Neg_img, param_dict)
        
        # gradient direction->txt, img, f_anc_txt, f_anc_img
        G_img, G_txt = fun_dict['gd'](fvec_img, fvec_txt, Neg_img_idx, Neg_txt_idx, wp_it, wn_it, wp_ti, wn_ti, wt_it, wt_ti)
        
        ctx.save_for_backward(G_img, G_txt)
        
        ############################################
        # origin triplets
        T_it = (Neg_txt - Pos).sum()
        T_ti = (Neg_img - Pos).sum()
        
        # loss
        loss = T_it+T_ti

        loss_log = {"loss_it": T_it, 
                    "loss_ti": T_ti
                    "positive": Pos, 
                    "negative_txt": Neg_txt,  
                    "negative_img": Neg_img, 
                    }
            
        return loss, loss_log
    
    @staticmethod
    def backward(ctx, grad_output, no_use):
        
        G_img, G_txt = ctx.saved_tensors
        return G_img, G_txt, None, None