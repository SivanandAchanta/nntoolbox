import torch
import torch.nn as nn
import torch.nn.functional as F
from layernorm import LayerNorm

class Transformer(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, num_layers, p_dp=0.1):

        super(Transformer, self).__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_k = int(self.d_model/self.num_heads)
        self.sqrt_dk = 8
        self.dropout_p = p_dp

        self.WQ_1 = nn.Linear(self.d_model,self.d_k)
        self.WK_1 = nn.Linear(self.d_model,self.d_k)
        self.WV_1 = nn.Linear(self.d_model,self.d_k)

        self.WQ_2 = nn.Linear(self.d_model,self.d_k)
        self.WK_2 = nn.Linear(self.d_model,self.d_k)
        self.WV_2 = nn.Linear(self.d_model,self.d_k)

        self.WQ_3 = nn.Linear(self.d_model,self.d_k)
        self.WK_3 = nn.Linear(self.d_model,self.d_k)
        self.WV_3 = nn.Linear(self.d_model,self.d_k)

        self.WQ_4 = nn.Linear(self.d_model,self.d_k)
        self.WK_4 = nn.Linear(self.d_model,self.d_k)
        self.WV_4 = nn.Linear(self.d_model,self.d_k)

        self.WQ_5 = nn.Linear(self.d_model,self.d_k)
        self.WK_5 = nn.Linear(self.d_model,self.d_k)
        self.WV_5 = nn.Linear(self.d_model,self.d_k)

        self.WQ_6 = nn.Linear(self.d_model,self.d_k)
        self.WK_6 = nn.Linear(self.d_model,self.d_k)
        self.WV_6 = nn.Linear(self.d_model,self.d_k)

        self.WQ_7 = nn.Linear(self.d_model,self.d_k)
        self.WK_7 = nn.Linear(self.d_model,self.d_k)
        self.WV_7 = nn.Linear(self.d_model,self.d_k)

        self.WQ_8 = nn.Linear(self.d_model,self.d_k)
        self.WK_8 = nn.Linear(self.d_model,self.d_k)
        self.WV_8 = nn.Linear(self.d_model,self.d_k)

        self.WO = nn.Linear(self.d_model,self.d_model)
 
        self.smax = nn.Softmax()
        self.dp1 = nn.Dropout(self.dropout_p)
 
        self.ln1 = LayerNorm(self.d_model)
        self.l1 = nn.Linear(self.d_model, self.d_ff)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(self.d_ff, self.d_model)
        self.dp2 = nn.Dropout(self.dropout_p)
        self.ln2 = LayerNorm(self.d_model) 
          

    def forward(self, x):
        """
            :param x: tensor with shape of [batch_size, size]
            :return: tensor with shape of [batch_size, size]
            applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | G and Q is affine transformation,
            f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
            and ⨀ is element-wise multiplication
            """

        #for layer in range(self.num_layers):
        
 
        # Multi-Head Attention
        Q1 = self.WQ_1(x)
        K1 = self.WK_1(x)
        V1 = self.WV_1(x)
        attn_weights1 = self.smax(torch.mm(Q1,K1.transpose(0,1))/self.sqrt_dk)
        h1 = torch.mm(attn_weights1, V1) 
 
        Q2 = self.WQ_2(x)
        K2 = self.WK_2(x)
        V2 = self.WV_2(x)
        attn_weights2 = self.smax(torch.mm(Q2,K2.transpose(0,1))/self.sqrt_dk)
        h2 = torch.mm(attn_weights2, V2) 

        Q3 = self.WQ_3(x)
        K3 = self.WK_3(x)
        V3 = self.WV_3(x)
        attn_weights3 = self.smax(torch.mm(Q3,K3.transpose(0,1))/self.sqrt_dk)
        h3 = torch.mm(attn_weights3, V3) 

        Q4 = self.WQ_4(x)
        K4 = self.WK_4(x)
        V4 = self.WV_4(x)
        attn_weights4 = self.smax(torch.mm(Q4,K4.transpose(0,1))/self.sqrt_dk)
        h4 = torch.mm(attn_weights4, V4) 

        Q5 = self.WQ_5(x)
        K5 = self.WK_5(x)
        V5 = self.WV_5(x)
        attn_weights5 = self.smax(torch.mm(Q5,K5.transpose(0,1))/self.sqrt_dk)
        h5 = torch.mm(attn_weights5, V5) 

        Q6 = self.WQ_6(x)
        K6 = self.WK_6(x)
        V6 = self.WV_6(x)
        attn_weights6 = self.smax(torch.mm(Q6,K6.transpose(0,1))/self.sqrt_dk)
        h6 = torch.mm(attn_weights6, V6) 

        Q7 = self.WQ_7(x)
        K7 = self.WK_7(x)
        V7 = self.WV_7(x)
        attn_weights7 = self.smax(torch.mm(Q7,K7.transpose(0,1))/self.sqrt_dk)
        h7 = torch.mm(attn_weights7, V7) 

        Q8 = self.WQ_8(x)
        K8 = self.WK_8(x)
        V8 = self.WV_8(x)
        attn_weights8 = self.smax(torch.mm(Q8,K8.transpose(0,1))/self.sqrt_dk)
        h8 = torch.mm(attn_weights8, V8) 

        mh = torch.cat((h1,h2,h3,h4,h5,h6,h7,h8), 1)
        bln1 = self.WO(mh)

        # Dropout and Add and Norm  
        bln1 = self.dp1(bln1)
        bln2 = bln1 + x 
        sl1 = self.ln1(bln2)
 
        # Position wise Feed Forward NN
        a1 = self.l1(sl1)
        a2 = self.relu(a1)
        a3 = self.l2(a2)

        # Dropout and Add and Norm
        a3 = self.dp2(a3)
        a4 = a3 + sl1 
        a5 = self.ln2(a4)

        return a5
