import torch
import numpy as np
import torch.nn as nn
import math


# some code adapted from https://wmathor.com/index.php/archives/1455/


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, n_heads):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.n_heads = n_heads

    def forward(self, Q, K, V):
        '''
        Q: [batch_size, n_heads, len_q=1, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(
            self.d_k)  # scores : [batch_size, n_heads, len_q, len_k]

        attn = nn.Softmax(dim=-1)(scores)  # [batch_size, n_heads, len_q, len_q]
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, len_q, len_k):
        super(MultiHeadAttention, self).__init__()

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.ScaledDotProductAttention = ScaledDotProductAttention(
            self.d_k, n_heads)
        self.len_q = len_q
        self.len_k = len_k

    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        # Q: [batch_size, n_heads, len_q, d_k]
        Q = self.W_Q(input_Q).view(batch_size, -1,
                                   self.n_heads, self.d_k).transpose(1, 2)

        # K: [batch_size, n_heads, len_k, d_k]
        K = self.W_K(input_K).view(batch_size, -1,
                                   self.n_heads, self.d_k).transpose(1, 2)

        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        V = self.W_V(input_V).view(batch_size, -1,
                                   self.n_heads, self.d_v).transpose(1, 2)

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = self.ScaledDotProductAttention(Q, K, V)
        # context: [batch_size, len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(
            batch_size, -1, self.n_heads * self.d_v)
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(self.d_model).cuda()(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.d_model = d_model

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(
            self.d_model).cuda()(
            output + residual)  # [batch_size, seq_len, d_model]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_heads, len_q):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(
            d_model, d_k, d_v, n_heads, 1, len_q)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(
            enc_inputs, enc_inputs, enc_inputs)  # enc_inputs to same Q,K,V
        # enc_outputs: [batch_size, src_len, d_model]
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_layers, n_heads, len_q):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, d_ff, d_k, d_v, n_heads, len_q)
             for _ in range(n_layers)])

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        '''
        enc_outputs = enc_inputs
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_heads, len_q):
        super(DecoderLayer, self).__init__()
        self.dec_enc_attn = MultiHeadAttention(
            d_model, d_k, d_v, n_heads, 1, len_q)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, dec_inputs, enc_outputs):
        '''
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        '''
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(
            dec_inputs, enc_outputs, enc_outputs)
        # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_enc_attn


class Decoder(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_layers, n_heads, len_q):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, d_ff, d_k, d_v, n_heads, len_q)
             for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_outputs):
        '''
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_intpus: [batch_size, src_len, d_model]
        enc_outputs: [batsh_size, src_len, d_model]
        '''
        dec_outputs = dec_inputs  # self.tgt_emb(dec_inputs) # [batch_size, tgt_len, d_model]
        # dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).cuda() # [batch_size, tgt_len, tgt_len]

        dec_enc_attns = []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_enc_attn = layer(dec_outputs, enc_outputs)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs


# d_model,   Embedding Size
# d_ff, FeedForward dimension
# d_k = d_v,   dimension of K(=Q), V
# n_layers,   number of Encoder of Decoder Layer
# n_heads,   number of heads in Multi-Head Attention

class Transformer2_3_1(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_layers, n_heads, len_q):
        super(Transformer2_3_1, self).__init__()
        self.encoder = Encoder(d_model, d_ff, d_k, d_v,
                               n_layers, n_heads, len_q).cuda()
        self.decoder = Decoder(d_model, d_ff, d_k, d_v,
                               1, n_heads, len_q).cuda()

    def forward(self, enc_inputs, dec_inputs):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        '''
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        dec_outputs = self.decoder(dec_inputs, enc_outputs)
        return dec_outputs


class Transformer(nn.Module):
    def __init__(self, hprms):
        super(Transformer, self).__init__()
        self.num_f_maps = hprms.mstcn_f_maps  # 32
        self.dim = hprms.mstcn_f_dim  # 2048
        self.num_classes = hprms.num_classes  # 3
        self.sequence_length = hprms.transformer_seq_length

        self.transformer = Transformer2_3_1(
            d_model=self.num_classes, d_ff=self.num_f_maps, d_k=self.num_f_maps,
            d_v=self.num_f_maps, n_layers=hprms.n_layers, n_heads=hprms.n_heads,
            len_q=self.sequence_length)
        self.fc = nn.Linear(self.dim, self.num_classes, bias=False)

    def forward(self, batch):
        spatial_features, temporal_features, targets = batch
        inputs = []
        for i in range(temporal_features.size(1)):
            if i < self.sequence_length-1:
                input = torch.zeros(
                    (1, self.sequence_length-1-i, self.num_classes)).cuda()
                input = torch.cat([input, temporal_features[:, 0:i+1]], dim=1)
            else:
                input = temporal_features[:, i-self.sequence_length+1:i+1]
            inputs.append(input)
        inputs = torch.stack(inputs, dim=0).squeeze(1)
        feas = torch.tanh(self.fc(spatial_features).transpose(0, 1))
        preds = self.transformer(inputs, feas)
        return {"preds": preds, "targets": targets}

    @staticmethod
    def add_specific_args(parser):  # pragma: no cover
        transformer_model_specific_args = parser.add_argument_group(
            title='mstcn reg specific args options')
        # mstcn_reg_model_specific_args.add_argument("--mstcn_stages",
        #                                            default=4,
        #                                            type=int)
        # mstcn_reg_model_specific_args.add_argument("--mstcn_layers",
        #                                            default=10,
        #                                            type=int)
        transformer_model_specific_args.add_argument("--mstcn_f_maps",
                                                     default=64,
                                                     type=int)
        transformer_model_specific_args.add_argument("--mstcn_f_dim",
                                                     default=2048,
                                                     type=int)
        transformer_model_specific_args.add_argument("--mstcn_causal_conv",
                                                     action='store_true')
        transformer_model_specific_args.add_argument("--n_layers",
                                                     default=1,
                                                     type=int)
        transformer_model_specific_args.add_argument("--n_heads",
                                                     default=8,
                                                     type=int)
        transformer_model_specific_args.add_argument("--transformer_seq_length",
                                                     default=30,
                                                     type=int)
        return parser
