from    torch import nn
from    .Module import (
    Encoder, Decoder, Embedding,
    PositionWiseFeedForwardNetworks,
    MultiHeadAttention, EncoderCell,
    DecoderCell)
from    copy import deepcopy
from    .SearchStrategy import SearchMethod
from    utils.tools import move2cuda, triu_mask


class transformer(nn.Module):
    def __init__(self, config):
            
        d_model  = config.embedding_dim
        num_head = config.num_head
        num_layer_encoder = config.num_layer_encoder
        num_layer_decoder = config.num_layer_decoder
        d_ff = config.d_ff
        dropout_embed = config.dropout_embed
        dropout_sublayer = config.dropout_sublayer
        share_embed = config.share_embed
        PAD_index   = config.PAD_index
        position_method = config.position_method
        max_src_position = config.max_src_position
        max_tgt_position = config.max_tgt_position
        super().__init__()

        assert d_model % num_head == 0, \
            ("Parameter Error, require embedding_dim % num head == 0.")

        d_qk = d_v = d_model // num_head
        attention = MultiHeadAttention(d_model, d_qk, d_v, num_head)
        FFN = PositionWiseFeedForwardNetworks(d_model, d_model, d_ff)
        if share_embed:
            vocab_size = config.vocab_size
            self.src_embed = Embedding(vocab_size,
                                       d_model,
                                       dropout=dropout_embed,
                                       padding_idx=PAD_index,
                                       position_method=position_method,
                                       max_length=max(max_src_position, max_tgt_position))
            self.tgt_embed = self.src_embed
            self.tgt_mask = triu_mask(max(max_src_position, max_tgt_position))

        else:
            src_vocab_size = config.src_vocab_size
            tgt_vocab_size = config.tgt_vocab_size
            self.src_embed = Embedding(src_vocab_size,
                                       d_model,
                                       dropout=dropout_embed,
                                       padding_idx=PAD_index,
                                       position_method=position_method,
                                       max_length=max_src_position)
            self.tgt_embed = Embedding(tgt_vocab_size,
                                       d_model,
                                       dropout=dropout_embed,
                                       padding_idx=PAD_index,
                                       position_method=position_method,
                                       max_length=max_tgt_position)
            self.tgt_mask = triu_mask(max_tgt_position)
            vocab_size = tgt_vocab_size
        normalize_before = config.normalize_before

        self.Encoder = Encoder(d_model=d_model, 
                               num_layer=num_layer_encoder,
                               layer=EncoderCell(d_model,
                                                 deepcopy(attention),
                                                 deepcopy(FFN),
                                                 dropout_sublayer,
                                                 normalize_before),
                               normalize_before=normalize_before)

        self.Decoder = Decoder(d_model=d_model,
                               vocab_size=vocab_size,
                               layer=DecoderCell(d_model,
                                                 deepcopy(attention),
                                                 deepcopy(FFN),
                                                 dropout_sublayer,
                                                 normalize_before),
                               num_layer=num_layer_decoder,
                               normalize_before=normalize_before)
        try:
            beam = config.beam
            search_method = config.decode_method
            self.decode_search = SearchMethod(search_method=search_method,
                                              BOS_index=config.BOS_index,
                                              EOS_index=config.EOS_index,
                                              beam=beam)
        except:
            None
        self.PAD_index = PAD_index

    def forward(self, **kwargs):

        assert kwargs['mode'] in ['train', 'test']

        if kwargs['mode'] == 'train':
            src_mask = move2cuda(move2cuda(kwargs['src_mask']))
            encoder_outputs = self.Encoder(self.src_embed(move2cuda(kwargs['source'])), src_mask)
            tgt_len = kwargs['target'].size(-1)
            outputs = self.Decoder(self.tgt_embed(move2cuda(kwargs['target'])),
                                   encoder_outputs,
                                   src_mask,
                                   self.tgt_mask[:, :tgt_len, :tgt_len].cuda())
            return outputs

        else:
            src_mask = move2cuda(kwargs['src_mask'])
            encoder_outputs = self.Encoder(self.src_embed(move2cuda(kwargs['source'])), src_mask)
            max_length = kwargs['max_length']
            return self.decode_search(decoder=self.Decoder.generate,
                                      tgt_embed=self.tgt_embed.single_embed,
                                      src_mask=src_mask,
                                      encoder_output=encoder_outputs,
                                      max_length=max_length)