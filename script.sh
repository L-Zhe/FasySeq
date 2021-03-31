python createVocab.py --file /home/linzhe/ACL2021/WMT14/bpe/train.tok.clean.bpe.32000.en \
                             /home/linzhe/ACL2021/WMT14/bpe/train.tok.clean.bpe.32000.de \
                      --save_path ./data/vocab/vocab.share

python preprocess.py --source /home/linzhe/ACL2021/WMT14/bpe/train.tok.clean.bpe.32000.en \
                     --target /home/linzhe/ACL2021/WMT14/bpe/train.tok.clean.bpe.32000.de \
                     --src_vocab ./data/vocab/vocab.share \
                     --tgt_vocab ./data/vocab/vocab.share \
                     --save_file ./data/para.pair

python preprocess.py --source /home/linzhe/ACL2021/WMT14/bpe/newstest2014.tok.bpe.32000.en   \
                     --src_vocab ./data/vocab/vocab.share \
                     --save_file ./data/sent.pt


python train.py --cuda --cuda_num 4 5 \
                --share_embed \
                --vocab ./data/vocab/vocab.share \
                --file ./data/para.pair\
                --checkpoint_path ./data/model \
                --restore_file ./data/model/checkpoint21.pkl\
                --batch_print_info 1000 \
                --grad_accum 2 \
                --max_tokens 5300 \
                --discard_invalid_data


python generator.py --cuda --cuda_num 4 \
                 --raw_file /home/linzhe/ACL2021/WMT14/bpe/newstest2014.tok.bpe.32000.en \
                 --ref_file /home/linzhe/ACL2021/WMT14/newstest2014.tok.bpe.32000.de   \
                 --max_tokens 5000 \
                 --vocab ./data/vocab/vocab.share \
                 --decode_method beam \
                 --beam 5 \
                 --model_path ./data/model/checkpoint.pkl \
                 --output_path ./data/output.txt \
                 --max_length 500

# de2en

python avg_param.py --input ./data/model/checkpoint83.pkl \
                            ./data/model/checkpoint84.pkl \
                            ./data/model/checkpoint85.pkl \
                            ./data/model/checkpoint86.pkl \
                            ./data/model/checkpoint87.pkl \
                            ./data/model/checkpoint88.pkl \
                            ./data/model/checkpoint89.pkl \
                            ./data/model/checkpoint90.pkl \
                            ./data/model/checkpoint91.pkl \
                            ./data/model/checkpoint92.pkl \
                    --output data/model/checkpoint.pkl

