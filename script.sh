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


python train.py --cuda --cuda_num 0 1 2 3 \
                --share_embed \
                --vocab ./data/vocab/vocab.share \
                --train_file ./data/para.pair\
                --checkpoint_path ./data/model \
                --restore_file ./data/model/checkpoint11.pkl\
                --batch_print_info 1000 \
                --grad_accum 2 \
                --max_tokens 5300 \
                --clip_length 500 \
                --discard_invalid_data


python generator.py --cuda --cuda_num 4 \
                 --file ./data/sent.pt\
                 --ref_file /home/linzhe/ACL2021/WMT14/newstest2014.tok.bpe.32000.de   \
                 --max_tokens 3000 \
                 --vocab ./data/vocab/vocab.share \
                 --decode_method beam \
                 --beam 5 \
                 --model_path ./data/model/checkpoint37.pkl \
                 --output_path ./data/output.txt \
                 --max_length 500

# de2en

python avg_param.py --input ./data/en2de/model/checkpoint89.pkl \
                            ./data/en2de/model/checkpoint85.pkl \
                            ./data/en2de/model/checkpoint86.pkl \
                            ./data/en2de/model/checkpoint87.pkl \
                            ./data/en2de/model/checkpoint88.pkl \
                    --outputata/en2de/model/checkpoint.pkl

