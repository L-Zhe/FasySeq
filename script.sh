python createVocab.py --file ./WMT14/bpe/train.tok.clean.bpe.32000.en \
                             ./WMT14/bpe/train.tok.clean.bpe.32000.de \
                      --lower --save_path ./data/vocab/vocab.share

python preprocess.py --source ./WMT14/bpe/train.tok.clean.bpe.32000.de \
                     --target ./WMT14/bpe/train.tok.clean.bpe.32000.en \
                     --src_vocab ./data/vocab/vocab.share \
                     --tgt_vocab ./data/vocab/vocab.share \
                     --save_file ./data/data.de2en

python preprocess.py --source ./WMT14/bpe/newstest2014.tok.bpe.32000.en   \
                     --src_vocab ./data/vocab/vocab.share \
                     --save_file ./data/sent.pt


python train.py --cuda_num 0 1 2 3 \
                --share_embed \
                --vocab ./data/vocab/vocab.share \
                --file ./data/data.de2en\
                --checkpoint_path ./data/model_de2en \
                --restore_file ./data/model_de2en/checkpoint21.pkl\
                --batch_print_info 1000 \
                --grad_accum 2 \
                --max_tokens 5300 \
                --discard_invalid_data


python generator.py --cuda_num 3 \
                 --raw_file ./WMT14/bpe/newstest2014.tok.bpe.32000.de \
                 --ref_file ./WMT14/newstest2014.tok.bpe.32000.en  \
                 --max_tokens 5000 \
                 --vocab ./data/vocab/vocab.share \
                 --decode_method beam \
                 --beam 5 \
                 --model_path ./data/model/checkpoint.pkl \
                 --output_path ./data/output.txt \
                 --max_add_token 50 \
                 --max_alpha 1.5

# de2en

python avg_param.py --input ./data/model/checkpoint93.pkl \
                            ./data/model/checkpoint94.pkl \
                            ./data/model/checkpoint95.pkl \
                            ./data/model/checkpoint96.pkl \
                            ./data/model/checkpoint87.pkl \
                            ./data/model/checkpoint88.pkl \
                            ./data/model/checkpoint89.pkl \
                            ./data/model/checkpoint90.pkl \
                            ./data/model/checkpoint91.pkl \
                            ./data/model/checkpoint92.pkl \
                            ./data/model/checkpoint77.pkl \
                            ./data/model/checkpoint78.pkl \
                            ./data/model/checkpoint79.pkl \
                            ./data/model/checkpoint80.pkl \
                            ./data/model/checkpoint81.pkl \
                            ./data/model/checkpoint82.pkl \
                            ./data/model/checkpoint83.pkl \
                            ./data/model/checkpoint84.pkl \
                            ./data/model/checkpoint85.pkl \
                            ./data/model/checkpoint86.pkl \

                    --output data/model/checkpoint.pkl

python avg_param.py --input ./data/model/checkpoint98.pkl \
                            ./data/model/checkpoint99.pkl \
                            ./data/model/checkpoint100.pkl \
                            ./data/model/checkpoint101.pkl \
                            ./data/model/checkpoint102.pkl \
                    --output data/model/checkpoint.pkl

python avg_param.py --input ./data/model/checkpoint110.pkl \
                            ./data/model/checkpoint106.pkl \
                            ./data/model/checkpoint107.pkl \
                            ./data/model/checkpoint108.pkl \
                            ./data/model/checkpoint109.pkl \
                    --output data/model/checkpoint.pkl

python avg_param.py --input ./data/model/checkpoint96.pkl \
                            ./data/model/checkpoint97.pkl \
                            ./data/model/checkpoint98.pkl \
                            ./data/model/checkpoint99.pkl \
                            ./data/model/checkpoint100.pkl \
                    --output data/model/checkpoint.pkl