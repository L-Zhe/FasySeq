# FasySeq



FasySeq is a shorthand as a **Fas**t and e**asy** sequential modeling toolkit. It aims to provide a seq2seq model to researchers and developers, which can be trained efficiently and modified easily. This toolkit is based on Transformer([Vaswani et al.](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)), and will add more seq2seq models in the future.

## Dependency

```
PyTorch >= 1.4
NLTK
```

## Result

...

## Structure

...

## To Be Updated

* top-k and top-p sampling
* multi-GPU inference
* length penalty in beam search
* ...

## Preprocess

### Build Vocabulary

``createVocab.py``

| NamedArguments | Description                                  |
| :------------------------------------------------ | :------------------------------------------------------------ |
| -f/--file                       | The files used to build the vocabulary.<br />``Type: List``  |
| --vocab_num                     | The maximum size of vocabulary, the excess word will be discard according to the frequency.<br />``Type: Int``  ``Default: -1`` |
| --min_freq                      | The minimum frequency of token in vocabulary. The word with frequency less than min_freq will be discard.<br />``Type: Int``   ``Default: 0`` |
| --lower                         | Whether to convert all words to lowercase                    |
| --save_path                     | The path to save voacbulary.<br />``Type: str``              |

### Process Data

``preprocess.py``

| NamedArguments | Description                                             |
| :------------- | ------------------------------------------------------- |
| --source       | The path of source file.<br />``Type: str``             |
| [--target]     | The path of target file.<br />``Type: str``             |
| --src_vocab    | The path of source vocabulary.<br />``Type: str``       |
| [--tgt_vocab]  | The path of target vocabulary.<br />``Type: str``       |
| --save_path    | The path to save the processed data.<br />``Type: str`` |

## Train

``train.py``

| NamedArguments | Description |
| :------------------- | ----------- |
| **Model**            | -           |
| --share_embed       | Source and target share the same vocabulary and word embedding. The max position of embedding is max(max_src_position, max_tgt_position) if the model employ share embedding. |
| --max_src_position   | The maximum source position, all src-tgt pairs which source sentences' lenght are greater than max_src_position will be cut or discard. If max_src_position > max source length, it wil be set to max source length.<br />``Type: Int``  ``Default: inf`` |
| --max_tgt_position   | The maximum target position, all src_tgt pairs which target sentences' length are greater than max_tgt_position will be cut or discard. If max_tgt_position > max target length, it wil be set to max target length.<br />``Type: Int``  ``Default: inf`` |
| --position_method    | The method to introduce positional information.<br />``Option: encoding/embedding`` |
| --normalize_before   | Leveraging before layer normalization. See [Xiong et al.](https://openreview.net/forum?id=B1x8anVFPr) |
|	**Checkpoint**		| - |
| --checkpoint_path | The path to save checkpoint file.<br />``Type: str``  ``Default: None`` |
| --restore_file | The checkpoint file to be loaded.<br />``Type: str`` ``Default: None`` |
| --checkpoint_num | Save the nearest *checkpoint_num* breakpoint.<br />``Type: Int`` ``Default: inf`` |
| **Data**           | - |
| --vocab | Vocabulary path. If you use share embedding, the vocabulary will be loaded from this path.<br />``Type: str`` ``Default: None`` |
| --src_vocab | Source vocabulary path.<br />``Type: str`` ``Default: None`` |
| --tgt_vocab | Target vocabulary path.<br />``Type: str`` ``Default: None`` |
| --file | The training data file.<br />``Type: str`` |
| --max_tokens | The maximum tokens in each batch.<br />``Type: Int`` ``Default: 1000`` |
| --discard_invalid_data | The data which length of source or data is more than maximum position will be discard if use this option, otherwise the long sentences will be cut into max position. |
| **Train**  | - |
| --cuda_num           | The device's ID of GPU.<br />``Type: List`` |
| --grad_accumulate    | The num of gradient accumulate.<br />``Type: Int`` ``Default: 1`` |
| --epoch | The total epoch to train.<br />``Type: Int``  ``Default: inf`` |
| --batch_print_info | The number of batch to print training information.<br />``Type: Int`` ``Default: 1000`` |

## Inference

``generator.py``

| NamedArguments                                     | Description                                                  |
| -------------------------------------------------- | ------------------------------------------------------------ |
| --cuda_num                                         | The device's ID of GPU.<br />``Type: List``                  |
| --file                                             | The inference data file which has been processed.<br />``Type: str`` |
| --raw_file                                         | The raw inference data file, and will be preprocessed before generated.<br />``Type: str`` |
| --ref_file                                         | The reference file.<br />``Type: str``                       |
| --max_length<br />--max_alpha<br />--max_add_token | Maximum generated length = min(max_length, max_alpha * max_src_len, max_add_token + max_src_token)<br />``Type: Int`` ``Default: inf`` |
| --max_tokens                                       | The maximum tokens in each batch.<br />``Type: Int`` ``Default: 1000`` |
| --src_vocab                                        | Source vocabulary path.<br />``Type: str`` ``Default: None`` |
| --tgt_vocab                                        | Target vocabulary path.<br />``Type: str`` ``Default: None`` |
| --vocab                                            | Vocabulary path. If you use share embedding, the vocabulary will be loaded from this path.<br />``Type: str`` ``Default: None`` |
| --model_path                                       | The path of pre-trained model.<br />``Type: str``            |
| --output_path                                      | The path of output. the result will be saved into ``output_path/result.txt``.<br />``Type: str`` |
| --decode_method                                    | The decode method.<br />``Option:greedy/beam``               |
| --beam                                             | Beam size.<br />``Type: Int`` ``Default: 5``                 |



## Postpreposs

``avg_param.py``

The average parameter code we employed is the same as [fairseq](https://github.com/pytorch/fairseq).

## License

FasySeq(-py) is [Apache-2.0 License](https://github.com/L-Zhe/FasySeq/blob/main/LICENSE). The license applies to the pre-trained models as well.

