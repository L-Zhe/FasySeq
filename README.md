# FasySeq

FasySeq is a shorthand as a **Fa**st and e**asy** sequential modeling toolkit. It aims to provide a seq2seq model to researchers and developers, which can be trained efficiently and modified easily. This toolkit is based on Transformer(), and will add more seq2seq models in the future.

## Preprocess

### Build Vocabulary

``createVocab.py``

| Named Arguments | Description                                                  |
| :-------------- | ------------------------------------------------------------ |
| -f/--file       | The files used to build the vocabulary.<br />Type: List      |
| [--vocab_num]   | The maximum size of vocabulary, the excess word will be discard according to the frequency.<br />Type: Int    Default: -1 |
| [--min_freq]    | The minimum frequency of token in vocabulary. The word with frequency less than min_freq will be discard.<br />Type: Int     Default: 0 |
| [--lower]       | Whether to convert all words to lowercase                    |
| --save_path     | The path to save voacbulary.<br />Type: str                  |

### Process Data

``preprocess.py``

| Named Arguments | Description                                         |
| :-------------- | --------------------------------------------------- |
| --source        | The path of source file.<br />Type: str             |
| [--target]      | The path of target file.<br />Type: str             |
| --src_vocab     | The path of source vocabulary.<br />Type: str       |
| [--tgt_vocab]   | The path of target vocabulary.<br />Type: str       |
| --save_path     | The path to save the processed data.<br />Type: str |

## Train

``train.py``

| Named Arguments      | Description |
| :------------------- | ----------- |
| **Model**            | -           |
| --share_embed       | Source and target share the same vocabulary and word embedding. The max position of embedding is max(max_src_position, max_tgt_position) if the model employ share embedding. |
| --max_src_position   | The maximum source position, all src-tgt pairs which source sentences' lenght are greater than max_src_position will be cut or discard. If max_src_position > max source length, it wil set as max source length.<br />Type: Int    Default: inf |
| --max_tgt_position   | The maximum target position, all src_tgt pairs which target sentences' length are greater than max_tgt_position will be cut or discard. If max_tgt_position > max target length, it wil set as max target length.<br />Type: Int    Default: inf |
| --position_method    | The method to introduce positional information.<br />Option: encoding/embedding |
| --normalize_before   |             |
|	**Checkpoint**		| - |
| --checkpoint_path |						 |
| --restore_file |						 |
| --checkpoint_num |						 |
| **Data**           | - |
| --vocab |						 |
| --src_vocab |						 |
| --tgt_vocab |						 |
| --file |						 |
| --max_tokens |						 |
| **Train**  | - |
| --cuda_num           |             |
| --grad_accumulate    |             |
| --epoch |						 |
| --batch_print_info |						 |

## Inference

``generator``

### Postprocess

``avg_param.py``

