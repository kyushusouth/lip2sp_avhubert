fairseq-hydra-train --config-dir ./conf/pretrain \
  --config-name base_lrs3_iter1 \
  common.user_dir=`pwd` \
  task.data=/path/to/data task.label_dir=/path/to/label 
  # task.tokenizer_bpe_model=/path/to/tokenizer model.w2v_path=/path/to/checkpoint \
  # hydra.run.dir=/path/to/experiment/finetune
