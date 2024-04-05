## Abstract
Nowadays, large language models have become increasingly prominent in various fields due to their ability to generate to give reasoning like the humans. One of the largest open-sourced language models in the current market is LLAMA2 and I worked on a developing a minimalistic version of LLAMA2 to understand the architecture in greater depth. In this project I developed the core components of LLAMA2 model for better understanding of its architecture.  

## Reference outputs/accuracies
- Text Continuation (python ```run_llama.py --option generate```) You should see continuations of the sentence ```I have wanted to see this thriller for a while, and it didn't disappoint. Keanu Reeves, playing the hero John Wick, is....``` Generated two continuations - one with temperature 0.0 (which has a reasonably coherent, if unusual, completion) and one with temperature 1.0 (which is logically inconsistent and may contain some coherence or grammar errors).
- Zero Shot Prompting Zero-Shot Prompting for SST:
  ```python run_llama.py --option prompt --batch_size 10  --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt --label-names data/sst-label-mapping.json --dev_out sst-dev-prompting-output.txt --test_out sst-test-prompting-output.txt [--use_gpu]```

  Prompting for SST: Dev Accuracy: 0.213 (0.000) Test Accuracy: 0.224 (0.000)
- Zero shot Prompting for CFIMDB:
  ```python run_llama.py --option prompt --batch_size 10  --train data/cfimdb-train.txt --dev data/cfimdb-dev.txt --test data/cfimdb-test.txt --l5abel-names data/cfimdb-label-mapping.json --dev_out cfimdb-dev-prompting-output.txt --test_out cfimdb-test-prompting-output.txt [--use_gpu]```
Prompting for CFIMDB: Dev Accuracy: 0.502 (0.000) Test Accuracy: 0.213

- Classification Finetuning
  ```python run_llama.py --option finetune --epochs 5 --lr 2e-5 --batch_size 80  --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt --label-names data/sst-label-mapping.json --dev_out sst-dev-finetuning-output.txt --test_out sst-test-finetuning-output.txt [--use_gpu]```
Finetuning for SST dev acc :: 0.414 test acc :: 0.418

```python run_llama.py --option finetune --epochs 5 --lr 2e-5 --batch_size 10  --train data/cfimdb-train.txt --dev data/cfimdb-dev.txt --test data/cfimdb-test.txt --label-names data/cfimdb-label-mapping.json --dev_out cfimdb-dev-finetuning-output.txt --test_out cfimdb-test-finetuning-output.txt [--use_gpu]```
Finetuning for CFIMDB: Dev Accuracy: 0.820 (0.115) Test Accuracy: 0.482







  
