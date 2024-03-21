# Turn-Level Active Learning for Dialogue State Tracking

This repository includes the data and code of the paper: [Turn-Level Active Learning for Dialogue State Tracking](https://arxiv.org/abs/2310.14513) (**EMNLP 2023**) by [*Zihan Zhang*](https://zhangzihangit.github.io/), *Meng Fang*, *Fanghua Ye*, *Ling Chen*, and *Mohammad-Reza Namazi-Rad*.



## Install Dependencies

The code has been tested under Python 3.7. The following are the steps to set up the environment.

Create conda environment:
```bash
conda create -n al_dst python=3.7 -y
conda activate al_dst
```

Install libraries:
```bash
pip install -r KAGE-GPT2/Src/requirements.txt
```

## Data 

Follow [KAGE-GPT2](https://github.com/LinWeizheDragon/Knowledge-Aware-Graph-Enhanced-GPT-2-for-Dialogue-State-Tracking) to download the MultiWOZ dataset at [here](https://drive.google.com/file/d/1utytDe3ojKPmDRBQgvm4gW_7Gn3AreBL/view) and unzip it.

## Model

We implement AL-DST using two base DST models, namely [KAGE-GPT2](https://github.com/LinWeizheDragon/Knowledge-Aware-Graph-Enhanced-GPT-2-for-Dialogue-State-Tracking) and [PPTOD](https://github.com/awslabs/pptod). Follow each folder to run active learning with DST.


**Run KAGE-GPT2 with AL**
```bash
python main.py ../configs/KAGE_GPT2_SparseSupervision.jsonnet --mode train --acquisition random --experiment_name mwz20/KAGE_RandomTurn --num_layer 4 --num_head 4 --num_hop 2 --graph_mode part --only_last_turn
```

**Run PPTOD with AL**
```bash
python main.py ./DST/active_learn.py --acquisition random
```

## Acknowledgement

Our data and code are based on previous works:
- [KAGE-GPT2](https://github.com/LinWeizheDragon/Knowledge-Aware-Graph-Enhanced-GPT-2-for-Dialogue-State-Tracking)
- [PPTOD](https://github.com/awslabs/pptod)


## Citation

If you find our code, data, or the paper useful, please cite the paper:

```bibtex
@inproceedings{zhang-etal-2023-turn,
    title = "Turn-Level Active Learning for Dialogue State Tracking",
    author = "Zhang, Zihan  and
      Fang, Meng  and
      Ye, Fanghua  and
      Chen, Ling  and
      Namazi-Rad, Mohammad-Reza",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.478",
    doi = "10.18653/v1/2023.emnlp-main.478",
    pages = "7705--7719",
    abstract = "Dialogue state tracking (DST) plays an important role in task-oriented dialogue systems. However, collecting a large amount of turn-by-turn annotated dialogue data is costly and inefficient. In this paper, we propose a novel turn-level active learning framework for DST to actively select turns in dialogues to annotate. Given the limited labelling budget, experimental results demonstrate the effectiveness of selective annotation of dialogue turns. Additionally, our approach can effectively achieve comparable DST performance to traditional training approaches with significantly less annotated data, which provides a more efficient way to annotate new dialogue data.",
}
```




