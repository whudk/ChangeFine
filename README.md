# ChangeFine: Fine-Grained Remote Sensing Change Detection via Change-Guided Contrastive Prompt Learning with SAM-CLIP

Official PyTorch implementation of ChangeFine:<br>
**[ChangeFine: Fine-Grained Remote Sensing Change Detection via Change-Guided Contrastive Prompt Learning with SAM-CLIP]()**
<br>
Kai Deng*, Siyuan Wei*, Xiangyun Hu<br>
(*Equal contribution)<br>

Abstract: Foundation models like SAM and CLIP are increasingly applied in computer vision due to their strong generalization and semantic understanding capabilities.
However, directly applying these models to change detection remains challenging due to semantic degradation in complex scenes, inconsistencies in multiscale structural features, 
and the ambiguity of fine-grained change localization.
To address these issues, we propose a prompt-guided supervised change detection method named ChangeFine that effectively fuses structural representations from SAM, semantic priors from CLIP, 
and change-aware prompts to achieve precise and fine-grained change localization across bi-temporal remote sensing imagery.
Specifically, we first propose the SAM-CLIP Feature Encoder (SCFE) to bridge the modality gap between structural perception and semantic understanding, fully leveraging the complementary priors of vision and language. 
To better handle ambiguous regions (hard pixels), we design a Hard Sample Contrast Module (HSCM) that dynamically mines highly uncertain samples to guide the model in learning more discriminative boundaries. 
In addition, inspired by the concept of semantic evolution, we develop a Prompt Change Decoder (PCD) that progressively reconstructs change regions from coarse to fine by combining explicit predictions and implicit semantic differences as prompts.
Extensive experiments conducted on five public remote sensing change detection datasets demonstrate that ChangePrompt achieves superior performance in terms of both accuracy and robustness. 
The code and datasets are available at https://github.com/whudk/ChangeFine.

<div align='center'>
<img src="Image/network_v1.png" alt="Architecture"  style="display: block;"/>
</div>

⚙️ Requirements
Clone the repository and set up the environment:
```bash
git clone https://github.com/your-repository/ChangeFine.git
cd ChangeFine
conda create -n changefine python=3.8
conda activate changefine
pip install -r requirements.txt
```

### 📁 Dataset Structure

Before training the **ChangeFine** model, you need to organize your dataset in the following structure:

```text
dataset/
├── train/
│   ├── A/                 # Pre-change images (t1)
│   │   ├── xxx.png
│   │   └── ...
│   ├── B/                 # Post-change images (t2)
│   │   ├── xxx.png
│   │   └── ...
│   └── label/             # Change labels
│       ├── xxx.png
│       └── ...
├── val/
│   ├── A/
│   ├── B/
│   └── label/
```


🔄 Convert Dataset to JSON
```bash
python scripts/cd2json.py --dir [path to dataset] --output_json [output json path (.eg WHUCD.json)]
```
Example:
```bash
python scripts/cd2json.py --dir /path/to/dataset --output_json /path/to/save/WHUCD.json
```


📦 Download Datasets
The five datasets used in this paper are available via the following cloud drive link:
Baidu Netdisk: https://pan.baidu.com/s/1mjZyELXo-oksg3ViOXHaZQ?pwd=whdk
Extraction Code: whdk

🚀 Pretrained Models
Download and place the following models in the pretrained/ directory:

🔸 ViT-B/16
Path: pretrained/vit_b16.pth

Baidu Netdisk: https://pan.baidu.com/s/170fxHKtMUxcDbuSkrARA5A?pwd=whdk

Code: whdk

🔸 SAM-B/16
Path: pretrained/sam_checkpoints/

Baidu Netdisk: https://pan.baidu.com/s/1DVqOGvVWyQgqMrRbEVELVA?pwd=whdk

Code: whdk

🏋️‍♀️ Training
```bash
python main.py --phase train \
               --configs [path to config] \
               --gpu 0 \
               --suptxt [path to train.json] \
               --valtxt [path to val.json] \
               --net_prompt clipsam
```
Example:
```bash
python main.py --phase train \
               --configs ./configs/default_config.yml \
               --gpu 0 \
               --suptxt ./dataset/train.json \
               --valtxt ./dataset/val.json \
               --net_prompt clipsam
```
🧪 Evaluation & Visualization
```bash
python main.py --phase test \
               --configs [path to config] \
               --gpu 0 \
               --suptxt [path to train.json] \
               --valtxt [path to val.json] \
               --net_prompt clipsam \
               --visualizer [0 or 1] \
               --resume [path to checkpoint]
```
Example (with visualization):
```bash
python main.py --phase test \
               --configs ./configs/default_config.yml \
               --gpu 0 \
               --suptxt ./dataset/train.json \
               --valtxt ./dataset/val.json \
               --net_prompt clipsam \
               --visualizer 1
```

[//]: # (## Pretrained Models)

[//]: # (We provide pretrained models of MaskDiT for ImageNet256 and ImageNet512 in the following table. For FID with guidance, the guidance scale is set to 1.5 by default.)

[//]: # (| Guidance | Resolution | FID   | Model                                                                                                                     |)

[//]: # (| :------- | :--------- | :---- | :------------------------------------------------------------------------------------------------------------------------ |)

[//]: # (| Yes      | 256x256    | 2.28  | [imagenet256-guidance.pt]&#40;https://slurm-ord.s3.amazonaws.com/ckpts/256/imagenet256-ckpt-best_with_guidance.pt&#41;       |)

[//]: # (| No       | 256x256    | 5.69  | [imagenet256-conditional.pt]&#40;https://slurm-ord.s3.amazonaws.com/ckpts/256/imagenet256-ckpt-best_without_guidance.pt&#41; |)

[//]: # (| Yes      | 512x512    | 2.50  | [imagenet512-guidance.pt]&#40;https://slurm-ord.s3.amazonaws.com/ckpts/512/1080000.pt&#41;                                   |)

[//]: # (| No       | 512x512    | 10.79 | [imagenet512-conditional.pt]&#40;https://slurm-ord.s3.amazonaws.com/ckpts/512/1050000.pt&#41;                                 |       )

### Citation
```

```



