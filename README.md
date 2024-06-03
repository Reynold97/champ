# Installation

- System requirement: Ubuntu20.04/Windows 11, Cuda 12.1
- Tested GPUs: A100, RTX3090

Install conda 

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```

Init Conda 

```bash
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
```

Restart terminal

Create conda environment:

```bash
  conda create -n champ python=3.10
  conda activate champ
```

Install packages with `pip`

```bash
  pip install -r requirements.txt
```

Install packages with [poetry](https://python-poetry.org/)
> If you want to run this project on a Windows device, we strongly recommend to use `poetry`.
```shell
poetry install --no-root
```

# Inference

The inference entrypoint script is `${PROJECT_ROOT}/inference.py`. Before testing your cases, there are two preparations need to be completed:
1. [Download all required pretrained models](#download-pretrained-models).
2. [Prepare your guidance motions](#preparen-your-guidance-motions).
2. [Run inference](#run-inference).

## Download pretrained models

You can easily get all pretrained models required by inference from our [HuggingFace repo](https://huggingface.co/fudan-generative-ai/champ).

Install git-lfs
```bash
  sudo apt update
  sudo apt install git-lfs
```

Clone the the pretrained models into `${PROJECT_ROOT}/pretrained_models` directory by cmd below:
```shell
git lfs install
git clone https://huggingface.co/fudan-generative-ai/champ pretrained_models
```

Or you can download them separately from their source repo:
   - [Champ ckpts](https://huggingface.co/fudan-generative-ai/champ/tree/main):  Consist of denoising UNet, guidance encoders, Reference UNet, and motion module.
   - [StableDiffusion V1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5): Initialized and fine-tuned from Stable-Diffusion-v1-2. (*Thanks to runwayml*)
   - [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse): Weights are intended to be used with the diffusers library. (*Thanks to stablilityai*)
   - [image_encoder](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/tree/main/image_encoder): Fine-tuned from CompVis/stable-diffusion-v1-4-original to accept CLIP image embedding rather than text embeddings. (*Thanks to lambdalabs*)

Finally, these pretrained models should be organized as follows:

```text
./pretrained_models/
|-- champ
|   |-- denoising_unet.pth
|   |-- guidance_encoder_depth.pth
|   |-- guidance_encoder_dwpose.pth
|   |-- guidance_encoder_normal.pth
|   |-- guidance_encoder_semantic_map.pth
|   |-- reference_unet.pth
|   `-- motion_module.pth
|-- image_encoder
|   |-- config.json
|   `-- pytorch_model.bin
|-- sd-vae-ft-mse
|   |-- config.json
|   |-- diffusion_pytorch_model.bin
|   `-- diffusion_pytorch_model.safetensors
`-- stable-diffusion-v1-5
    |-- feature_extractor
    |   `-- preprocessor_config.json
    |-- model_index.json
    |-- unet
    |   |-- config.json
    |   `-- diffusion_pytorch_model.bin
    `-- v1-inference.yaml
```

## Prepare your guidance motions

Guidance motion data which is produced via SMPL & Rendering is necessary when performing inference.

You can download our pre-rendered samples on our [HuggingFace repo](https://huggingface.co/datasets/fudan-generative-ai/champ_motions_example) and place into `${PROJECT_ROOT}/example_data` directory:
```shell
git lfs install
git clone https://huggingface.co/datasets/fudan-generative-ai/champ_motions_example example_data
```

Or you can follow the [SMPL & Rendering doc](https://github.com/fudan-generative-vision/champ/blob/master/docs/data_process.md) to produce your own motion datas.

Finally, the `${PROJECT_ROOT}/example_data` will be like this:
```
./example_data/
|-- motions/  # Directory includes motions per subfolder
|   |-- motion-01/  # A motion sample
|   |   |-- depth/  # Depth frame sequance
|   |   |-- dwpose/ # Dwpose frame sequance
|   |   |-- mask/   # Mask frame sequance
|   |   |-- normal/ # Normal map frame sequance
|   |   `-- semantic_map/ # Semanic map frame sequance
|   |-- motion-02/
|   |   |-- ...
|   |   `-- ...
|   `-- motion-N/
|       |-- ...
|       `-- ...
`-- ref_images/ # Reference image samples(Optional)
    |-- ref-01.png
    |-- ...
    `-- ref-N.png
```

## Run inference

Now we have all prepared models and motions in `${PROJECT_ROOT}/pretrained_models` and `${PROJECT_ROOT}/example_data` separately. 

Here is the command for inference:

```bash
  python inference.py --config configs/inference/inference.yaml
```

If using `poetry`, command is 
```shell
poetry run python inference.py --config configs/inference/inference.yaml
```

Animation results will be saved in `${PROJECT_ROOT}/results` folder. You can change the reference image or the guidance motion by modifying `inference.yaml`.

The default motion-02 in `inference.yaml` has about 250 frames, requires ~20GB VRAM.

**Note**: If your VRAM is insufficient, you can switch to a shorter motion sequence or cut out a segment from a long sequence. We provide a frame range selector in `inference.yaml`, which you can replace with a list of `[min_frame_index, max_frame_index]` to conveniently cut out a segment from the sequence.

# Train the Model

The training process consists of two distinct stages. For more information, refer to the `Training Section` in the [paper on arXiv](https://arxiv.org/abs/2403.14781).

## Prepare Datasets

Prepare your own training videos with human motion (or use [our sample training data on HuggingFace](https://huggingface.co/datasets/fudan-generative-ai/champ_trainning_sample)) and modify `data.video_folder` value in training config yaml.

All training videos need to be processed into SMPL & DWPose format. Refer to the [Data Process doc](https://github.com/fudan-generative-vision/champ/blob/master/docs/data_process.md).

The directory structure will be like this:
```txt
/training_data/
|-- video01/          # A video data frame
|   |-- depth/        # Depth frame sequance
|   |-- dwpose/       # Dwpose frame sequance
|   |-- mask/         # Mask frame sequance
|   |-- normal/       # Normal map frame sequance
|   `-- semantic_map/ # Semanic map frame sequance
|-- video02/
|   |-- ...
|   `-- ...
`-- videoN/
|-- ...
`-- ...
```

Select another small batch of data as the validation set, and modify the `validation.ref_images` and `validation.guidance_folders` roots in training config yaml.

## Run Training Scripts

To train the Champ model, use the following command:
```shell
# Run training script of stage1
accelerate launch train_s1.py --config configs/train/stage1.yaml

# Modify the `stage1_ckpt_dir` value in yaml and run training script of stage2
accelerate launch train_s2.py --config configs/train/stage2.yaml
```

# Datasets

| Type | HuggingFace |       ETA       |
| :----: | :----------------------------------------------------------------------------------------- | :-------------: |
|   Inference   | **[SMPL motion samples](https://huggingface.co/datasets/fudan-generative-ai/champ_motions_example)** | Thu Apr 18 2024 |
|   Training | **[Sample datasets for Training](https://huggingface.co/datasets/fudan-generative-ai/champ_trainning_sample)** | Sun May 05 2024 |
# Roadmap

| Status | Milestone                                                                                  |       ETA       |
| :----: | :----------------------------------------------------------------------------------------- | :-------------: |
|   ✅   | **[Inference source code meet everyone on GitHub first time](https://github.com/fudan-generative-vision/champ)** | Sun Mar 24 2024 |
|   ✅   | **[Model and test data on Huggingface](https://huggingface.co/fudan-generative-ai/champ)** | Tue Mar 26 2024 |
|   ✅   | **[Optimize dependencies and go well on Windows](https://github.com/fudan-generative-vision/champ?tab=readme-ov-file#installation)** | Sun Mar 31 2024 |
|   ✅   | **[Data preprocessing code release](https://github.com/fudan-generative-vision/champ/blob/master/docs/data_process.md)**                                                    | Fri Apr 12 2024 |
|   ✅   | **[Training code release](https://github.com/fudan-generative-vision/champ/pull/99)**                                                  | Thu May 02 2024 |
|   ✅   | **[Sample of training data release on HuggingFace](https://huggingface.co/datasets/fudan-generative-ai/champ_trainning_sample)**                                                  | Sun May 05 2024 |
|   ✅  | **[Smoothing SMPL motion](https://github.com/fudan-generative-vision/champ/pull/96)**                                                  | Sun Apr 28 2024 |
|   🚀🚀🚀  | **[Gradio demo on HuggingFace]()**                                                  | TBD |

# Citation

If you find our work useful for your research, please consider citing the paper:

```
@misc{zhu2024champ,
      title={Champ: Controllable and Consistent Human Image Animation with 3D Parametric Guidance},
      author={Shenhao Zhu and Junming Leo Chen and Zuozhuo Dai and Yinghui Xu and Xun Cao and Yao Yao and Hao Zhu and Siyu Zhu},
      year={2024},
      eprint={2403.14781},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

# Opportunities available

Multiple research positions are open at the **Generative Vision Lab, Fudan University**! Include:

- Research assistant
- Postdoctoral researcher
- PhD candidate
- Master students

Interested individuals are encouraged to contact us at [siyuzhu@fudan.edu.cn](mailto://siyuzhu@fudan.edu.cn) for further information.