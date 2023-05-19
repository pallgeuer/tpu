# Open-Vocabulary Detection via Vision and Language Knowledge Distillation
• [Paper](https://arxiv.org/abs/2104.13921) • [Colab](https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/models/official/detection/projects/vild/ViLD_demo.ipynb)

<p style="text-align:center;"><img src="https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/vild/assets/new_teaser.png" alt="teaser" width="500"/></p>

Xiuye Gu, Tsung-Yi Lin, Weicheng Kuo, Yin Cui,
[Open-Vocabulary Detection via Vision and Language Knowledge Distillation](https://arxiv.org/abs/2104.13921).

Repository modified by Philipp Allgeuer.

This repo contains the colab demo, code, and pretrained checkpoints for our open-vocabulary detection method, ViLD (**Vi**sion and **L**anguage **D**istillation).

Open-vocabulary object detection detects objects described by arbitrary text inputs. The fundamental challenge is the availability of training data. Existing object detection datasets only contain hundreds of categories, and it is costly to scale further. To overcome this challenge, we propose ViLD. Our method distills the knowledge from a pretrained open-vocabulary image classification model (teacher) into a two-stage detector (student). Specifically, we use the teacher model to encode category texts and image regions of object proposals. Then we train a student detector, whose region embeddings of detected boxes are aligned with the text and image embeddings inferred by the teacher. 

We benchmark on LVIS by holding out all rare categories as novel categories not seen during training. ViLD obtains 16.1 mask APr, even outperforming the supervised counterpart by 3.8 with a ResNet-50 backbone. The model can directly transfer to other datasets without finetuning, achieving 72.2 AP50, 36.6 AP and 11.8 AP on PASCAL VOC, COCO and Objects365, respectively. On COCO, ViLD outperforms previous SOTA by 4.8 on novel AP and 11.4 on overall AP.

The figure below shows an overview of ViLD's architecture.
![architecture overview](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/vild/assets/new_overview_new_font.png)


# Colab Demo
In this [colab](https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/models/official/detection/projects/vild/ViLD_demo.ipynb) or this [jupyter notebook](./ViLD_demo.ipynb), we created a demo with two examples. You can also try your own images and specify the categories you want to detect. 


# Getting Started
## Prerequisite
* Install [TensorFlow](https://www.tensorflow.org/install).
* Install the packages in [`requirements.txt`](./requirements.txt).


## Data preprocessing
1. Download and unzip the [LVIS v1.0](https://www.lvisdataset.org/dataset) validation sets to `DATA_DIR`.

The `DATA_DIR` should be organized as below:

```
DATA_DIR
+-- lvis_v1_val.json
+-- val2017
|   +-- ***.jpg
|   +-- ...
```

2. Create tfrecords for the validation set (adjust `max_num_processes` if needed; specify `DEST_DIR` to the tfrecords output directory):

```shell
DATA_DIR=[DATA_DIR]
DEST_DIR=[DEST_DIR]
VAL_JSON="${DATA_DIR}/lvis_v1_val.json"
python3 preprocessing/create_lvis_tf_record.py \
  --image_dir="${DATA_DIR}" \
  --json_path="${VAL_JSON}" \
  --dest_dir="${DEST_DIR}" \
  --include_mask=True \
  --split='val' \
  --num_parts=100 \
  --max_num_processes=100
```

## Trained checkpoints
| Method        | Backbone     | Distillation weight | APr   |  APc |  APf | AP   | config | ckpt |
|:------------- |:-------------| -------------------:| -----:|-----:|-----:|-----:|--------|------|
| ViLD          | ResNet-50    | 0.5                 | 16.6  | 19.8 | 28.2 | 22.5 | [vild_resnet.yaml](./configs/vild_resnet.yaml) |[ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/vild/pretrained_ckpts/resnet50_vild.tar.gz)|
| ViLD-ensemble | ResNet-50    | 0.5                 |  18	 | 24.7	| 30.6 | 25.9 | [vild_resnet.yaml](./configs/vild_resnet.yaml) |[ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/vild/pretrained_ckpts/resnet50_vild_ensemble.tar.gz)|
| ViLD          | ResNet-152   | 1.0                 | 19.6	 | 21.6	| 28.5 | 24.0 | [vild_ensemble_resnet.yaml](./configs/vild_ensemble_resnet.yaml) |[ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/vild/pretrained_ckpts/resnet152_vild.tar.gz)|
| ViLD-ensemble | ResNet-152   | 2.0                 | 19.2	 | 24.8	| 30.8 | 26.2 | [vild_ensemble_resnet.yaml](./configs/vild_ensemble_resnet.yaml) |[ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/vild/pretrained_ckpts/resnet152_vild_ensemble.tar.gz)|

We provide the steps here in order to be able to use the provided checkpoints for saved model inferencing:

1. Create a conda environment for running ViLD:

```shell
ENV=vild
PYTHON=3.9
conda create -n $ENV python=$PYTHON
conda activate $ENV
conda install -c conda-forge cudatoolkit==11.2.2 cudatoolkit-dev==11.2.2 cudnn==8.2.1.32 pytorch-gpu==1.11.0 torchvision==0.12.0 'libtiff<4.5' 'pillow!=8.3.*'
conda install -c conda-forge tensorflow-gpu==2.8.1
conda install -c conda-forge easydict libstdcxx-ng ftfy regex tqdm pyyaml matplotlib
conda install pybind11
pip install opencv-python
pip install git+https://github.com/openai/CLIP.git
```

2. Clone the repository:

```shell
ROOTDIR=/path/to/clone/tpu/repo/into/as/subdirectory
cd "$ROOTDIR" && git clone https://github.com/pallgeuer/tpu.git
```

3. Download and extract the model checkpoints:

```shell
VILDDIR="$ROOTDIR/tpu/models/official/detection/projects/vild"
DATADIR="$VILDDIR/data"
CHECKPOINTSDIR="$DATADIR/checkpoints"
for model in resnet50_vild resnet50_vild_ensemble resnet152_vild resnet152_vild_ensemble; do wget -P "$CHECKPOINTSDIR" "https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/vild/pretrained_ckpts/$model.tar.gz" && tar -xf "$CHECKPOINTSDIR/$model.tar.gz" -C "$CHECKPOINTSDIR" && rm "$CHECKPOINTSDIR/$model.tar.gz"; done
```

4. Fix incorrectly-named model parameters in the ResNet152 checkpoints:

```shell
conda activate $ENV
for model in resnet152_vild resnet152_vild_ensemble; do echo; echo "FIXING BATCH NORM NAMING: $model"; mv "$CHECKPOINTSDIR/$model" "$CHECKPOINTSDIR/${model}_raw" && mkdir "$CHECKPOINTSDIR/$model" && python -c "import re; import tensorflow.compat.v1 as tf; tf.disable_eager_execution(); tf.disable_v2_behavior(); reader = tf.train.NewCheckpointReader('$CHECKPOINTSDIR/${model}_raw/model.ckpt-180000'); variables = {re.sub('_sim_clr', '', name): tf.Variable(reader.get_tensor(name)) for name in reader.get_variable_to_shape_map()}; init = tf.global_variables_initializer(); saver = tf.train.Saver(variables); sess = tf.Session().__enter__(); sess.run(init); saver.save(sess, '$CHECKPOINTSDIR/${model}/model.ckpt-180000'); sess.__exit__(None, None, None)" && rm -r "$CHECKPOINTSDIR/${model}_raw"; done
for modeldir in "$DATADIR"/checkpoints/*/; do model="${modeldir}model.ckpt-180000"; echo; echo "GETTING VARIABLES: $model"; python -c "import tensorflow.compat.v1 as tf; import pprint; reader = tf.train.NewCheckpointReader('$model'); pprint.pprint(reader.get_variable_to_shape_map(), width=120)" | tee "${modeldir}variables.txt"; done
```

5. Export the checkpoints as saved models (see `python "$VILDDIR/export/export_saved_model.py" --help` for all model customisation options):

```shell
SAVEDMODELSDIR="$DATADIR/saved_models"
NUM_ROIS=1000
NUM_DETS=300
mkdir "$SAVEDMODELSDIR"
conda activate $ENV
for resnet in 50 152; do for ensemble in '' _ensemble; do PYTHONPATH="$ROOTDIR/tpu/models/official/detection:$ROOTDIR/tpu/models" python "$VILDDIR/export/export_saved_model.py" --config_file="$VILDDIR/configs/vild${ensemble}_resnet.yaml" --resnet_depth="$resnet" --num_rois="$NUM_ROIS" --num_dets="$NUM_DETS" --checkpoint_path="$CHECKPOINTSDIR/resnet${resnet}_vild${ensemble}/model.ckpt-180000" --classifier_weights="$DATADIR/clip_synonym_prompt.npy" --export_dir="$SAVEDMODELSDIR/resnet${resnet}_vild${ensemble}_${NUM_ROIS}_${NUM_DETS}" --overwrite_export_dir --image_size=640 --nooutput_backbone_input --nooutput_fpn_features --nooutput_roi_boxes --nooutput_roi_features --output_objects --nooutput_boxes --nooutput_clip_features --nooutput_class_probs --nooutput_classes --nooutput_masks --noapply_nms; done; done
```

6. Test inference of the saved models (see `python "$VILDDIR/export/inference_saved_model.py" --help` for all inference options):

```shell
for saved_model in "$SAVEDMODELSDIR"/*; do if [[ -d "$saved_model" ]]; then PYTHONPATH="$ROOTDIR/tpu/models/official/detection" python "$VILDDIR/export/inference_saved_model.py" --saved_model_dir="$saved_model" --class_map="$DATADIR/lvis_class_map.txt" --bench=5 --nocpu --image="$DATADIR/images/000000013923.jpg" --image="$DATADIR/images/000000007088.jpg" --image="$DATADIR/images/000000007574.jpg"; fi; done
```

## Inference
1. Download the [classification weights](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/vild/weights/clip_synonym_prompt.npy) (CLIP text embeddings) and the [binary masks](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/vild/weights/lvis_rare_masks.npy) for rare categories. And put them in `[WEIGHTS_DIR]`.
2. Download and unzip the trained model you want to run inference in `[MODEL_DIR]`.
3. Replace `[RESNET_DEPTH], [MODEL_DIR], [DATA_DIR], [DEST_DIR], [WEIGHTS_DIR], [CONFIG_FILE]` with your values in the script below and run it.

Please refer [getting_started.md](https://github.com/tensorflow/tpu/blob/master/models/official/detection/GETTING_STARTED.md) for more information.

```shell
BATCH_SIZE=1
RESNET_DEPTH=[RESNET_DEPTH]
MODEL_DIR=[MODEL_DIR]
EVAL_FILE_PATTERN="[DEST_DIR]/val*"
VAL_JSON_FILE="[DATA_DIR]/lvis_v1_val.json"
RARE_MASK_PATH="[WEIGHTS_DIR]/lvis_rare_masks.npy"
CLASSIFIER_WEIGHT_PATH="[WEIGHTS_DIR]/clip_synonym_prompt.npy"
CONFIG_FILE="tpu/models/official/detection/projects/vild/configs/[CONFIG_FILE]"
python3 tpu/models/official/detection/main.py \
  --model="vild" \
  --model_dir="${MODEL_DIR?}" \
  --mode=eval \
  --use_tpu=False \
  --config_file="${CONFIG_FILE?}" \
  --params_override="{ resnet: {resnet_depth: ${RESNET_DEPTH?}}, predict: {predict_batch_size: ${BATCH_SIZE?}}, eval: {eval_batch_size: ${BATCH_SIZE?}, val_json_file: ${VAL_JSON_FILE?}, eval_file_pattern: ${EVAL_FILE_PATTERN?} }, frcnn_head: {classifier_weight_path: ${CLASSIFIER_WEIGHT_PATH?}}, postprocess: {rare_mask_path: ${RARE_MASK_PATH?}}}"
```


# License
This repo is under the same license as  [tensorflow/tpu](https://github.com/tensorflow/tpu), see
[license](https://github.com/tensorflow/tpu/blob/master/LICENSE).

# Citation
If you find this repo to be useful to your research, please cite our paper:

```
@article{gu2021open,
  title={Open-Vocabulary Detection via Vision and Language Knowledge Distillation},
  author={Gu, Xiuye and Lin, Tsung-Yi and Kuo, Weicheng and Cui, Yin},
  journal={arXiv preprint arXiv:2104.13921},
  year={2021}
}
```

# Acknowledgements
In this repo, we use [OpenAI's CLIP model](https://github.com/openai/CLIP) as the open-vocabulary image classification model, i.e., the teacher model.

The code is built upon [Cloud TPU detection](https://github.com/tensorflow/tpu/tree/master/models/official/detection).
