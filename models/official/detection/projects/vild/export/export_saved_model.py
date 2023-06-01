#!/usr/bin/env python3
# Export a ViLD checkpoint as a saved model
#
# The PYTHONPATH must include:
#  - tpu/models
#  - tpu/models/official/detection

# Imports
import io
import os
import shutil
import pprint
import functools
import contextlib
from absl import flags
import tensorflow.compat.v1 as tf  # noqa
from tensorflow.python.ops import control_flow_util as tensorflow_control_flow_util
import tensorflow.python.tools.saved_model_cli as saved_model_cli
import hyperparameters.params_dict
import serving.detection
import serving.inputs
import dataloader.mode_keys
import projects.vild.configs.vild_config as vild_config
import projects.vild.modeling.vild_model as vild_model

# Command line flags
FLAGS = flags.FLAGS
flags.DEFINE_string('config_file', '', "JSON/YAML configuration file to use")
flags.DEFINE_string('params_override', '', "String or JSON/YAML specifying configuration parameters to overlay on top of the loaded configuration file")
flags.DEFINE_integer('resnet_depth', 50, "ResNet backbone depth")
flags.DEFINE_integer('num_rois', 1000, "Number of regions of interest to select")
flags.DEFINE_integer('num_dets', 300, "Number of object detections to select")
flags.DEFINE_boolean('apply_nms', False, "Apply non-maximum suppression (NMS) to the final detections as part of postprocessing")
flags.DEFINE_string('checkpoint', None, "Input checkpoint specification (appending .index or .meta to this path should yield existing file paths)")
flags.DEFINE_string('classifier_weights', None, "Weights to initialise the CLIP classification dense layer with")
flags.DEFINE_string('export_dir', None, "Export directory to write saved model files into (created if doesn't exist)")
flags.DEFINE_boolean('overwrite_export_dir', False, "Delete and recreate the export directory if it exists")
flags.DEFINE_integer('image_size', 640, "Model input image dimensions (should be divisible by 2^params.architecture.max_level)")
flags.DEFINE_boolean('output_backbone_input', False, "Include model output: Backbone input tensor")
flags.DEFINE_boolean('output_fpn_features', False, "Include model output: FPN features per level")
flags.DEFINE_boolean('output_roi_boxes', False, "Include model output: RoI boxes and scores")
flags.DEFINE_boolean('output_roi_features', False, "Include model output: RoI features")
flags.DEFINE_boolean('output_objects', False, "Include model output: Pre-classification object bounding boxes and CLIP features")
flags.DEFINE_boolean('output_boxes', False, "Include model output: Detected object bounding boxes")
flags.DEFINE_boolean('output_clip_features', False, "Include model output: CLIP features per detected object")
flags.DEFINE_boolean('output_class_probs', False, "Include model output: Object class probabilities")
flags.DEFINE_boolean('output_classes', False, "Include model output: Predicted object classes and scores")
flags.DEFINE_boolean('output_masks', False, "Include model output: Object segmentation masks")
flags.mark_flag_as_required('checkpoint')
flags.mark_flag_as_required('classifier_weights')
flags.mark_flag_as_required('export_dir')

# Main function
# noinspection PyUnusedLocal
def main(argv):
	export(
		config_file=FLAGS.config_file,
		params_override=FLAGS.params_override,
		resnet_depth=FLAGS.resnet_depth,
		num_rois=FLAGS.num_rois,
		num_dets=FLAGS.num_dets,
		apply_nms=FLAGS.apply_nms,
		checkpoint=FLAGS.checkpoint,
		classifier_weights=FLAGS.classifier_weights,
		export_dir=FLAGS.export_dir,
		overwrite_export_dir=FLAGS.overwrite_export_dir,
		image_size=(FLAGS.image_size, FLAGS.image_size),
		output_backbone_input=FLAGS.output_backbone_input,
		output_fpn_features=FLAGS.output_fpn_features,
		output_roi_boxes=FLAGS.output_roi_boxes,
		output_roi_features=FLAGS.output_roi_features,
		output_objects=FLAGS.output_objects,
		output_boxes=FLAGS.output_boxes,
		output_clip_features=FLAGS.output_clip_features,
		output_class_probs=FLAGS.output_class_probs,
		output_classes=FLAGS.output_classes,
		output_masks=FLAGS.output_masks,
	)

# Export checkpoint as saved model
def export(
	config_file,
	params_override,
	resnet_depth,
	num_rois,
	num_dets,
	apply_nms,
	checkpoint,
	classifier_weights,
	export_dir,
	overwrite_export_dir,
	image_size,
	output_backbone_input,
	output_fpn_features,
	output_roi_boxes,
	output_roi_features,
	output_objects,
	output_boxes,
	output_clip_features,
	output_class_probs,
	output_classes,
	output_masks,
):

	export_dir_exists = os.path.exists(export_dir)
	print("Exporting ViLD checkpoint as saved model:")
	print(f"  Config file: {config_file}")
	print(f"  Params override: {params_override if params_override else '<none>'}")
	print(f"  ResNet depth: {resnet_depth}")
	print(f"  Num RoIs: {num_rois}")
	print(f"  Num detections: {num_dets}")
	print(f"  Apply NMS: {apply_nms}")
	print(f"  Input checkpoint: {checkpoint}.*")
	print(f"  Classifier weights: {classifier_weights}")
	print(f"  Export dir: {export_dir}{' [OVERWRITE]' if overwrite_export_dir and export_dir_exists else ''}")
	print(f"  Image size: {image_size[1]}x{image_size[0]}")
	print(f"  Output backbone input: {output_backbone_input}")
	print(f"  Output FPN features: {output_fpn_features}")
	print(f"  Output RoI boxes: {output_roi_boxes}")
	print(f"  Output RoI features: {output_roi_features}")
	print(f"  Output objects: {output_objects}")
	print(f"  Output boxes: {output_boxes}")
	print(f"  Output CLIP features: {output_clip_features}")
	print(f"  Output class probs: {output_class_probs}")
	print(f"  Output classes: {output_classes}")
	print(f"  Output masks: {output_masks}")
	print()

	tensorflow_control_flow_util.enable_control_flow_v2()

	if export_dir_exists and not overwrite_export_dir:
		print("Export directory exists => Use --overwrite_export_dir if you wish to overwrite it")
		return
	export_dir_base = os.path.dirname(export_dir)
	if not os.path.isdir(export_dir_base):
		raise OSError(f"Parent directory of export directory must exist (and be a directory): {export_dir_base}")

	params = hyperparameters.params_dict.ParamsDict(vild_config.VILD_CFG, vild_config.VILD_RESTRICTIONS)
	if config_file:
		params = hyperparameters.params_dict.override_params_dict(params, config_file, is_strict=True)
	if params_override:
		params = hyperparameters.params_dict.override_params_dict(params, params_override, is_strict=False)

	params.override({
		'architecture': {
			'use_bfloat16': False,
			'include_mask': output_masks,
		},
		'eval': {'eval_batch_size': 1},
		'frcnn_class_loss': {'mask_rare': False},
		'frcnn_head': {'classifier_weight_path': classifier_weights},
		'postprocess': {
			'apply_nms': apply_nms,
			'mask_rare': False,
			'max_total_size': num_dets,
			'pre_nms_num_boxes': num_rois,
		},
		'predict': {'predict_batch_size': 1},
		'resnet': {'resnet_depth': resnet_depth},
		'roi_proposal': {
			'test_rpn_post_nms_top_k': num_rois,
			'test_rpn_pre_nms_top_k': num_rois,
		},
		'train': {'train_batch_size': 1},
	}, is_strict=True)
	params.validate()
	params.lock()

	model_params = dict(params.as_dict(), use_tpu=False, mode=tf.estimator.ModeKeys.PREDICT, transpose_input=False)
	print("Model parameters:")
	pprint.pprint(model_params, width=120)
	print()

	print("Creating serving input receiver function...")
	serving_input_receiver_fn = functools.partial(
		serving_input_fn,
		image_size=image_size,                        # ImageTensor: 1xHxWx3 uint8 representing RGB image tensor with values in 0-255
		max_level=params.architecture.max_level,
	)
	print("Creating serving model function...")
	serving_model_outputs_fn = functools.partial(
		serving_model_fn,                             # The R x RPN RoIs / N x detections are guaranteed to be sorted in descending RPNRoIScores / DetScores order respectively (regressed object sorting matches RPN RoIs)
		output_backbone_input=output_backbone_input,  # BackboneInput: 1xIxIx3 float representing normalised (approx zero mean and unit std dev) image tensor that is passed into the model backbone
		output_fpn_features=output_fpn_features,      # FPNFeaturesLevel{L}: 1xSxSxF float representing the FPN features for level L of the model backbone
		output_roi_boxes=output_roi_boxes,            # RPNRoIBoxes/RPNRoIScores: 1xRx4 float representing the regions of interest in the image (Format: {ymin, xmin, ymax, xmax} per rectangular region) / 1xR float of objectness probability scores
		output_roi_features=output_roi_features,      # RPNRoIFeatures: 1xRxTxTxF float representing the FPN features for each region of interest (interpolated from the most appropriate level)
		output_objects=output_objects,                # ObjBoxes/ObjCLIPFeatures: 1xRx4 float representing regressed object bounding boxes (Format: {ymin, xmin, ymax, xmax} per rectangular region) / 1xRxC float representing the projected CLIP-comparable features for each object
		output_boxes=output_boxes,                    # DetBoxes: 1xNx4 float representing regressed detected object bounding boxes (Format: {ymin, xmin, ymax, xmax} per rectangular region)
		output_clip_features=output_clip_features,    # DetCLIPFeatures: 1xNxC float representing the projected CLIP-comparable features for each detection
		output_class_probs=output_class_probs,        # DetClassProbs: 1xNxK float representing the class probabilities (NOT including background) for each detection
		output_classes=output_classes,                # DetClasses/DetScores: 1xN integer representing the predicted (1-indexed) class for each detection (may be background) / 1xN float representing the predicted class scores (probabilities) for each detection (guaranteed to be in decreasing order)
		output_masks=output_masks,                    # DetMasks: 1xNxMxM float representing the object segmentation mask (as probabilities) for each detection
	)
	print("Done")
	print()

	print("Creating TPU estimator...")
	estimator = tf.estimator.tpu.TPUEstimator(
		model_fn=serving_model_outputs_fn,
		model_dir=None,
		config=tf.estimator.tpu.RunConfig(
			tpu_config=tf.estimator.tpu.TPUConfig(iterations_per_loop=1),
			evaluation_master='local',
			master='local',
		),
		params=model_params,
		use_tpu=False,
		train_batch_size=1,
		eval_batch_size=1,
		predict_batch_size=1,
		batch_axis=None,
		eval_on_tpu=False,
		export_to_tpu=False,
		export_to_cpu=True,
	)
	print("Done")
	print()

	print("Exporting checkpoint as saved model...")
	export_path = estimator.export_saved_model(
		export_dir_base=export_dir_base,
		serving_input_receiver_fn=serving_input_receiver_fn,
		checkpoint_path=checkpoint,
	)
	print(f"Temporarily exported saved model as: {export_path.decode('utf-8')}")
	if os.path.exists(export_dir):
		print(f"Deleting existing directory: {export_dir}")
		shutil.rmtree(export_dir, ignore_errors=True)
	print(f"Renaming exported saved model directory to: {export_dir}")
	os.rename(export_path, export_dir)
	print("Done")
	print()

	print("Creating params.yaml with used config parameters...")
	hyperparameters.params_dict.save_params_dict_to_yaml(
		params=hyperparameters.params_dict.ParamsDict(model_params),
		file_path=os.path.join(export_dir, 'params.yaml'),
	)
	print()

	print("Creating variables.txt with list of saved model variables...")
	reader = tf.train.NewCheckpointReader(os.path.join(export_dir, 'variables', 'variables'))
	variable_shape_map = reader.get_variable_to_shape_map()
	pprint.pprint(variable_shape_map, width=120)
	with open(os.path.join(export_dir, 'variables.txt'), 'w') as file:
		pprint.pprint(variable_shape_map, width=120, stream=file)
	print()

	print("Creating info.txt with saved model information...")
	with contextlib.redirect_stdout(io.StringIO()) as string_buffer:
		parser = saved_model_cli.create_parser()
		args = parser.parse_args(['show', '--dir', export_dir, '--all'])
		args.func(args)
	info = string_buffer.getvalue()
	print(info)
	with open(os.path.join(export_dir, 'info.txt'), 'w') as file:
		print(info, file=file)
	print()

# Serving input function
def serving_input_fn(image_size, max_level):
	image_placeholder = tf.placeholder(dtype=tf.uint8, shape=(1, None, None, 3), name='ImageTensor')
	image = tf.squeeze(image_placeholder, axis=0)
	image, image_info = serving.inputs.preprocess_image(image=image, desired_size=image_size, stride=2 ** max_level)
	images = tf.expand_dims(image, axis=0)
	images_info = tf.expand_dims(image_info, axis=0)
	return tf.estimator.export.ServingInputReceiver(
		features={'images': images, 'image_info': images_info},
		receiver_tensors={'image_tensor': image_placeholder},
	)

# Serving model function
# noinspection PyUnusedLocal
def serving_model_fn(
	features, labels, mode, params,
	output_backbone_input,
	output_fpn_features,
	output_roi_boxes,
	output_roi_features,
	output_objects,
	output_boxes,
	output_clip_features,
	output_class_probs,
	output_classes,
	output_masks,
):

	if mode != tf.estimator.ModeKeys.PREDICT:
		raise ValueError(f"Mode must be PREDICT: {mode}")

	image_info = features['image_info']
	backbone_input = features['images']

	params = hyperparameters.params_dict.ParamsDict(params)
	model_outputs = vild_model.ViLDModel(params).build_outputs(backbone_input, labels={'image_info': image_info}, mode=dataloader.mode_keys.PREDICT)

	predictions = {'image_info': tf.identity(features['image_info'], 'ImageInfo')}

	if output_backbone_input:
		predictions['backbone_input'] = tf.identity(backbone_input, 'BackboneInput')
	if output_fpn_features:
		for level, fpn_features in model_outputs['fpn_features'].items():
			predictions[f'fpn_features_level{level}'] = tf.identity(fpn_features, f'FPNFeaturesLevel{level}')

	if output_roi_boxes:
		predictions.update(
			rpn_roi_boxes=tf.identity(model_outputs['rpn_rois'], 'RPNRoIBoxes'),
			rpn_roi_scores=tf.identity(model_outputs['rpn_roi_scores'], 'RPNRoIScores'),
		)
	if output_roi_features:
		predictions['rpn_roi_features'] = tf.identity(model_outputs['roi_features'], 'RPNRoIFeatures')

	if output_objects:
		predictions.update(
			obj_boxes=tf.identity(model_outputs['object_boxes'], 'ObjBoxes'),
			obj_clip_features=tf.identity(model_outputs['object_feat'], 'ObjCLIPFeatures'),
		)

	if output_boxes:
		predictions['det_boxes'] = tf.identity(model_outputs['detection_boxes'], 'DetBoxes')
	if output_clip_features:
		predictions['det_clip_features'] = tf.identity(model_outputs['detection_feat'], 'DetCLIPFeatures')
	if output_class_probs:
		predictions['det_class_probs'] = tf.identity(model_outputs['detection_probs'], 'DetClassProbs')
	if output_classes:
		predictions.update(
			det_classes=tf.identity(model_outputs['detection_classes'], 'DetClasses'),
			det_scores=tf.identity(model_outputs['detection_scores'], 'DetScores'),
		)
	if output_masks:
		predictions['det_masks'] = tf.identity(model_outputs['detection_masks'], 'DetMasks')
	if any(key.startswith('det_') for key in predictions.keys()):
		predictions['det_count'] = tf.identity(model_outputs['num_detections'], 'NumDetections')

	return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

# Run main function
if __name__ == '__main__':
	tf.app.run(main)
# EOF
