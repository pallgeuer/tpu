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
import projects.vild.configs.vild_config as vild_config

# Command line flags
FLAGS = flags.FLAGS
flags.DEFINE_string('config_file', '', "JSON/YAML configuration file to use")
flags.DEFINE_string('params_override', '', "String or JSON/YAML specifying configuration parameters to overlay on top of the loaded configuration file")
flags.DEFINE_integer('resnet_depth', 50, "ResNet backbone depth")
flags.DEFINE_string('checkpoint_path', None, "Input checkpoint specification (appending .index or .meta to this path should yield existing file paths)")
flags.DEFINE_string('classifier_weights', None, "Weights to use in the classification head to classify the base classes")
flags.DEFINE_string('export_dir', None, "Export directory to write saved model files into (created if doesn't exist)")
flags.DEFINE_boolean('overwrite_export_dir', False, "Delete and recreate the export directory if it exists")
flags.DEFINE_string('image_size', '640x640', "Expected input image width and height separated by 'x'")
flags.mark_flag_as_required('checkpoint_path')
flags.mark_flag_as_required('classifier_weights')
flags.mark_flag_as_required('export_dir')

# Main function
# noinspection PyUnusedLocal
def main(argv):
	image_width_str, image_height_str = FLAGS.image_size.split('x')
	export(
		config_file=FLAGS.config_file,
		params_override=FLAGS.params_override,
		resnet_depth=FLAGS.resnet_depth,
		checkpoint_path=FLAGS.checkpoint_path,
		classifier_weights=FLAGS.classifier_weights,
		export_dir=FLAGS.export_dir,
		overwrite_export_dir=FLAGS.overwrite_export_dir,
		image_size=(int(image_height_str), int(image_width_str)),
	)

# Export saved model function
def export(config_file, params_override, resnet_depth, checkpoint_path, classifier_weights, export_dir, overwrite_export_dir, image_size):

	export_dir_exists = os.path.exists(export_dir)
	print("Exporting ViLD checkpoint as saved model:")
	print(f"  Config file: {config_file}")
	print(f"  Params override: {params_override if params_override else '<none>'}")
	print(f"  ResNet depth: {resnet_depth}")
	print(f"  Input checkpoint: {checkpoint_path}.*")
	print(f"  Classifier weights: {classifier_weights}")
	print(f"  Export dir: {export_dir}{' [OVERWRITE]' if overwrite_export_dir and export_dir_exists else ''}")
	print(f"  Image size: {image_size[1]}x{image_size[0]}")
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
		'architecture': {'use_bfloat16': False},
		'eval': {'eval_batch_size': 1},
		'frcnn_class_loss': {'mask_rare': False},
		'frcnn_head': {'classifier_weight_path': classifier_weights},
		'postprocess': {'mask_rare': False},
		'predict': {'predict_batch_size': 1},
		'resnet': {'resnet_depth': resnet_depth},
		'train': {'train_batch_size': 1},
	}, is_strict=True)
	params.validate()
	params.lock()

	model_params = dict(params.as_dict(), use_tpu=False, mode=tf.estimator.ModeKeys.PREDICT, transpose_input=False)
	print("Model parameters:")
	pprint.pprint(model_params, width=120)
	print()

	print("Creating serving model function...")
	serving_model_fn = serving.detection.serving_model_fn_builder(  # TODO: Need to change something here to customise which outputs are served
		export_tpu_model=False,
		output_image_info=True,
		output_normalized_coordinates=False,
		cast_num_detections_to_float=False,
		cast_detection_classes_to_float=False,
	)
	print("Creating serving input receiver function...")
	serving_input_receiver_fn = functools.partial(
		serving.detection.serving_input_fn,
		batch_size=1,
		desired_image_size=image_size,
		stride=2 ** params.architecture.max_level,
		input_type='image_tensor',
		input_name='image_tensor'
	)
	print("Done")
	print()

	print("Creating TPU estimator...")
	estimator = tf.estimator.tpu.TPUEstimator(
		model_fn=serving_model_fn,
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
		checkpoint_path=checkpoint_path,
	)
	print(f"Temporarily exported saved model as: {export_path.decode('utf-8')}")
	if os.path.exists(export_dir):
		print(f"Deleting existing directory: {export_dir}")
		shutil.rmtree(export_dir, ignore_errors=True)
	print(f"Renaming exported saved model directory to: {export_dir}")
	os.rename(export_path, export_dir)
	print("Done")
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

	print("Creating variables.txt with list of saved model variables...")
	reader = tf.train.NewCheckpointReader(os.path.join(export_dir, 'variables', 'variables'))
	variable_shape_map = reader.get_variable_to_shape_map()
	pprint.pprint(variable_shape_map, width=120)
	with open(os.path.join(export_dir, 'variables.txt'), 'w') as file:
		pprint.pprint(variable_shape_map, width=120, stream=file)
	print()

# Run main function
if __name__ == '__main__':
	tf.app.run(main)
# EOF
