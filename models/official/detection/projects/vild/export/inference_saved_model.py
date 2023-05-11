#!/usr/bin/env python3
# Inference a ViLD saved model

# Imports
import re
import csv
import time
from absl import flags
from typing import Any
import numpy as np
import cv2
import tensorflow.compat.v1 as tf  # noqa
from tensorflow.core.framework.types_pb2 import DataType as TensorDataType  # noqa
from tensorflow.python.ops import control_flow_util as tensorflow_control_flow_util
import utils.box_utils
import utils.mask_utils
import utils.input_utils
import utils.object_detection.visualization_utils

# Constants
LINE_THICKNESS = 3

# Command line flags
FLAGS = flags.FLAGS
flags.DEFINE_string('saved_model_dir', None, "Saved model directory to inference")
flags.DEFINE_string('class_map', None, "LVIS class ID to name mapping file (each line contains e.g. '19:armchair', IDs are 1-indexed, should be 1203 of them)")
flags.DEFINE_multi_string('image', None, "Input image file path(s)")
flags.DEFINE_boolean('cpu', False, "Whether to run on CPU instead of GPU")
flags.DEFINE_integer('bench', 1, "Number of times to repeat inference for the purpose of more accurate timing benchmark")
flags.DEFINE_boolean('graph_opt', True, "Perform graph optimisation on loaded model")
flags.DEFINE_integer('max_boxes_roi', 200, "Maximum number of RoI boxes to draw")
flags.DEFINE_integer('max_boxes_obj', 200, "Maximum number of regressed object boxes to draw")
flags.DEFINE_integer('max_boxes_det', 100, "Maximum number of detection boxes to draw")
flags.DEFINE_float('min_score_roi', 0.9, "Minimum score in order to draw a RoI box")
flags.DEFINE_float('min_score_det', 0.2, "Minimum score in order to draw a detection box")
flags.mark_flag_as_required('saved_model_dir')
flags.mark_flag_as_required('image')

# Main function
# noinspection PyUnusedLocal
def main(argv):
	inference(
		saved_model_dir=FLAGS.saved_model_dir,
		class_map_path=FLAGS.class_map,
		image_paths=FLAGS.image,
		cpu=FLAGS.cpu,
		bench=FLAGS.bench,
		graph_opt=FLAGS.graph_opt,
		max_boxes_roi=FLAGS.max_boxes_roi,
		max_boxes_obj=FLAGS.max_boxes_obj,
		max_boxes_det=FLAGS.max_boxes_det,
		min_score_roi=FLAGS.min_score_roi,
		min_score_det=FLAGS.min_score_det,
	)

# Inference saved model on image path
def inference(
	saved_model_dir,
	class_map_path,
	image_paths,
	cpu,
	bench,
	graph_opt,
	max_boxes_roi,
	max_boxes_obj,
	max_boxes_det,
	min_score_roi,
	min_score_det,
):

	print("Initialising TensorFlow...")
	num_gpus = len(tf.config.list_physical_devices('GPU'))
	print("Done")
	print()

	print("Inferencing saved model on image:")
	print(f"  Saved model: {saved_model_dir}")
	print(f"  Class map: {class_map_path}")
	if not image_paths:
		raise ValueError("Must provide at least one image to inference on")
	elif len(image_paths) == 1:
		print(f"  Image: {image_paths[0]}")
	else:
		print("  Images:")
		for image_path in image_paths:
			print(f"    {image_path}")
	print(f"  Device: {'CPU' if cpu or num_gpus <= 0 else 'GPU'}")
	print(f"  Optimise graph: {graph_opt}")
	print(f"  Max RoI boxes: {max_boxes_roi}")
	print(f"  Max obj boxes: {max_boxes_obj}")
	print(f"  Max det boxes: {max_boxes_det}")
	print(f"  Min RoI score: {min_score_roi}")
	print(f"  Min det score: {min_score_det}")
	print()

	tensorflow_control_flow_util.enable_control_flow_v2()

	if class_map_path:
		print('Loading LVIS base class label map...')
		class_map = {}
		with open(class_map_path, 'r') as file:
			reader = csv.reader(file, delimiter=':')
			for row in reader:
				if len(row) != 2:
					raise ValueError(f"Each row of the class map file must have the format 'id:name': {row}")
				class_id = int(row[0])
				class_map[class_id] = {'id': class_id, 'name': row[1]}
		print("Done")
		print()
	else:
		class_map = {i: {'id': i, 'name': str(i)} for i in range(1, 1204)}

	print("Starting TensorFlow session...")

	session_config = tf.ConfigProto(device_count={'GPU': 0 if cpu else num_gpus})
	session_config.gpu_options.allow_growth = True
	if not graph_opt:
		session_config.graph_options.optimizer_options.opt_level = -1
		session_config.graph_options.rewrite_options.disable_meta_optimizer = True

	with tf.Session(graph=tf.Graph(), config=session_config) as session:

		print("Done")
		print()

		print("Loading saved model...")
		meta_graph_def = tf.saved_model.load(session, [tf.saved_model.tag_constants.SERVING], saved_model_dir)
		session.graph.finalize()
		print("Done")
		print()

		print("Inspecting loaded model...")
		serving_def = meta_graph_def.signature_def['serving_default']
		sig_inputs = dict(serving_def.inputs)
		sig_outputs = dict(serving_def.outputs)
		print_tensor_map('Input', sig_inputs)
		print_tensor_map('Output', sig_outputs)
		input_node = sig_inputs.get('image_tensor', None)
		if input_node is None:
			raise RuntimeError("A model with 'image_tensor' as an input is required for inference")
		output_nodes = {name: sig_outputs[name].name for name in sig_outputs}
		print("Done")
		print()

		for image_path in image_paths:

			print(f"Inferencing saved model on image: {image_path}")

			image_bgr = cv2.imread(image_path, flags=cv2.IMREAD_COLOR).astype(np.uint8)
			cv2.imshow('Input image', image_bgr)
			cv2.waitKey(1)

			bench_times = []
			for b in range(bench):
				start_time = time.perf_counter()
				output_results = session.run(output_nodes, feed_dict={input_node.name: image_bgr[None, :, :, ::-1]})
				elapsed = time.perf_counter() - start_time
				bench_times.append(elapsed)
				print(f"Inference runtime: {elapsed:.3f}")
			if bench >= 4:
				print(f"Benched inference runtime: {sum(bench_times[2:]) / (bench - 2):.3f}")
			print()

			if (image_info := output_results.get('image_info', None)) is not None:
				image_info = image_info[0]

				input_width = round(image_info[0, 1])
				input_height = round(image_info[0, 0])
				print(f"Target size: {round(image_info[1, 1])}x{round(image_info[1, 0])}")
				print(f"Input size:  {input_width}x{input_height}")
				print(f"Scaled size: {round(image_info[0, 1] * image_info[2, 1])}x{round(image_info[0, 0] * image_info[2, 0])}")
				print(f"Padded size: {round(image_info[4, 1])}x{round(image_info[4, 0])} (top-left aligned)")
				if image_info[3, 1] != 0 or image_info[3, 0] != 0:
					raise RuntimeError(f"Internal image translation is always assumed to be zero: ({image_info[3, 1]}, {image_info[3, 0]})")
				box_units = np.tile(image_info[2:3, :], reps=(1, 2))
				print()

				if (backbone_input := output_results.get('backbone_input', None)) is not None:
					backbone_input = backbone_input[0]
					print(f"Backbone input size: {shape_str(backbone_input)} (RGB)")
					print(f"Backbone input min: {backbone_input.min(axis=(0, 1))}")
					print(f"Backbone input mean: {backbone_input.mean(axis=(0, 1))}")
					print(f"Backbone input max: {backbone_input.max(axis=(0, 1))}")
					print(f"Backbone input zero: {np.mean(np.all(backbone_input == 0, axis=2)):.1%}")
					cv2.imshow('Backbone input', np.rint(255 * (backbone_input * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406)))).clip(0, 255).astype(np.uint8)[:, :, ::-1])
					print()

				have_fpn_features = False
				for key, fpn_features in sorted(output_results.items()):
					if match := re.fullmatch(r'fpn_features_level([0-9]+)', key):
						print(f"FPN features level {match.group(1)}: {shape_str(fpn_features[0])}")
						have_fpn_features = True
				if have_fpn_features:
					print()

				if (rpn_roi_scores := output_results.get('rpn_roi_scores', None)) is not None:
					rpn_roi_scores = rpn_roi_scores[0]
					if not np.all(np.diff(rpn_roi_scores) <= 0):
						raise RuntimeError("RPN RoI scores must be in decreasing order")
					print(f"RPN RoI scores: {shape_str(rpn_roi_scores)}")
					print(f"RPN RoI scores min: {rpn_roi_scores.min()}")
					print(f"RPN RoI scores max: {rpn_roi_scores.max()}")
					print()

				if (rpn_roi_boxes := output_results.get('rpn_roi_boxes', None)) is not None:
					rpn_roi_boxes = rpn_roi_boxes[0] / box_units
					print(f"RPN RoI boxes: {shape_str(rpn_roi_boxes)}")
					print(f"RPN RoI boxes format: [ymin xmin ymax xmax] relative to {input_width}x{input_height} input image")
					print(f"RPN RoI boxes min: {rpn_roi_boxes.min(axis=0)}")
					print(f"RPN RoI boxes max: {rpn_roi_boxes.max(axis=0)}")
					show_annotations(title='Regions of interest', image_bgr=image_bgr, boxes=rpn_roi_boxes, scores=rpn_roi_scores, max_boxes=max_boxes_roi, min_score=min_score_roi)
					print()

				if (rpn_roi_features := output_results.get('rpn_roi_features', None)) is not None:
					rpn_roi_features = rpn_roi_features[0]
					print(f"RPN RoI features: {shape_str(rpn_roi_features)}")
					print(f"RPN RoI features min: {rpn_roi_features.min()}")
					print(f"RPN RoI features mean: {rpn_roi_features.mean()}")
					print(f"RPN RoI features std: {rpn_roi_features.std()}")
					print(f"RPN RoI features max: {rpn_roi_features.max()}")
					print()

				if (obj_boxes := output_results.get('obj_boxes', None)) is not None:
					obj_boxes = obj_boxes[0] / box_units
					print(f"Obj boxes: {shape_str(obj_boxes)}")
					print(f"Obj boxes format: [ymin xmin ymax xmax] relative to {input_width}x{input_height} input image")
					print(f"Obj boxes min: {obj_boxes.min(axis=0)}")
					print(f"Obj boxes max: {obj_boxes.max(axis=0)}")
					show_annotations(title='Regressed objects', image_bgr=image_bgr, boxes=obj_boxes, scores=False, max_boxes=max_boxes_obj, min_score=0)
					print()

				if (obj_clip_features := output_results.get('obj_clip_features', None)) is not None:
					obj_clip_features = obj_clip_features[0]
					print(f"Obj CLIP features: {shape_str(obj_clip_features)}")
					obj_clip_features_norm: Any = np.linalg.norm(obj_clip_features, axis=1)
					print(f"Obj CLIP features norm min: {obj_clip_features_norm.min()}")
					print(f"Obj CLIP features norm mean: {obj_clip_features_norm.mean()}")
					print(f"Obj CLIP features norm max: {obj_clip_features_norm.max()}")
					print()

				if (det_count := output_results.get('det_count', None)) is not None:
					det_count = det_count[0]
					print(f"Num detections: {det_count}")
					print()

				if (det_clip_features := output_results.get('det_clip_features', None)) is not None:
					det_clip_features = det_clip_features[0]
					if det_count is not None:
						det_clip_features = det_clip_features[:det_count]
					print(f"Det CLIP features: {shape_str(det_clip_features)}")
					det_clip_features_norm: Any = np.linalg.norm(det_clip_features, axis=1)
					print(f"Det CLIP features norm min: {det_clip_features_norm.min()}")
					print(f"Det CLIP features norm mean: {det_clip_features_norm.mean()}")
					print(f"Det CLIP features norm max: {det_clip_features_norm.max()}")
					print()

				if (det_class_probs := output_results.get('det_class_probs', None)) is not None:
					det_class_probs = det_class_probs[0]
					if det_count is not None:
						det_class_probs = det_class_probs[:det_count]
					print(f"Det class probs: {shape_str(det_class_probs)}")
					print(f"Det class probs min: {det_class_probs.min()}")
					print(f"Det class probs max: {det_class_probs.max()}")
					det_class_max_prob = det_class_probs.max(axis=1)
					print(f"Det class max-prob min: {det_class_max_prob.min()}")
					print(f"Det class max-prob mean: {det_class_max_prob.mean()}")
					print(f"Det class max-prob max: {det_class_max_prob.max()}")
					print()

				if (det_classes := output_results.get('det_classes', None)) is not None:
					det_classes = det_classes[0]
					if det_count is not None:
						det_classes = det_classes[:det_count]
					print(f"Det classes: {shape_str(det_classes)}")
					print(f"Det classes min: {det_classes.min()}")
					print(f"Det classes max: {det_classes.max()}")
					print()

				if (det_scores := output_results.get('det_scores', None)) is not None:
					det_scores = det_scores[0]
					if not np.all(np.diff(det_scores) <= 0):
						raise RuntimeError("Det scores must be in decreasing order")
					if det_count is not None:
						det_scores = det_scores[:det_count]
					if det_class_probs is not None:
						if det_classes is not None and not np.array_equal(np.squeeze(np.take_along_axis(det_class_probs, det_classes[:, None] - 1, axis=1), axis=1), det_scores):
							raise RuntimeError("Det scores are inconsistent with combination of det class probs and det classes")
						print(f"Det scores that are not max-prob: {np.count_nonzero(det_scores != det_class_max_prob)}/{det_scores.shape[0]}")
					print(f"Det scores: {shape_str(det_scores)}")
					print(f"Det scores min: {det_scores.min()}")
					print(f"Det scores max: {det_scores.max()}")
					print()

				if (det_masks := output_results.get('det_masks', None)) is not None:
					det_masks = det_masks[0]
					if det_count is not None:
						det_masks = det_masks[:det_count]
					print(f"Det masks: {shape_str(det_masks)}")
					print(f"Det masks value min: {det_masks.min()}")
					print(f"Det masks value max: {det_masks.max()}")
					det_masks_fill = det_masks.mean(axis=(1, 2))
					print(f"Det masks fill min: {det_masks_fill.min():.1%}")
					print(f"Det masks fill max: {det_masks_fill.max():.1%}")
					print()

				if (det_boxes := output_results.get('det_boxes', None)) is not None:
					det_boxes = det_boxes[0]
					if det_count is not None:
						det_boxes = det_boxes[:det_count]
					det_boxes = det_boxes / box_units
					print(f"Det boxes: {shape_str(det_boxes)}")
					print(f"Det boxes format: [ymin xmin ymax xmax] relative to {input_width}x{input_height} input image")
					print(f"Det boxes min: {det_boxes.min(axis=0)}")
					print(f"Det boxes max: {det_boxes.max(axis=0)}")
					instance_masks = utils.mask_utils.paste_instance_masks(det_masks, utils.box_utils.yxyx_to_xywh(det_boxes), input_height, input_width) if det_masks is not None else None
					show_annotations(title='Object detections', image_bgr=image_bgr, boxes=det_boxes, scores=det_scores, classes=det_classes, class_map=class_map, instance_masks=instance_masks, max_boxes=max_boxes_det, min_score=min_score_det)
					print()

			cv2.waitKey(0)

		cv2.destroyAllWindows()

# Show an annotated image of bounding boxes
def show_annotations(title, image_bgr, boxes, scores, max_boxes, min_score, classes=None, class_map=None, instance_masks=None):
	cv2.imshow(title, utils.object_detection.visualization_utils.visualize_boxes_and_labels_on_image_array(
		image=image_bgr.copy(),
		boxes=boxes,
		classes=classes,
		agnostic_mode=classes is None,
		scores=None if scores is False else np.ones(boxes.shape[0], dtype=np.float32) if scores is None or scores is True else scores,
		category_index=class_map,
		instance_masks=instance_masks,
		use_normalized_coordinates=False,
		max_boxes_to_draw=max_boxes,
		min_score_thresh=min_score,
		line_thickness=LINE_THICKNESS,
		groundtruth_box_visualization_color='#008cff',
	))
	num_boxes = boxes.shape[0]
	if isinstance(scores, np.ndarray):
		scored_boxes = np.searchsorted(-scores, -min_score, side='left')
		if scored_boxes <= 0:
			print(f"Annotated 0 boxes (none are above score of {min_score:.1%})")
		elif scored_boxes > max_boxes:
			print(f"Annotated {max_boxes} (limit) boxes down to a score of {scores[max_boxes - 1]:.1%}")
		elif scored_boxes < num_boxes:
			print(f"Annotated {scored_boxes} boxes down to a score of {scores[scored_boxes - 1]:.1%} (limit)")
		else:
			print(f"Annotated {scored_boxes} boxes down to a score of {scores[scored_boxes - 1]:.1%}")
	elif num_boxes <= max_boxes:
		print(f"Annotated {num_boxes} boxes")
	else:
		print(f"Annotated {max_boxes} (limit) boxes")

# Shape to string
def shape_str(array):
	return 'x'.join(str(dim) for dim in array.shape)

# Print a tensor map
TF_TYPE_MAP = {value: key for (key, value) in TensorDataType.items()}
def print_tensor_map(name, tensor_map):
	max_key_len = max(len(key) for key in tensor_map.keys())
	max_name_len = max(len(tensor.name) for tensor in tensor_map.values())
	for key, tensor in sorted(tensor_map.items()):
		print(f"{name}[{key:{max_key_len}s}] = {tensor.name:{max_name_len}s} ==> {TF_TYPE_MAP[tensor.dtype]:8s} {'x'.join('?' if dim.size < 0 else str(dim.size) for dim in tensor.tensor_shape.dim)}")

# Run main function
if __name__ == '__main__':
	tf.app.run(main)
# EOF
