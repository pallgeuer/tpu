#!/usr/bin/env python3
# Inference a ViLD saved model

# Imports
import csv
import time
from absl import flags
import numpy as np
import cv2
import tensorflow.compat.v1 as tf  # noqa
import utils.box_utils
import utils.mask_utils
import utils.input_utils
import utils.object_detection.visualization_utils

# Command line flags
FLAGS = flags.FLAGS
flags.DEFINE_string('saved_model_dir', None, "Saved model directory to inference")
flags.DEFINE_string('class_map', None, "LVIS class ID to name mapping file (each line contains e.g. '19:armchair', IDs are 1-indexed, should be 1203 of them)")
flags.DEFINE_multi_string('image', None, "Input image file path(s)")
flags.mark_flag_as_required('saved_model_dir')
flags.mark_flag_as_required('image')

# Main function
# noinspection PyUnusedLocal
def main(argv):
	inference(
		saved_model_dir=FLAGS.saved_model_dir,
		class_map_path=FLAGS.class_map,
		image_paths=FLAGS.image,
	)

# Inference saved model on image path
def inference(saved_model_dir, class_map_path, image_paths):

	# TODO: Disable V2 features like for exporting the model (I ACTIVATE stuff there, not deactivate?)? Does that help in any way?

	print("Inferencing saved model on image:")
	print(f"  Saved model: {saved_model_dir}")
	for image_path in image_paths:
		print(f"  Image: {image_path}")
	print()

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
	session_config = tf.ConfigProto(log_device_placement=False)  # TODO: TEMP
	session_config.gpu_options.allow_growth = True
	session_config.graph_options.optimizer_options.opt_level = -1  # TODO: Does this disable grappler? Doesn't seem so
	session_config.graph_options.rewrite_options.disable_meta_optimizer = True  # TODO: Does this disable grappler? --> 30s startup time instead of ~4-5min and 2.67s inference instead of 2.89s after Grappler
	with tf.Session(graph=tf.Graph(), config=session_config) as session:  # TODO: , session.graph.device('/cpu:0'):
		print("Done")
		print()

		print("Loading saved model...")
		meta_graph_def = tf.saved_model.load(session, [tf.saved_model.tag_constants.SERVING], saved_model_dir)
		session.graph.finalize()
		serving_def = meta_graph_def.signature_def['serving_default']
		print(serving_def.inputs, type(serving_def.inputs))
		sig_inputs = dict(serving_def.inputs)
		sig_outputs = dict(serving_def.outputs)
		input_node = sig_inputs['image_tensor']
		output_nodes = {
			'image_info': sig_outputs['image_info'].name,
			'num_detections': sig_outputs['num_detections'].name,
			'detection_boxes': sig_outputs['detection_boxes'].name,
			'detection_classes': sig_outputs['detection_classes'].name,
			'detection_scores': sig_outputs['detection_scores'].name,
		}
		if include_mask := 'detection_masks' in sig_outputs:
			output_nodes['detection_masks'] = sig_outputs['detection_masks'].name
		print("Done")
		print()

		for image_path in image_paths:

			print(f"Inference saved model on image: {image_path}")

			raw_image_bgr = cv2.imread(image_path, flags=cv2.IMREAD_COLOR).astype(np.uint8)
			raw_image_rgb = cv2.cvtColor(raw_image_bgr, cv2.COLOR_BGR2RGB)
			print(raw_image_rgb.shape, raw_image_rgb.dtype)

			# TODO: Avoid performing the graph optimisation every time?? (https://stackoverflow.com/questions/74219568/optimize-and-resave-saved-model-with-grappler)
			# TODO: Optimise model evaluation time (remove masks, do not perform base class estimation at all, literally just return visual features?, Reduce number of object RoIs that have their visual features computed - just the 20 most objecty ones?)
			start_time = time.perf_counter()
			output_results = session.run(output_nodes, feed_dict={input_node.name: raw_image_rgb[None, ...]})
			stop_time = time.perf_counter()
			print(f"Elapsed: {stop_time - start_time:.3f}")
			import pprint
			pprint.pprint(output_results)

			num_detections = int(output_results['num_detections'][0])
			print(num_detections)
			np_boxes = output_results['detection_boxes'][0, :num_detections]
			print('D', np_boxes)
			np_image_info = output_results['image_info'][0]
			np_boxes = np_boxes / np.tile(np_image_info[1:2, :], (1, 2))  # TODO: Better to just divide by 2:3 (scale factor) (scaled image is top-left aligned with padded image)
			print('E', np_boxes)
			ymin, xmin, ymax, xmax = np.split(np_boxes, 4, axis=-1)
			ymin = ymin * 1280
			ymax = ymax * 1280
			xmin = xmin * 1280
			xmax = xmax * 1280
			np_boxes = np.concatenate([ymin, xmin, ymax, xmax], axis=-1)
			print('F', np_boxes)
			np_scores = output_results['detection_scores'][0, :num_detections]
			print(np_scores)
			np_classes = output_results['detection_classes'][0, :num_detections]
			np_classes = np_classes.astype(np.int32)
			print('H', np_classes)
			if include_mask:
				np_masks = output_results['detection_masks'][0, :num_detections]
				print(np_masks.shape)
				np_masks = utils.mask_utils.paste_instance_masks(np_masks, utils.box_utils.yxyx_to_xywh(np_boxes), 960, 1280)
				print('J', np_masks.shape)
			else:
				np_masks = None

			image_with_detections = (
				utils.object_detection.visualization_utils.visualize_boxes_and_labels_on_image_array(
					raw_image_bgr,
					np_boxes,
					np_classes,
					np_scores,
					class_map,
					instance_masks=np_masks,
					use_normalized_coordinates=False,
					max_boxes_to_draw=30,
					min_score_thresh=0.05))

			cv2.imshow('Detections', image_with_detections)
			cv2.waitKey(0)

			print("Done")
			print()

		cv2.destroyAllWindows()

# Run main function
if __name__ == '__main__':
	tf.app.run(main)
# EOF
