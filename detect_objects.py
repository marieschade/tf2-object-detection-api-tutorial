import os
import cv2
import time
import argparse

from detector import DetectorTF2


def DetectFromVideo(detector, Video_path, save_output=False, output_dir='output/', show_output=False):

	cap = cv2.VideoCapture(Video_path)
	output_path = os.path.join(output_dir, 'detection_'+ Video_path.split("/")[-1])
	frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (frame_width, frame_height))

	while (cap.isOpened()):
		ret, img = cap.read()
		if not ret: break

		timestamp1 = time.time()
		det_boxes = detector.DetectFromImage(img)
		elapsed_time = round((time.time() - timestamp1) * 1000) #ms
		img = detector.DisplayDetections(img, det_boxes, det_time=elapsed_time)
		
		if show_output:
			cv2.imshow('TF2 Detection', img)
			if cv2.waitKey(1) == 27: break

		out.write(img)

	cap.release()
	out.release()


def DetectImagesFromFolder(detector, images_dir, output_dir='output/', show_output=False, save_txt=False):
	
	for file in os.scandir(images_dir):
		if file.is_file() and file.name.endswith(('.jpg', '.jpeg', '.png')) :
			image_path = os.path.join(images_dir, file.name)
			print(image_path)
			img = cv2.imread(image_path)
			det_boxes = detector.DetectFromImage(img)
			
			if save_txt:
				if file.name.endswith(('.jpg')) :
					file_name = file.name.split(".jpg",1)[0]
				if file.name.endswith(('.jpeg')) :
					file_name = file.name.split(".jpeg",1)[0]
				if file.name.endswith(('.png')) :
					file_name = file.name.split(".png",1)[0]
				txt_path = os.path.join(output_dir, 'predicted_labels', file_name) # set output path for .txt file
				for idx in range(len(det_boxes)):
					x_min = str(det_boxes[idx][0])
					y_min = str(det_boxes[idx][1])
					x_max = str(det_boxes[idx][2])
					y_max = str(det_boxes[idx][3])
					cls = str(det_boxes[idx][4])
					score = str(det_boxes[idx][-1])
					
					with open(txt_path + '.txt', 'a') as f:
						f.write(cls + ' ' + score + ' ' + x_min + ' ' + y_min + ' ' + x_max + ' ' + y_max + '\n')
				
				
			
			img = detector.DisplayDetections(img, det_boxes)
			
			if show_output:
				cv2.imshow('TF2 Detection', img)
				cv2.waitKey(0)

			img_out = os.path.join(output_dir, file.name)
			cv2.imwrite(img_out, img)


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Object Detection from Images or Video')
	parser.add_argument('--model_path', help='Path to frozen detection model',
						default='models/efficientdet_d0_coco17_tpu-32/saved_model')
	parser.add_argument('--path_to_labelmap', help='Path to labelmap (.pbtxt) file',
	                    default='models/mscoco_label_map.pbtxt')
	parser.add_argument('--class_ids', help='id of classes to detect, expects string with ids delimited by ","',
	                    type=str, default=None) # example input "1,3" to detect person and car
	parser.add_argument('--threshold', help='Detection Threshold', type=float, default=0.4)
	parser.add_argument('--images_dir', help='Directory to input images)', default='data/samples/images/')
	parser.add_argument('--video_path', help='Path to input video)', default='data/samples/pedestrian_test.mp4')
	parser.add_argument('--output_directory', help='Path to output images and video', default='data/samples/output')
	parser.add_argument('--video_input', help='Flag for video input, default: False', action='store_true')  # default is false
	#parser.add_argument('--save_output', help='Flag for save images and video with detections visualized, default: False',
	#                    action='store_true')  # default is false
	parser.add_argument('--show_output', help='Flag for showing images right after detection, default: False',
	                    action='store_true')  # default is false
	parser.add_argument('--save_txt', help='Save bounding box .txt file for every image, default: False',
	                    action='store_true')  # default is false
	args = parser.parse_args()

	id_list = None
	if args.class_ids is not None:
		id_list = [int(item) for item in args.class_ids.split(',')]

	if not os.path.exists(args.output_directory):
		os.makedirs(args.output_directory) # create output directory
				
	if args.save_txt:
		os.chdir(args.output_directory)
		if not os.path.exists('predicted_labels'):
			os.makedirs('predicted_labels') # create predicted_labels directory

	# instance of the class DetectorTF2
	detector = DetectorTF2(args.model_path, args.path_to_labelmap, class_id=id_list, threshold=args.threshold)
	
	t0 = time.time()

	if args.video_input:
		DetectFromVideo(detector, args.video_path, output_dir=args.output_directory, show_output=args.show_output)
	else:
		DetectImagesFromFolder(detector, args.images_dir, output_dir=args.output_directory, show_output=args.show_output, save_txt=args.save_txt)

	print('Done. (%.3fs)' % (time.time() - t0))
	cv2.destroyAllWindows()
