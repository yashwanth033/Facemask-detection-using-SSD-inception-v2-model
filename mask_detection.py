import cv2
import tensorflow as tf
from imutils.video import VideoStream
import numpy as np
import argparse
from datetime import *

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = 'frozen_inference_graph.pb'
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--display', dest='display', type=int,
                        default=1, help='Display the detected images using OpenCV. This reduces FPS')
args = vars(ap.parse_args())


def load_inference_graph():
    # load frozen tensorflow model into memory
    print("> ====== Loading frozen graph into memory")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    print(">  ====== Inference graph loaded.")
    return detection_graph, sess


def draw_text_on_image(fps, image_np):
    cv2.putText(image_np, fps, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)


def detect_objects(image_np, detection_graph, sess):
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name(
        'detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name(
        'detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name(
        'detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name(
        'num_detections:0')

    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores,
         detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    return np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes), num


def draw_box_on_image(num_masks, score_thresh, scores, boxes, classes, im_width, im_height, image_np):
    color = None
    color0 = (255, 0, 0)
    color1 = (0, 50, 255)
    for i in range(num_masks):
        if scores[i] > score_thresh:
            if classes[i] == 1:
                id = 'no mask'
            if classes[i] == 2:
                id = 'mask'
            if classes[i] == 3:
                id = 'improper mask'
            if i == 0:
                color = color0
            else:
                color = color1

            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)
            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))

            cv2.rectangle(image_np, p1, p2, color, 3, 1)

            cv2.putText(image_np, 'Mask ' + str(i) + ': ' + id, (int(left), int(top) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.putText(image_np, 'confidence: ' + str("{0:.2f}".format(scores[i])),
                        (int(left), int(top) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


vs = VideoStream(0).start()
cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture(0)
im_height, im_width = (None, None)
detection_graph, sess = load_inference_graph()
score_thresh = 0.80
start_time = datetime.now()
num_frames = 0

try:
    while True:
        frame = vs.read()
        frame = np.array(frame)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if im_height == None:
            im_height, im_width = frame.shape[:2]

        # Convert image to rgb since opencv loads images in bgr, if not accuracy will decrease
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except:
            print("Error converting to RGB")
            # Run image through tensorflow graph

        boxes, scores, classes, num_masks = detect_objects(frame, detection_graph, sess)

        # Draw bounding boxeses and text
        draw_box_on_image(int(num_masks[0]), score_thresh, scores, boxes, classes, im_width, im_height, frame)
        # Calculate Frames per second (FPS)
        num_frames += 1
        elapsed_time = (datetime.now() -
                        start_time).total_seconds()
        fps = num_frames / elapsed_time

        if args['display']:
            # Display FPS on frame
            draw_text_on_image("FPS : " + str("{0:.2f}".format(fps)), frame)
            cv2.imshow('Detection', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                vs.stop()
                break
    print("Average FPS: ", str("{0:.2f}".format(fps)))
except KeyboardInterrupt:
    print("Average FPS: ", str("{0:.2f}".format(fps)))
