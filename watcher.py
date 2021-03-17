import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os,sys
import glob
import threading
import time
from os import rename, remove
import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import logging
import boto3
from botocore.exceptions import ClientError

import ikncu

class S3:

    @staticmethod
    def Upload(file, object_name):

        """Upload a file to an S3 bucket

        :param file_name: File to upload
        :param bucket: Bucket to upload to
        :param object_name: S3 object name. If not specified then file_name is used
        :return: True if file was uploaded, else False
        """

        s3_client = boto3.client('s3')
        try:
            response = s3_client.upload_file(file, ikncu.s3_backet_frames, object_name)
        except ClientError as e:
            logging.error(e)
            return False
        return True
 
class Detector:
    PATH_TO_CKPT = os.path.join(ikncu.path_to_model,'frozen_inference_graph.pb')
    PATH_TO_LABELS = os.path.join(ikncu.path_to_data,'mscoco_label_map.pbtxt')
    NUM_CLASSES = 90

    def __init__(self):
        label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=self.NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

        # find category ID for person
        self.person_id = None
        for cat in categories: 
            if cat['name'] == 'person': 
                self.person_id = cat['id'] 
                break

        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.compat.v1.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.compat.v1.Session(graph=detection_graph)

        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        self.detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')


    def process(self,frame):
        frame = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)
        frame.setflags(write=1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_expanded = np.expand_dims(frame_rgb, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: frame_expanded})

        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.40)

        exists = False
        for i in range (0,int(num[0])):
            if classes[0][i] == self.person_id:
                exists = True
                break

        if exists:
            return frame
        else: 
            return None    


class Watcher:
    DIRECTORY_TO_WATCH = ikncu.path_to_new

    def __init__(self):
        self.observer = Observer()

    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler, self.DIRECTORY_TO_WATCH, recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(1)
        except:
            self.observer.stop()
            print("Error")

        self.observer.join()

class Splitter:

    @staticmethod
    def process(path): 
        
        #wait for upload to finish
        size=os.path.getsize(path)
        same=0
        while same < 3:
            time.sleep(1)

            if size == os.path.getsize(path):
                same = same + 1
            else:
                same = same - 1
                size = os.path.getsize(path)

        print("     %s: start processing" % (path))

        try:
            file = os.path.basename(path)

            sep = '.'

            video = cv2.VideoCapture(path)
            fps = int(video.get(cv2.CAP_PROP_FPS))
            print("   %6.2f fps" % fps)

            # capture 1 frame per second
            count = 0
            start = time.process_time()
            success, frame = video.read()
            while success:
                success, frame = video.read()
                if not success:
                    print("   cannot read!")
                    break

                if count % fps == 0:
                    original_frame_file = "%s-%#05d.jpg" % (file.split(sep, 1)[0], count + 1)
                    processed_frame_file = "%s-%#05d-processed.jpg" % (file.split(sep, 1)[0], count + 1)

                    original_frame_file_path = os.path.join(ikncu.path_to_frames, original_frame_file)
                    processed_frame_file_path = os.path.join(ikncu.path_to_frames, processed_frame_file)

                    ### detect a person
                    processed_frame = detector.process(frame)

                    if processed_frame is not None:
                        print("    saving original to %s" % (original_frame_file_path))
                        print("    saving processed to %s" % (processed_frame_file_path))
                        cv2.imwrite(original_frame_file_path, frame)
                        #cv2.imwrite(processed_frame_file_path, processed_frame)
                        #S3.Upload(original_frame_file_path,original_frame_file)
                        S3.Upload(processed_frame_file_path,processed_frame_file)

                count = count + 1

            video.release()
            end = time.process_time() - start
            if fps>0:
                print("    % d frames in %.2f seconds" % (count / fps, end - start))
            else:
                print("    0 fps")
        
        except:
            print("[processVideos] Unexpected error:", sys.exc_info()[0])
            processes.remove(path) 
            raise    
            remove(path)

        try:
            rename(path, os.path.join(ikncu.path_to_archive, file))
        except:
            print("Warning: file already exists")
         
        processes.remove(path)    
        return True

class Handler(FileSystemEventHandler):

    @staticmethod
    def file_is_ready(file):
        if os.path.exists(file):
            try:
                os.rename(file, file)
                return True
            except OSError as e:
                return False

    @staticmethod
    def on_any_event(event):
        if event.is_directory:
            return None

        elif (event.event_type == 'created') and event.src_path not in processes:
            processes.append(event.src_path)
            print("%d: %s - added to queue" % (len(processes),event.src_path))

            threading.Thread(target=Splitter.process, args=(event.src_path,)).start()


if __name__ == '__main__':
    processes = []
    detector = Detector()
    w = Watcher()
    w.run()