from models.research.object_detection.utils import label_map_util
from models.research.object_detection.utils import visualization_utils as vis_util

import pyrealsense2 as rs
import numpy as np
import cv2
import copy
import time
import tensorflow as tf
from threading import Thread, Lock
from queue import Queue

from gql import gql, Client
from gql.transport.aiohttp import AIOHTTPTransport

# https://hororolol.tistory.com/218
# protoc object_detection/protos/*.proto --python_out=.
# https://stackoverflow.com/questions/50077353/tensorflow-object-detection-api-get-labels-in-array

# ImportError: cannot import name 'builder' from 'google.protobuf.internal'  -> pip install --upgrade protobuf 실행


PATH_TO_LABELS = 'Larva_Recognition.v18-minatureonly.tfrecord/SFU/test/Characters_label_map.pbtxt'
NUM_CLASSES = 24
COLOR_HASH = {}


class CameraDetection(object):
    def __init__(self):
        self.depth_scale = None

        self.input_frame = None
        self.input_image = None
        self.input_depth = None
        self.display_image = None
        self.depth_image = None
        self._is_det_res_ready = False
        self.det_result = None
        self.profile = None
        self.hole_filling = None
        self.boxes = None

        self.lock = Lock()

        self.input_queue = Queue()
        self.disaply_queue = Queue()

        self.input_thread = Thread(target=self.input_process)
        self.input_thread.start()

        self.detection_thread = Thread(target=self.detection_process)
        self.detection_thread.start()

        self.display_thread = Thread(target=self.display_process)
        self.display_thread.start()

    def detection_process(self):
        # download model from: https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API#run-network-in-opencv
        print("[INFO] Loading model...")
        PATH_TO_CKPT = "fine_tuned_model3/frozen_inference_graph.pb"

        # Load the Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.compat.v1.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.compat.v1.import_graph_def(od_graph_def, name='')
            sess = tf.compat.v1.Session(graph=detection_graph)

        # Input tensor is the image
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        print("[INFO] Model loaded.")

        while True:
            self.lock.acquire()
            input_image = copy.deepcopy(self.input_image)
            self.lock.release()

            if input_image is None:
                time.sleep(0.003)
                continue

            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: input_image})

            boxes = np.squeeze(boxes)
            classes = np.squeeze(classes).astype(np.int32)
            scores = np.squeeze(scores)

            if self._is_det_res_ready is False:
                self.det_result = [boxes, classes, scores, int(num)]
                self._is_det_res_ready = True


    def input_process(self):
        # Configure depth and color streams
        pipeline = rs.pipeline()
        config = rs.config()
        profile = pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        hole_filling = rs.hole_filling_filter(1)
        config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.y8, 30)  # 추가
        # depth_image = self.depth_image

        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            color_image = cv2.cvtColor(np.asanyarray(color_frame.get_data()),cv2.COLOR_BGR2RGB)
            depth_frame = frames.get_depth_frame()

            colorizer = rs.colorizer()
            colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())  # 추가???

            scaled_size = (color_frame.width, color_frame.height)
            image_expanded = np.expand_dims(color_image, axis=0)
            # Perform the actual detection by running the model with the image as input

            self.lock.acquire()
            try:
                self.input_image = image_expanded
                self.display_image = color_image
                self.depth_image = colorized_depth
                self.profile = profile
                self.hole_filling = hole_filling
                self.input_frame = color_frame
                self.input_depth = depth_frame
            except Exception as e:
                print("카메라 인풋 에러")
            self.lock.release()



            time.sleep(0.005)

    def display_process(self):
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        print(category_index)
        #print(category_index[1]['name'])
        count =0
        counts= {"sea":0,"ether":0,"space":0}
        wordcount=[]
        wordscount =[]
        global results
        results = []
        while True:
            self.lock.acquire()
            display_image = self.display_image
            profile = self.profile
            hole_filling = self.hole_filling
            input_depth = self.input_depth
            self.lock.release()
            if display_image is None:
                time.sleep(0.003)
                continue

            if self._is_det_res_ready is True:
                _boxes, _classes, _scores, _num = self.det_result
                self._is_det_res_ready = False
                for idx in range(int(_num)):
                    class_ = _classes[idx]
                    score = _scores[idx]
                    box = _boxes[idx]

                    if class_ not in COLOR_HASH:
                        COLOR_HASH[class_] = tuple(np.random.choice(range(256), size=3))

                    if score > 0.6:
                        left = int(box[1] * self.input_frame.width)
                        top = int(box[0] * self.input_frame.height)
                        right = int(box[3] * self.input_frame.width)
                        bottom = int(box[2] * self.input_frame.height)

                        p1 = (left, top)
                        p2 = (right, bottom)
                        # draw box
                        r, g, b = COLOR_HASH[class_]
                        cv2.rectangle(display_image, p1, p2, (int(r), int(g), int(b)), 2, 1)

                labels, image = vis_util.visualize_boxes_and_labels_on_image_array(
                    display_image,
                    np.squeeze(_boxes),
                    np.squeeze(_classes).astype(np.int32),
                    np.squeeze(_scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8) # add labels in visualization_utils.py

                image

                filled_depth = hole_filling.process(input_depth)
                depth = np.asanyarray(filled_depth.get_data())
                depth = _boxes.astype(float)
                depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
                depth = depth * depth_scale
                dist, _, _, _ = cv2.mean(depth)
                # print("Detected a {0:.3} meters away.".format(dist))

                print(labels)

                time_duration = 5
                if count < 60 :
                    if dist <0.000500 :
                        #print("Sea")
                        counts["sea"] +=1
                        count+=1
                        wordcount.extend(labels)
                        print(count)
                    elif dist >0.000500 and dist <0.000520 :
                        #print("ether")
                        counts["ether"] +=1
                        count+=1
                        wordcount.extend(labels)
                        print(count)
                    elif dist > 0.000520 :
                        #print("Space")
                        counts["space"] +=1
                        count+=1
                        wordcount.extend(labels)
                        print(count)
                elif count > 59 :
                    global max_key
                    max_key = max(counts,key=counts.get)
                    print(max_key)
                    for i in category_index:
                        wordscount.append(wordcount.count(category_index[i]['name']))
                    #print(wordscount)
                    #print(len(wordscount))
                    #print(len(category_index))
                    for s in category_index :
                        if wordscount[s-1] > 0 :
                            results.append(category_index[s]['name'])
                    print(results)
                    webclient()
                    time.sleep(time_duration)
                    count = 0
                    results = []
                    wordscount = []
                    counts = dict.fromkeys(counts,0)


            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', display_image) # https://github.com/IntelRealSense/librealsense/issues/4542 Color issue

            cv2.waitKey(1)
            time.sleep(0.01)

    def stop(self):
        self.input_thread.join(5)
        self.detection_thread.join(5)
        self.display_thread.join(5)


def main():
    cam_process = CameraDetection()
    while True:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            print("press keyboard interrupt")
            cam_process.stop()

def webclient():
    # Select your transport with a defined url endpoint
    transport = AIOHTTPTransport(url="http://maemesoft.iptime.org:8053/graphql")

    # Create a GraphQL client using the defined transport
    client = Client(transport=transport, fetch_schema_from_transport=True)

    # Provide a GraphQL query
    query = gql(
        """
        mutation Mutation($place: String!) {
        setPlace(place: $place)
    }
    """
    )
    query2 = gql(
        """
    mutation Mutation($shape: [String!]!) {
        setShape(shape: $shape)
    }
    """
    )

    # cherry 부서짐
    # megaphone x
    # ukulele 훈련중
    # castle x
    # cup x -> Pocket Watch가 cup으로 대체
    # mic 이거 부서짐
    # magnifyingGlass 훈련중
    # ['Minature_Airplane', 'Minature_Skull']
    res = results.copy()
    for i in range(len(res)):
        if (res[i] == "Minature_Airplane"):
            res[i] = "aircraft"
        elif (res[i] == "Minature_Pocket_Watch"):
            res[i] = "cup"  # 이거 부서짐
        elif (res[i] == "Minature_Skull"):
            res[i] = "skull"
        elif (res[i] == "Minature_Sofa"):
            res[i] = "sofa"
        elif (res[i] == "Minature_Statue"):
            res[i] = "marble_bust"
        elif (res[i] == "Miniature_Mushroom"):
            res[i] = "mushroom"
        elif (res[i] == "Miniature_Horse"):
            res[i] = "horse"
        elif (res[i] == "Miniature_Elephant"):
            res[i] = "elephant"

    # Execute the query on the transport
    params = {"place": max_key}
    params2 = {"shape": res}
    result = client.execute(query, variable_values=params)
    result2 = client.execute(query2, variable_values=params2)
    print(result)
    print(result2)

if __name__ == '__main__':
    main()
    print(max_key)

