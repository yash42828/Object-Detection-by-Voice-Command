def detectObject():
    import numpy as np
    import os
    import six.moves.urllib as urllib
    import sys
    import tarfile
    import tensorflow as tf
    import win32
    from PIL import Image
    
    # This is needed since the notebook is stored in the object_detection folder.
    sys.path.append("..")
    from object_detection.utils import ops as utils_ops
    
    if tf.__version__ < '1.4.0':
      raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')
    
    import cv2
    cap = cv2.VideoCapture(0)
    
    # ## Env setup
    
    from utils import label_map_util
    
    from utils import visualization_utils as vis_util
    
    
    # # Model preparation
    
    # ## Variables
    #
    # Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_FROZEN_GRAPH` to point to a new .pb file.  
    #
    # By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
    
    
    # What model to download.
    MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
    
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
    
    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
    
    NUM_CLASSES = 90
    
    
    # ## Download Model
    
    # Check the frozen_inference_graph is exists at path.
    # If not ssd_mobilenet_v1_coco_2017_11_17 model is downloaded and frozen_inference_graph.pb moved to path.
    if os.path.exists(PATH_TO_FROZEN_GRAPH):
      print("ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb Exists at path. Skipping download... ")
    else:
      opener = urllib.request.URLopener()
      opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
      tar_file = tarfile.open(MODEL_FILE)
      for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
          tar_file.extract(file, os.getcwd())
    
    
    
    # ## Load a (frozen) Tensorflow model into memory.
    
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.compat.v1.GraphDef()
      with tf.io.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    
    
    # ## Loading label map
    # Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
    
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    
    
    # # Detection
    
    def run_inference_for_single_image(image, graph):
      with graph.as_default():
        with tf.compat.v1.Session() as sess:
          # Get handles to input and output tensors
          ops = tf.compat.v1.get_default_graph().get_operations()
          all_tensor_names = {output.name for op in ops for output in op.outputs}
          tensor_dict = {}
          for key in [
              'num_detections', 'detection_boxes', 'detection_scores',
              'detection_classes', 'detection_masks'
          ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
              tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(
                  tensor_name)
          if 'detection_masks' in tensor_dict:
            # The following processing is only for single image
            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
            # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
            real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, image.shape[0], image.shape[1])
            detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            # Follow the convention by adding back the batch dimension
            tensor_dict['detection_masks'] = tf.expand_dims(
                detection_masks_reframed, 0)
          image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('image_tensor:0')
    
          # Run inference
          output_dict = sess.run(tensor_dict,
                                 feed_dict={image_tensor: np.expand_dims(image, 0)})
    
          # all outputs are float32 numpy arrays, so convert types as appropriate
          output_dict['num_detections'] = int(output_dict['num_detections'][0])
          output_dict['detection_classes'] = output_dict[
              'detection_classes'][0].astype(np.uint8)
          output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
          output_dict['detection_scores'] = output_dict['detection_scores'][0]
          if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]
      return output_dict
    
    while True:
      ret, image_np = cap.read()
      # Actual detection.
      output_dict = run_inference_for_single_image(image_np, detection_graph)
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          output_dict['detection_boxes'],
          output_dict['detection_classes'],
          output_dict['detection_scores'],
          category_index,
          instance_masks=output_dict.get('detection_masks'),
          use_normalized_coordinates=True,
          line_thickness=8)
      #print(output_dict['detection_scores'])
      lbl =output_dict['detection_classes'][0]
      #print(category_index[lbl]['name'])
      win32.voice(category_index[lbl]['name'])
      cv2.imshow('Image Detction', cv2.resize(image_np, (1280,720)))
      if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
    
