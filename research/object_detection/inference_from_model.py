import numpy as np
import os
import tensorflow as tf
import pathlib
import glob
import fnmatch

# this is important to ensure that your script is using the local "utils" folder
os.chdir(os.getcwd())

from PIL import Image

# export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim # Don't forget to run this on the research folder!!!

from utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

# If you don't want to use colab, change the path here!

PATH_TO_LABELS = '/content/tf-models/research/object_detection/label_map/label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

PATH_TO_UNLABELED_IMAGES_DIR = pathlib.Path('/content/tf-models/research/object_detection/unlabeled_data')
UNLABELED_IMAGE_PATHS = sorted(list(PATH_TO_UNLABELED_IMAGES_DIR.glob("*.jpg")))

def load_model(mode_dir):
    model_dir = pathlib.Path(mode_dir)

    model = tf.compat.v2.saved_model.load(str(model_dir), None)

    model = model.signatures['serving_default']

    return model

def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]

    # Run inference
    output_dict = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
                    for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    output_dict['detection_masks'], output_dict['detection_boxes'],
                    image.shape[0], image.shape[1])      
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                            tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict

detection_model = load_model('/content/tf-models/research/fine_tuned_model/saved_model')

def show_inference(model, image_path):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    file_name = image_path.stem
    image_np = np.array(Image.open(image_path))
    # Actual detection.
    output_dict = run_inference_for_single_image(model, image_np)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        xml_file_name=file_name,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8)

    # image = Image.fromarray(image_np)
    # image.show()

for image_path in UNLABELED_IMAGE_PATHS:
    show_inference(detection_model, image_path)

def partition_data(test_ratio):
    # counts all JPG files in each folder
    new_labeled_files_count = fnmatch.filter(os.listdir('/content/tf-models/research/object_detection/labeled_data/'), '*.jpg')
    train_images_count = len(fnmatch.filter(os.listdir('/content/tf-models/research/object_detection/train_images'), '*.jpg'))
    test_images_count = len(fnmatch.filter(os.listdir('/content/tf-models/research/object_detection/test_images'), '*.jpg'))

    if len(new_labeled_files_count) > 0: 
        # calculate the number of images in test folder (all the rest will go to the train)
        test_quantity = int((test_ratio*(
            len(new_labeled_files_count) + train_images_count + test_images_count)) / 100)

        for image in new_labeled_files_count:
            if test_quantity > 0:
                new_img_path = '/content/tf-models/research/object_detection/test_images/' + image # set path to test folder 
                new_xml_path = '/content/tf-models/research/object_detection/test_images/' + image.replace('jpg','xml')
            else:
                new_img_path = '/content/tf-models/research/object_detection/train_images/' + image # set path to train folder
                new_xml_path = '/content/tf-models/research/object_detection/train_images/' + image.replace('jpg','xml')

            old_img_path = '/content/tf-models/research/object_detection/labeled_data/' + image
            old_xml_path = '/content/tf-models/research/object_detection/labeled_data/' + image.replace('jpg','xml')

            os.rename(old_img_path, new_img_path) # move JPG
            os.rename(old_xml_path, new_xml_path) # move XML

            test_quantity -= 1 # decreases test, when 0 all the files goes to train
            
    print('IMAGES TRANSPORTED')

partition_data(20) # pass a int between 0 and 100