import glob, os, random
import tensorflow as tf

from configs import config as cfg
from tensorflow.python.ops import control_flow_ops

def _smallest_size_at_least(h, w, np_smallest_side):
    smallest_side = tf.convert_to_tensor(np_smallest_side, dtype=tf.int32)
    h = tf.to_float(h)
    w = tf.to_float(w)
    smallest_side = tf.to_float(smallest_side)

    scale = tf.cond(tf.greater(h, w),
                    lambda : smallest_side / w,
                    lambda : smallest_side / h)
    new_h = tf.to_int32(h * scale)
    new_w = tf.to_int32(w * scale)
    return new_h, new_w

def _apply_with_random_selector(x, func, num_cases):
  """Computes func(x, sel), with sel sampled from [0...num_cases-1].

  Args:
    x: input Tensor.
    func: Python function to apply.
    num_cases: Python int32, number of cases to sample sel from.

  Returns:
    The result of func(x, sel), where func receives the value of the
    selector as a python integer, but sel is sampled dynamically.
  """
  sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
  # Pass the real x only to one of the func calls.
  return control_flow_ops.merge([
      func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
      for case in range(num_cases)])[0]


def _distort_color(image, color_ordering=0, fast_mode=True, scope=None):
  """
  Args:
    image: 3-D Tensor containing single image in [0, 1].
    color_ordering: Python int, a type of distortion (valid values: 0-3).
    fast_mode: Avoids slower ops (random_hue and random_contrast)
    scope: Optional scope for name_scope.
  Returns:
    3-D Tensor color-distorted image on range [0, 1]
  Raises:
    ValueError: if color_ordering not in [0, 3]
  """
  with tf.name_scope(scope, 'distort_color', [image]):
    if fast_mode:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    else:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
      elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      elif color_ordering == 3:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
      else:
        raise ValueError('color_ordering must be in [0, 3]')

    # The random_* ops do not necessarily clamp.
    return tf.clip_by_value(image, 0.0, 1.0)

def _distorted_bounding_box_crop(image, bbox,
                                 min_object_covered=0.1,
                                 aspect_ratio_range=(0.75, 1.33),
                                 area_range=(0.875, 1.0),
                                 max_attempts=100,
                                 scope=None):
    with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bbox]):
        # Each bounding box has shape [1, num_boxes, box coords] and
        # the coordinates are ordered [ymin, xmin, ymax, xmax].

        # A large fraction of image datasets contain a human-annotated bounding
        # box delineating the region of the image containing the object of interest.
        # We choose to create a new bounding box for the object which is a randomly
        # distorted version of the human-annotated bounding box that obeys an
        # allowed range of aspect ratios, sizes and overlap with the human-annotated
        # bounding box. If no box is supplied, then we assume the bounding box is
        # the entire image.
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=bbox,
            min_object_covered=min_object_covered,
            aspect_ratio_range=aspect_ratio_range,
            area_range=area_range,
            max_attempts=max_attempts,
            use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

        # Crop the image to the specified bounding box.
        cropped_image = tf.slice(image, bbox_begin, bbox_size)
        return cropped_image, distort_bbox


def _preprocess_for_training(image, bbox=None):
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    """ step 1. crop the image """
    if bbox is None:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
                           dtype=tf.float32,
                           shape=[1, 1, 4])
    image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0), bbox)
    tf.summary.image('image_with_bounding_boxes', image_with_box)
    cropped_image, crop_bbox = _distorted_bounding_box_crop(image, bbox)

    """ step 2. resize images """
    cropped_image = tf.expand_dims(cropped_image, 0)
    resize_image = tf.image.resize_bilinear(cropped_image, [224, 224], align_corners=False)
    resize_image = tf.squeeze(resize_image, 0)
    tf.summary.image('cropped_resize_image', tf.expand_dims(resize_image, 0))
    resize_image.set_shape([224, 224, 3])

    """ step 3. Randomly flip the image horizontally. """
    flip_lr_image = tf.image.random_flip_left_right(resize_image)

    """ step 4. Randomly flip the image horizontally. """
    flip_ud_image = tf.image.random_flip_up_down(flip_lr_image)

    """ step 5. Randomly distort the colors. There are 4 ways to do it. """
    distorted_image = _apply_with_random_selector(
        flip_ud_image,
        lambda x, ordering: _distort_color(x, ordering, fast_mode=True),
        num_cases=4)

    tf.summary.image('final_distorted_image', tf.expand_dims(distorted_image, 0))

    return distorted_image

def _preprocess_for_test(image):
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image,
                                     [cfg.FLAGS.image_min_size, cfg.FLAGS.image_min_size],
                                     align_corners=False)
    image = tf.squeeze(image, [0])
    image.set_shape([224, 224, 3])

    return image

def _find_image_files(image_dir, train_info):
    files = []
    labels = []

    with open(train_info, 'r') as txt:
        datas = [l.strip() for l in txt.readlines()]
        for f in datas:
            img_name, label = f.split(' ')
            if cfg.FLAGS.dataset_split_name is 'val':
                files.append(os.path.join(image_dir, img_name))
            else:
                files.append(image_dir+img_name)
            labels.append(int(label))

        shuffled_index = list(range(len(files)))

        random.shuffle(shuffled_index)

        shuffled__files = [files[i] for i in shuffled_index]

        shuffled_labels = [labels[i] for i in shuffled_index]
        return shuffled__files, shuffled_labels

def image2queue(split_name, is_training, im_batch):
    image_dir = os.path.join(cfg.FLAGS.dataset_dir,'train_images')
    image_names = os.path.join(cfg.FLAGS.dataset_dir, 'meta/places365_%s.txt' % split_name)

    filenames, classes = _find_image_files(image_dir, image_names)

    images = tf.convert_to_tensor(filenames)
    labels = tf.convert_to_tensor(classes)
    input_queue = tf.train.slice_input_producer([images, labels])
    image = tf.read_file(input_queue[0])
    image = tf.image.decode_image(image, channels=3)

    """ preprocessing which is random flipping, min size resizeing and zero mean """
    if is_training is True:
        print("Start ==> preprocessing for training")
        preprocessed_image = _preprocess_for_training(image)
    else:
        print("Start ==> preprocessing for test")
        preprocessed_image = _preprocess_for_test(image)

    min_after_dequeue = 3 * im_batch
    capacity = min_after_dequeue * 2
    batches = tf.train.shuffle_batch([preprocessed_image, input_queue[1]],
                                     batch_size=im_batch,
                                     capacity=capacity,
                                     min_after_dequeue=min_after_dequeue,
                                     num_threads=cfg.FLAGS.num_preprocessing_threads)

    tf.summary.image(name='distorted_input', tensor=batches[0], max_outputs=1)
    return batches[0], batches[1]

def get_dataset(split_name,
                is_training,
                im_batch=4):

    return image2queue(split_name, is_training, im_batch)
