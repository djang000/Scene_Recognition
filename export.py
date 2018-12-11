import tensorflow as tf
import networks.mobilenet_V2 as model
import os


from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference

class_file = open('data/meta/categories_places365.txt', 'r')

classes = []
for line in class_file:
    classes.append(line.strip().split(' ')[0][3:])
labels_to_class = dict(zip(range(len(classes)), classes))


def main():
    INPUT_POINT = "input_image"
    PB_END_POINT = "Predictions/Reshape_1"
    OUTPUT_PB_FILENAME = "output/models/SceneRecognition_MobileNetV2.pb"
    # Training model
    graph = tf.Graph()
    with graph.as_default():
        # A placeholder for a test image
        tf_test_image = tf.placeholder(dtype=tf.float32, shape=(None, 224, 224, 3), name='input_image')

        # model
        nets = model.MobileNet_V2(tf_test_image,
                                  num_classes=len(classes),
                                  is_training=False)

        # Restore ops
        saver = tf.train.Saver()
        with tf.Session(graph=graph) as sess:
            save_path = "output/models/scene_recognition_final.ckpt"
            saver.restore(sess, save_path)

            constant_graph = convert_variables_to_constants(sess,
                                                            sess.graph_def,
                                                            [INPUT_POINT, PB_END_POINT])
            optimized_constant_graph = optimize_for_inference(constant_graph,
                                                              [INPUT_POINT],
                                                              [PB_END_POINT],
                                                              tf.float32.as_datatype_enum)

            # create PB file, and we also create text file for debug
            tf.train.write_graph(optimized_constant_graph, '.', OUTPUT_PB_FILENAME, as_text=False)
            tf.train.write_graph(optimized_constant_graph, '.', OUTPUT_PB_FILENAME + ".txt", as_text=True)

        # print file size
        filesize = os.path.getsize(OUTPUT_PB_FILENAME)
        filesize_mb = filesize / 1024 / 1024
        print(str(round(filesize_mb, 3)) + " MB")
        sess.close()


if __name__ == '__main__':
    main()
