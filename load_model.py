import sys
import tensorflow as tf
import numpy as np
from datasets.base import load_data, _read_image

loaded_graph = tf.Graph()
label_choices = "0123456789abcdefghijklmnopqrstuvwxyz"


def _read_label(labels, label_choices=label_choices):
    data = []

    for c in labels:
        try:
            idx = label_choices.index(c)
            tmp = [0] * len(label_choices)
            tmp[idx] = 1
            data.extend(tmp)
        except:
            pass
    return np.array([data])


def class_to_label(y):
    label = "".join([label_choices[i] for i in y[0][0]])
    return label


def decaptcha(filename):
    with tf.Session(graph=loaded_graph) as sess:
        saver = tf.train.import_meta_graph("./model/model_20180518_084549.ckpt.meta")
        saver.restore(sess, tf.train.latest_checkpoint("./model"))

        x = loaded_graph.get_tensor_by_name("input/Placeholder:0")
        y_ = loaded_graph.get_tensor_by_name("input/Placeholder_1:0")
        keep_prob = loaded_graph.get_tensor_by_name("dropout/Placeholder:0")
        pred = loaded_graph.get_tensor_by_name("forword-prop/ArgMax:0")

        # data_dir = "datasets/captcha-5100/corpus"
        # meta, train_data, test_data = load_data(data_dir, flatten=False)
        # test_x, test_y_ = test_data.next_batch(10)

        test_x = _read_image(filename, False, 200, 45).reshape(1, 45, 200)
        test_y = _read_label("00000")
        pred_class = sess.run(
            [pred],
            feed_dict={
                x: test_x,
                y_: test_y,
                keep_prob: 1.0
            }
        )
        pred_label = class_to_label(pred_class)
        return pred_label


if __name__ == '__main__':
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "datasets/captcha-5100/corpus/test/2acfw_0.png"
    label = decaptcha(filename)
    print(filename, "->", label)
