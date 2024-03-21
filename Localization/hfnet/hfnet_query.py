import cv2
import json
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
tf.contrib.resampler
import os
import time

class HFNet:
    def __init__(self, model_path, outputs):
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.session = tf.Session(config=sess_config)
        self.image_ph = tf.placeholder(tf.float32, shape=(None, None, 3))

        net_input = tf.image.rgb_to_grayscale(self.image_ph[None])
        tf.saved_model.loader.load(
            self.session, [tag_constants.SERVING], str(model_path),
            clear_devices=True,
            input_map={'image:0': net_input})

        graph = tf.get_default_graph()
        self.outputs = {n: graph.get_tensor_by_name(n+':0')[0] for n in outputs}
        self.nms_radius_op = graph.get_tensor_by_name('pred/simple_nms/radius:0')
        self.num_keypoints_op = graph.get_tensor_by_name('pred/top_k_keypoints/k:0')

    def inference(self, image, nms_radius=4, num_keypoints=1000):
        inputs = {
            self.image_ph: image[..., ::-1].astype(np.float),
            self.nms_radius_op: nms_radius,
            self.num_keypoints_op: num_keypoints,
        }
        return self.session.run(self.outputs, feed_dict=inputs)

def get_hfnet(model_path) :
    model_path = Path(model_path, 'saved_models/hfnet')
    outputs = ['global_descriptor', 'keypoints', 'local_descriptors', 'scores']
    return HFNet(model_path, outputs)

def main(image_dir, output_dir, nms_radius=4, keypoints=4096,model_path=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    hfnet = get_hfnet(model_path)
    mode = cv2.IMREAD_COLOR
    image = cv2.imread(image_dir, mode)
    start=time.time()
    db = hfnet.inference(image, nms_radius=nms_radius, num_keypoints=keypoints)
    if os.path.exists(os.path.join(output_dir,'global_features.json')):
        with open(os.path.join(output_dir,'global_features.json'),'r') as f:
            file=json.load(f)
            file.update({image_dir:{'time':time.time()-start,'gd':db['global_descriptor'].tolist()}})
    else:
        file={image_dir:{'time':time.time()-start,'gd':db['global_descriptor'].tolist()}}
    with open(os.path.join(output_dir, 'global_features.json'), 'w') as f:
        json.dump(file,f)

if __name__ == '__main__':
    parser = ArgumentParser(description='Features Extract')
    parser.add_argument('-i', '--image', type=str, help='image')
    parser.add_argument('-model', '--model_dir', type=Path, help='model dir')
    parser.add_argument('-output', '--output',type=str, default='output',help='Output folder for featues')
    parser.add_argument('-nm', '--nms_radius',type=int, default=4)
    parser.add_argument('-keypoints', '--keypoints',type=int, default=4096)
    args=parser.parse_args()
    main(args.image, args.output, nms_radius=args.nms_radius, keypoints=args.keypoints,model_path=args.model_dir)
