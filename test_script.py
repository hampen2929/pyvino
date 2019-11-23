import sys
from pyvino.model import Model
from pyvino.detector import DetectorFace, DetectorHumanPose
from pyvino.util import imshow, cv2pil
import cv2
import numpy as np
from argparse import ArgumentParser


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-t', '--task',
                      help='task for test.',
                      default='detect_face', type=str)
    return parser


def main():
    args = build_argparser().parse_args()
    test_image = './data/test/person1.jpg'
    frame = cv2.imread(test_image)
    
    # model = Model(args.task)
    # new_frame = model.compute(frame)
    
    # model = DetectorFace(model_fp='FP16')
    model = DetectorHumanPose()
    new_frame = model.compute(frame, frame_flag=True, mask_flag=True)
    new_frame = new_frame['frame']

    new_frame = np.asarray(cv2pil(new_frame))
    imshow(new_frame)

if __name__ == '__main__':
    sys.exit(main() or 0)