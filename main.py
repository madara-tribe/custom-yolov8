import argparse
import sys
from custom import YOLO


# https://docs.ultralytics.com/modes/predict/


def train(model, opt):
    results = model.train(
            data=opt.data, task='detect',
            imgsz=opt.imgsz,
            epochs=opt.epochs,
            batch=opt.train_batch)
    print(results)


def main(opt):
    print('weight is {}'.format(opt.weights))
    if opt.weights:
        model = YOLO(opt.weights)
    else:
        model = YOLO()
    if opt.mode=='train':
        train(model, opt)
    elif opt.mode=='valid':
        model.val()
    elif opt.mode=='predict':
        model.predict(opt.source, save=True, imgsz=opt.imgsz, conf=opt.conf)
    elif opt.mode=='onnx':
        success = model.export(format="onnx")
        print(success)
    else:
        print('[--mode] sholud be [train/valid/predict/onnx]')
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights', type=str, default=None, help='initial weights path')
    parser.add_argument('-m', '--mode', type=str, default='train', help='train/val/predict/onnx')
    parser.add_argument('--agnostic_nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='apply image augmentation to prediction sources')
    parser.add_argument('--half', action='store_true', help='use half precision (FP16)')
    parser.add_argument('--train_batch', type=int, default=4, help='train batch size')
    parser.add_argument('--epochs', type=int, default=300, help='total epochs')
    parser.add_argument('--iou', type=float, default=0.7, help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--conf', type=float, default=0.5, help='object confidence threshold for detection')
    parser.add_argument('--imgsz', type=int, default=640, help='image size')
    parser.add_argument('--data', type=str, default='data.yaml', help='yaml for dataset')
    parser.add_argument('--source', type=str, default='datasets/road299.png', help='predict image')
    opt = parser.parse_args()
    try:
        main(opt)
    except KeyboardInterrupt:
        sys.exit(1)
        raise
