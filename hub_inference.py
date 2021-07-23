import torch


def hub_model(img):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5x, custom
    results = model(img)
    results.print()  # or .show(), .save(), .crop(), .pandas(), etc.


def hub_local(img, weight_path):
    model = torch.hub.load('../yolov5', 'custom', path=weight_path, source='local')
    results = model(img)
    results.print()  # or .show(), .save(), .crop(), .pandas(), etc.


if __name__ == '__main__':
    hub_local('sample/zidane.jpg', './weights/yolov5s.pt')
