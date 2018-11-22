#!/usr/bin/env python
from __future__ import print_function
from .code import hopenet, utils
from .config import Config
import cv2
import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image


class HopenetPose(object):
    def __init__(self):
        self.config = Config()
        self.transformations = transforms.Compose([transforms.Scale(224),
                                              transforms.CenterCrop(224), transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.init_model()

    def init_model(self):
        model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
        saved_state_dict = torch.load(self.config.MODEL)
        model.load_state_dict(saved_state_dict)
        idx_tensor = [idx for idx in xrange(66)]
        idx_tensor = torch.FloatTensor(idx_tensor)
        if self.config.USE_CUDA:
            model = model.cuda()
            idx_tensor = idx_tensor.cuda()
        model.eval()
        self.model = model
        self.idx_tensor = idx_tensor

    def get_pose(self, face_region):
        width, height = face_region.shape[:2]
        img = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = self.transformations(img)
        img = Variable(img)
        if self.config.USE_CUDA:
            img = img.cuda()
        img_shape = img.size()
        img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
        yaw, pitch, roll = self.model(img)

        yaw_predicted = F.softmax(yaw)
        pitch_predicted = F.softmax(pitch)
        roll_predicted = F.softmax(roll)

        yaw_predicted = torch.sum(yaw_predicted.data[0] * self.idx_tensor) * 3 - 99
        pitch_predicted = torch.sum(pitch_predicted.data[0] * self.idx_tensor) * 3 - 99
        roll_predicted = torch.sum(roll_predicted.data[0] * self.idx_tensor) * 3 - 99

        return yaw_predicted.item(), pitch_predicted.item(), roll_predicted.item()

    def draw_axis(self, face_region, yaw, pitch, roll, width, height):
        x_min = 0
        x_max = width
        y_min = 0
        y_max = height
        bbox_width = width
        bbox_height = height
        utils.draw_axis(face_region, yaw, pitch, roll, tdx = (x_min + x_max) / 2, tdy= (y_min + y_max) / 2, size = bbox_height/2)


def main():
    hopenet = HopenetPose()
    img = cv2.imread(hopenet.config.TEST_IMAGE)
    y, p, r = hopenet.get_pose(img)
    # print (y.item(), p.item(), r.item())
    print (y, p, r)
    width, height = img.shape[:2]
    hopenet.draw_axis(img, y, p, r, width, height)
    cv2.imshow("test result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
