# -*- coding: utf-8 -*-
"""
Function: BaseDetector
Author: Wujia
Create Time: 2020/8/18 13:49
"""
import torch
from torchvision.ops import nms
import sys
sys.path.append('./Networks')
from RetinaFace import RetinaFace
from utils import PriorBox, Decode, no_deform_resize_pad, rgb_mean_gpu

torch.set_num_threads(1)

class FaceDetector(object):
    def __init__(self, model_path, gpu_ids, layers, score_thresh=0.5):
        """
        检测整体基本流程
        :param model_path: 模型路径
        :param gpu_ids: gpu序列号
        :param layers: 18 , 50
        :param score_thresh: 置信度过滤
        """
        self.keep_top_k = 100
        self.nms_threshold = 0.3
        self.nms_score = score_thresh
        self.nms_threshold = self.nms_threshold

        self.test_size = 640
        self.__model_path = model_path
        self.__gpu_ids = gpu_ids

        self.device = torch.device('cuda:{}'.format(str(gpu_ids)) if torch.cuda.is_available() else 'cpu')

        self.layers = layers
        self.model = RetinaFace(self.layers)
        self.model = self.__load_model(self.model, self.__model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.priorbox = PriorBox(box_specs_list=[[(0.6, 0.5), (0.75, 1.), (0.9, 1.)],
                                                 [(0.2, 0.5), (0.4, 1.), (0.6, 1.)],
                                                 [(0.05, 0.5), (0.1, 1.), (0.2, 1.)],
                                                 [(0.0125, 0.5), (0.025, 1.), (0.05, 1.)]],
                                 base_anchor_size=[1.0, 1.0])
        self.priors = self.priorbox.generate(feature_map_shape_list=[(10, 10), (20, 20), (40, 40), (80, 80)],
                                             im_height=640,
                                             im_width=640)
        self.priors = self.priors.to(self.device)

        self.mean = torch.Tensor([104, 117, 123]).to(self.device)
        self.variance = torch.Tensor([0.1, 0.2]).to(self.device)
        self.Decode = Decode(self.priors.data, self.variance)

    def detect(self, detect_input):
        images, percent = self.preprocess(detect_input)                                  #前处理
        loc, conf, landms = self.inference(images)                                       #推理
        boxes, landms, scores = self.decode(loc, conf, landms, percent)                  #解码
        boxes, landms, scores = self.postprocess(boxes, landms, scores)                  #后处理
        if len(scores) == 0:
            return None, None, None
        else:
            return boxes, landms, scores

    def __check_keys(self, model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        print('Missing keys:{}'.format(len(missing_keys)))
        print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
        print('Used keys:{}'.format(len(used_pretrained_keys)))
        assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
        return True

    def __remove_prefix(self, state_dict, prefix):
        ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
        print('remove prefix \'{}\''.format(prefix))
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}

    def __load_model(self, model,model_path):
        print('Loading pretrained model from {}'.format(model_path))
        if self.__gpu_ids == None:
            device = torch.cuda.current_device()
            pretrained_dict = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(device))

        else:
            pretrained_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self.__remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = self.__remove_prefix(pretrained_dict, 'module.')
        self.__check_keys(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        return model

    def preprocess(self, image):
        """Preprocess"""
        image_t, percent = no_deform_resize_pad(image, self.test_size)
        image_t = rgb_mean_gpu(image_t, self.mean, self.device)
        images = image_t.permute(2, 0, 1).reshape(1, 3, self.test_size, self.test_size)
        return images, percent


    def inference(self, images):
        """网络推理"""
        with torch.no_grad():
            loc, conf, landms = self.model(images)
        return loc, conf, landms

    def decode(self, loc, conf, landms, percent):
        """推理结果的解码"""
        boxes = self.Decode.decode_bbox(loc.squeeze(0).data)
        landms = self.Decode.decode_landm(landms.squeeze(0).data)
        detect_boxes = boxes * self.test_size * percent
        detect_landmas = landms * self.test_size * percent
        scores = conf.squeeze(0).data[:, 1]
        return detect_boxes, detect_landmas, scores

    def postprocess(self, boxes, landms, scores):
        """后处理NMS"""
        inds = torch.where(scores >= self.nms_score)[0]
        boxes = boxes[inds]
        scores = scores[inds]
        landms = landms[inds]

        keep = nms(boxes, scores, self.nms_threshold)
        boxes = boxes[keep]
        scores = scores[keep]
        landms = landms[keep]

        boxes = boxes[:self.keep_top_k, :].cpu().numpy()
        scores = scores[:self.keep_top_k].cpu().numpy()
        landms = landms[:self.keep_top_k, :].cpu().numpy()
        return boxes, landms, scores



