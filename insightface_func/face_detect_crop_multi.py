from __future__ import division
import collections
import numpy as np
import glob
import os
import os.path as osp
import cv2
from insightface.model_zoo import model_zoo
from insightface.utils import face_align
from .common import Face

__all__ = ['Face_detect_crop', 'Face']

Face = collections.namedtuple('Face', [
    'bbox', 'kps', 'det_score', 'embedding', 'gender', 'age',
    'embedding_norm', 'normed_embedding',
    'landmark'
])

Face.__new__.__defaults__ = (None, ) * len(Face._fields)


class Face_detect_crop:
    def __init__(self, name, root='~/.insightface_func/models'):
        self.models = {}
        root = os.path.expanduser(root)
        onnx_files = glob.glob(osp.join(root, name, '*.onnx'))
        onnx_files = sorted(onnx_files)
        for onnx_file in onnx_files:
            if onnx_file.find('_selfgen_')>0:
                #print('ignore:', onnx_file)
                continue
            model = model_zoo.get_model(onnx_file)
            if model.taskname not in self.models:
                print('find model:', onnx_file, model.taskname)
                self.models[model.taskname] = model
            else:
                print('duplicated model task type, ignore:', onnx_file, model.taskname)
                del model
        assert 'detection' in self.models
        self.det_model = self.models['detection']


    def prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640)):
        self.det_thresh = det_thresh
        assert det_size is not None
        print('set det-size:', det_size)
        self.det_size = det_size
        for taskname, model in self.models.items():
            if taskname=='detection':
                model.prepare(ctx_id, input_size=det_size)
            else:
                model.prepare(ctx_id)

    def get(self, img, crop_size, max_num=0):
        bboxes, kpss = self.det_model.detect(img,
                                             threshold=self.det_thresh,
                                             max_num=max_num,
                                             metric='default')
        if bboxes.shape[0] == 0:
            return None
        ret = []

        align_img_list = []
        M_list = []
        for i in range(bboxes.shape[0]):
            kps = None
            if kpss is not None:
                kps = kpss[i]
            M, _ = face_align.estimate_norm(kps, crop_size, mode ='None') 
            align_img = cv2.warpAffine(img, M, (crop_size, crop_size), borderValue=0.0)
            align_img_list.append(align_img)
            M_list.append(M)
        
        return align_img_list, M_list
    
    def draw_faces(self, img, max_num=0):
        faces = self.detect_faces(img, max_num=max_num)

        if len(faces) == 0:
            return None
        
        dimg = img.copy()

        for i in range(len(faces)):
            face = faces[i]
            box = face.bbox.astype(np.int)
            color = (0, 0, 255)
            
            cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(dimg,'%d'%(i), (box[0]-1, box[1]-4),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),1)

        return dimg

    def detect_faces(self, img, max_num=0):
        bboxes, kpss = self.det_model.detect(img, max_num=max_num, metric='default')

        if bboxes.shape[0] == 0:
            return []

        ret = []

        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            
            kps = None
            
            if kpss is not None:
                kps = kpss[i]

            ret.append(Face(bbox=bbox, kps=kps, det_score=det_score))

        return ret



        
