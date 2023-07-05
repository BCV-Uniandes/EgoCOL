import os
import sys
import numpy as np
import torch
import math
sys.path.append('../annotation_API/API/')
from bounding_box import BoundingBox
from pytorch3d.ops import box3d_overlap


class mAP():

    def calcular_distancia(self, punto1, punto2):
        """
        Calcula la distancia entre dos puntos 3D.
        """
        x1, y1, z1, _ = punto1
        x2, y2, z2, _ = punto2

        distancia = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

        return distancia

    def get_dimensions(self, box1:BoundingBox, box2:BoundingBox):
        v_box1 = box1.build_box()
        v_box2 = box2.build_box()
        #Compute size
        h1 = self.calcular_distancia(v_box1[2], v_box1[3]) #v2-v3
        w1 = self.calcular_distancia(v_box1[2], v_box1[5]) #v2-v5
        d1 = self.calcular_distancia(v_box1[2], v_box1[0]) #v2-v0

        h2 = self.calcular_distancia(v_box2[2], v_box2[3]) #v2-v3
        w2 = self.calcular_distancia(v_box2[2], v_box2[5]) #v2-v5
        d2 = self.calcular_distancia(v_box2[2], v_box2[0]) #v2-v0

        h = (h1+h2)/2
        w = (w1+w2)/2
        d = (d1+d2)/2

        return w, d, h
    def construct_pred_box(self, t, w, d, h):
        x, y, z = t[0], t[1], t[2]

        half_h = h / 2
        half_w = w / 2
        half_d = d / 2

        vertices = [
        [x - half_w, y - half_d, z - half_h],  # bottom back left
        [x - half_w, y + half_d, z - half_h],  # top back left
        [x - half_w, y - half_d, z + half_h],  # bottom front left
        [x - half_w, y + half_d, z + half_h],  # top front left
        [x + half_w, y + half_d, z + half_h],  # top front right
        [x + half_w, y - half_d, z + half_h],  # bottom front right
        [x + half_w, y + half_d, z - half_h],  # top back right
        [x + half_w, y - half_d, z - half_h]   # bottom back right
        ]
        return vertices

    def compute_IoU(self, t:np.ndarray, box1:BoundingBox, box2:BoundingBox):
        w, d, h = self.get_dimensions(box1, box2)
        pred_box = self.construct_pred_box(t, w, d, h)

        box1_vertices = box1.build_box()
        box1_tensor =  torch.Tensor([[box1_vertices[3], box1_vertices[4], 
                                    box1_vertices[5], box1_vertices[2],
                                    box1_vertices[1], box1_vertices[6],
                                    box1_vertices[7], box1_vertices[0] 
                                    ]])[:,:,:3]
        box2_vertices = box2.build_box()
        box2_tensor =  torch.Tensor([[box2_vertices[3], box2_vertices[4], 
                                    box2_vertices[5], box2_vertices[2],
                                    box2_vertices[1], box2_vertices[6],
                                    box2_vertices[7], box2_vertices[0] 
                                    ]])[:,:,:3]
        
        pred_tensor = torch.Tensor([[pred_box[3], pred_box[4], 
                                    pred_box[5], pred_box[2],
                                    pred_box[1], pred_box[6],
                                    pred_box[7], pred_box[0] 
                                    ]])
        try:
            _, iou_3d_1 = box3d_overlap(box1_tensor, pred_tensor)
            _, iou_3d_2 = box3d_overlap(box2_tensor, pred_tensor)
            print(box3d_overlap(box2_tensor, pred_tensor)[1])
        except:
            return 0

        return iou_3d_1[0][0] if iou_3d_1>iou_3d_2 else iou_3d_2[0][0]




class distL2():
    def compute(self, v1:np.ndarray, v2:np.ndarray) -> float:
        d = np.linalg.norm(v1-v2)
        return d

class angularError():
    def compute(self, v1:np.ndarray, v2:np.ndarray) -> float:
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        return angle

Rz_90 = np.array([[np.cos(-np.pi/2), -np.sin(-np.pi/2), 0, 0],
                  [np.sin(-np.pi/2),  np.cos(-np.pi/2), 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1],
                 ])

class accuracy():
    def compute(self, t:np.ndarray, box1:BoundingBox, box2:BoundingBox) -> bool:
        c = (box1.center + box2.center) / 2.0

        c = np.append(c, 1.)
        c = np.matmul(Rz_90, c)
        c = c[:3] / c[3]

        d = np.linalg.norm(c-t)
        d_gt = np.linalg.norm(box1.center - box2.center)

        diag1 = np.sqrt(np.sum(box1.sizes**2))
        diag2 = np.sqrt(np.sum(box2.sizes**2))

        m = np.mean([diag1, diag2])
        delta = np.exp(-m)
        return d < 6*(d_gt + delta), d, 6*(d_gt + delta)
        #return d < 5, d, 5

    def compute_with_cosest(self, t:np.ndarray, box1:BoundingBox, box2:BoundingBox) -> bool:
        c = box1.center if sum(np.abs(box1.center-t))<sum(np.abs(box2.center-t)) else box2.center

        c = np.append(c, 1.)
        c = np.matmul(Rz_90, c)
        c = c[:3] / c[3]

        d = np.linalg.norm(c-t)
        d_gt = np.linalg.norm(box1.center - box2.center)

        diag1 = np.sqrt(np.sum(box1.sizes**2))
        diag2 = np.sqrt(np.sum(box2.sizes**2))

        m = np.mean([diag1, diag2])
        delta = np.exp(-m)
        #return d < 6*(d_gt + delta), d, 6*(d_gt + delta)
        return d < 6, d, 6


