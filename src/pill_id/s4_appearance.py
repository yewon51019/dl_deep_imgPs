"""4단계: 겉모습 CNN — 자른 알약 사진에서 모양(4칸)·제형(3칸)·색(12칸, 여러 개 가능) 확률을 낸다.

ResNet18 (ImageNet 사전학습) + 출력 19개, 학습용 사진으로 학습한 가중치(attr_cnn_v1 best.pt). 입력 224 정사각형 RGB.
(exp_cns/attr_cnn_cns.py 의 build_model·predict 와 같은 계산) CUDA GPU 가 있으면 GPU, 없으면 CPU.
"""
import cv2
import numpy as np

from . import config as C
from .imageio import fit_side

SIDE = 224
SHAPES = ["원형", "장방형", "타원형", "기타"]
FORMS = ["정제", "경질캡슐", "연질캡슐"]
COLORS = ["하양", "노랑", "주황", "분홍", "빨강", "갈색", "연두", "초록", "파랑", "보라", "회색", "검정"]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class AppearanceCNN:
    def __init__(self, weights=C.ATTR_WEIGHTS):
        import torch
        import torch.nn as nn
        import torchvision
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        net = torchvision.models.resnet18(weights=None)
        net.fc = nn.Linear(net.fc.in_features, len(SHAPES) + len(FORMS) + len(COLORS))
        net.load_state_dict(torch.load(weights, map_location=self.dev)["state"])
        self.net = net.to(self.dev).eval()

    def predict(self, crops_bgr):
        """자른 알약 BGR 목록 → 알약마다 확률 dict {'s_원형': .., 'f_정제': .., 'c_하양': ..}."""
        if not crops_bgr:                             # 빈 목록이면 np.stack 이 실패하므로 바로 돌려줌
            return []
        x = np.stack([cv2.cvtColor(fit_side(c, SIDE), cv2.COLOR_BGR2RGB) for c in crops_bgr])
        x = (x.astype(np.float32) / 255 - MEAN) / STD
        t = self.torch.from_numpy(x.transpose(0, 3, 1, 2).copy()).to(self.dev)
        with self.torch.no_grad():
            o = self.net(t).float().cpu()
        a, b = len(SHAPES), len(SHAPES) + len(FORMS)
        ps, pf, pc = o[:, :a].softmax(1).numpy(), o[:, a:b].softmax(1).numpy(), o[:, b:].sigmoid().numpy()
        return [{**{f"s_{c}": float(ps[i, j]) for j, c in enumerate(SHAPES)},
                 **{f"f_{c}": float(pf[i, j]) for j, c in enumerate(FORMS)},
                 **{f"c_{c}": float(pc[i, j]) for j, c in enumerate(COLORS)}} for i in range(len(crops_bgr))]


def summary(prob):
    """확률 → (가장 높은 모양, 가장 높은 제형, 0.5 넘는 색 'a|b')."""
    sh = max(SHAPES, key=lambda c: prob[f"s_{c}"])
    fo = max(FORMS, key=lambda c: prob[f"f_{c}"])
    return sh, fo, "|".join(c for c in COLORS if prob[f"c_{c}"] > 0.5)
