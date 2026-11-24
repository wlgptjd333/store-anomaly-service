"""시간 순서 1D-CNN — 학습(unmanned-store-anomaly)과 추론(store-anomaly-service)이 같이 쓰는 파일.

서비스 ai_service/src/ 에 그대로 복사한다 (scripts/update_model.sh 가 같은지 확인).

입력: ExtraTrees 와 같은 30단계 × Frame Feature 40 (펼친 1,200값, t00_*, t01_* … 순서)
  → (채널 80 = 관절 40 + 변화량 40, 시간 30). 채널마다 학습 데이터 평균·표준편차로 맞춤.
모델: Conv1d(5) → Conv1d(5) → Conv1d(3), 시간 방향 평균·최댓값 → 17 Class.
  필터가 시간 방향으로 미끄러져서 "몇 번째 단계에" 보다 "어떤 움직임 모양이 있나"를 본다
  → 사람마다 동작 시작 시점이 다른 whole 창에 강함 (docs/DESIGN_DECISIONS.md 16번).
학습: 매 묶음마다 시간 자르기(앞뒤 0~15%)·작은 잡음, 120 epoch, label smoothing 0.1. 좌우 반전은 호출하는 쪽에서 더함.
저장: 난수 여러 개로 학습한 모델을 한 파일에 (평균해서 씀). torch.load(weights_only=True) 로 읽을 수 있는 값만 넣음.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


def to_seq(X: np.ndarray, steps: int, n_frame: int) -> np.ndarray:
    """(n, steps × n_frame) → (n, 2 × n_frame, steps): 관절 + 변화량 (첫 단계 변화량 0)"""
    s = np.asarray(X, dtype=np.float32).reshape(len(X), steps, n_frame)
    v = np.diff(s, axis=1, prepend=s[:, :1])
    return np.concatenate([s, v], 2).transpose(0, 2, 1).astype(np.float32)


class CNN(nn.Module):
    def __init__(self, c_in: int, n_cls: int, w: int = 64, p: float = 0.3):
        super().__init__()

        def block(a, b, k):
            return nn.Sequential(nn.Conv1d(a, b, k, padding=k // 2), nn.BatchNorm1d(b), nn.ReLU())
        self.f = nn.Sequential(block(c_in, w, 5), block(w, w, 5), nn.Dropout(p), block(w, 2 * w, 3))
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(4 * w, n_cls))

    def forward(self, x):
        h = self.f(x)
        return self.head(torch.cat([h.mean(2), h.amax(2)], 1))


def crop_batch(x: torch.Tensor, cut: float = 0.15) -> torch.Tensor:
    """시간 자르기: 앞뒤를 0~cut 만큼 잘라 다시 같은 단계 수로 (선형 보간)"""
    n, c, t = x.shape
    a = torch.rand(n, device=x.device) * cut
    b = 1 - torch.rand(n, device=x.device) * cut
    grid = a[:, None] + (b - a)[:, None] * torch.linspace(0, 1, t, device=x.device)[None]
    pos = grid * (t - 1)
    lo = pos.floor().long().clamp(0, t - 1)
    hi = (lo + 1).clamp(max=t - 1)
    w = (pos - lo)[:, None, :]

    def gather(idx):
        return torch.gather(x, 2, idx[:, None, :].expand(n, c, t))
    return gather(lo) * (1 - w) + gather(hi) * w


def fit_cnn(Xseq: np.ndarray, y: np.ndarray, seed: int, n_cls: int, epochs: int = 120,
            device: str | None = None, width: int = 64):
    """Xseq (n, C, T), y 정수 0..n_cls-1 → (eval 상태 모델, mu, sd)"""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True     # 같은 난수면 같은 결과 (GPU 합산 순서 고정)
    torch.backends.cudnn.benchmark = False
    mu = Xseq.mean((0, 2), keepdims=True).astype(np.float32)
    sd = (Xseq.std((0, 2), keepdims=True) + 1e-6).astype(np.float32)
    xt = torch.tensor((Xseq - mu) / sd, device=device)
    yt = torch.tensor(y, device=device)
    model = CNN(xt.shape[1], n_cls, w=width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * ((len(xt) + 31) // 32))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(xt), device=device)
        for i in range(0, len(xt), 32):
            idx = perm[i:i + 32]
            xb = crop_batch(xt[idx])
            xb = xb + 0.05 * torch.randn_like(xb)
            opt.zero_grad()
            loss_fn(model(xb), yt[idx]).backward()
            opt.step()
            sched.step()
    model.eval()
    return model, mu, sd


def predict_cnn(model: nn.Module, mu: np.ndarray, sd: np.ndarray, Xseq: np.ndarray) -> np.ndarray:
    dev = next(model.parameters()).device
    with torch.no_grad():
        x = torch.tensor((Xseq - mu) / sd, device=dev)
        return torch.softmax(model(x), 1).cpu().numpy()


class CnnEnsemble:
    """난수 여러 개로 학습한 1D-CNN 들의 확률 평균 (같은 정규화 값을 씀)"""

    def __init__(self, models: list[nn.Module], mu: np.ndarray, sd: np.ndarray, steps: int, n_frame: int,
                 n_cls: int, width: int = 64, seeds: list[int] | None = None):
        self.models, self.mu, self.sd = models, mu, sd
        self.steps, self.n_frame, self.n_cls, self.width = steps, n_frame, n_cls, width
        self.seeds = seeds or []

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, steps: int, n_frame: int, n_cls: int, seeds, epochs: int = 120):
        """X (n, steps × n_frame) 펼친 값. 정규화 값은 첫 모델과 같게 (모두 같은 데이터라 같음)."""
        Xs = to_seq(X, steps, n_frame)
        fitted = [fit_cnn(Xs, y, s, n_cls, epochs) for s in seeds]
        models = [m.cpu() for m, _, _ in fitted]
        return cls(models, fitted[0][1], fitted[0][2], steps, n_frame, n_cls, seeds=list(seeds))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = to_seq(X, self.steps, self.n_frame)
        return np.mean([predict_cnn(m, self.mu, self.sd, Xs) for m in self.models], 0)

    def save(self, path) -> None:
        torch.save({"version": 1, "steps": self.steps, "n_frame": self.n_frame, "n_cls": self.n_cls,
                    "width": self.width, "seeds": self.seeds,
                    "mu": torch.from_numpy(self.mu), "sd": torch.from_numpy(self.sd),
                    "state_dicts": [{k: v.cpu() for k, v in m.state_dict().items()} for m in self.models]}, path)

    @classmethod
    def load(cls, path) -> "CnnEnsemble":
        d = torch.load(path, map_location="cpu", weights_only=True)
        models = []
        for sd_ in d["state_dicts"]:
            m = CNN(2 * d["n_frame"], d["n_cls"], w=d["width"])
            m.load_state_dict(sd_)
            m.eval()
            models.append(m)
        return cls(models, d["mu"].numpy(), d["sd"].numpy(), d["steps"], d["n_frame"], d["n_cls"],
                   width=d["width"], seeds=list(d["seeds"]))
