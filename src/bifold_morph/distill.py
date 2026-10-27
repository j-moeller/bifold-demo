"""Teaching a small, fast network the edits StyleGAN2 can make: the recipe of the face-changing apps.

StyleGAN2 can turn the faces it draws into the other gender or another age: move the latent code along an
InterFaceGAN direction (:mod:`.directions`). It cannot do that to a photo without a slow inversion. So, as in
"StyleGAN2 Distillation for Feed-forward Image Manipulation" (Viazovetskyi et al., ECCV 2020):

1. :func:`make_pairs` draws random faces and the same faces moved by a random amount along the gender and the age
   direction: before/after pairs of synthetic people, with the two amounts;
2. :func:`train` fits :class:`~.student.Student`, a U-Net that takes a face and the two amounts and draws the
   change, to those pairs;
3. the app runs only the student: one pass per face and frame.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from .warp import TEMPLATES

SIZE = 256  # the student's aligned face: the "ffhq_512" template at 256 px
GENDER_RANGE = 1.5  # pairs move a face up to 1.5 female-to-male gaps either way
AGE_RANGE = 40.0  # ... and up to 40 years younger or older


@dataclass
class Directions:
    """InterFaceGAN directions in the generator's w space (unit, 512): ``gender`` towards female, ``age`` towards
    older; ``gap``: units from the average male to the average female code; ``years``: years per unit."""

    gender: np.ndarray
    age: np.ndarray
    gap: float
    years: float

    def offset(self, gender: float, years: float) -> np.ndarray:
        """``gender`` in female-to-male gaps (+ towards female), ``years`` older (negative: younger)."""
        return gender * self.gap * self.gender + (years / self.years) * self.age

    def save(self, path) -> None:
        np.savez(path, gender=self.gender, age=self.age, gap=self.gap, years=self.years)

    @classmethod
    def load(cls, path) -> "Directions":
        with np.load(path) as d:
            return cls(d["gender"], d["age"], float(d["gap"]), float(d["years"]))


def ridge(x: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Ridge regression weights (d,) of ``y`` (n,) on ``x`` (n, d), both centred; the penalty is ``alpha`` times
    the average variance of ``x``'s columns, which keeps the direction among the ways faces really vary."""
    x = x - x.mean(0)
    y = y - y.mean()
    gram = x.T @ x
    lam = alpha * np.trace(gram) / gram.shape[0]
    return np.linalg.solve(gram + lam * np.eye(gram.shape[0]), x.T @ y)


def unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-12)


def without(v: np.ndarray, other: np.ndarray) -> np.ndarray:
    """``v`` without its part along ``other``, as a unit vector."""
    other = unit(other)
    return unit(v - (v @ other) * other)


def logit(p: np.ndarray, eps: float = 0.02) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_directions(w: np.ndarray, female: np.ndarray, age: np.ndarray) -> Directions:
    """Conditional InterFaceGAN directions from codes ``w`` (n, 512) and a judge's P(female) and age for them."""
    beta_g, beta_a = ridge(w, logit(female)), ridge(w, age)
    d_g, d_a = without(unit(beta_g), beta_a), without(unit(beta_a), beta_g)
    p = (w - w.mean(0)) @ d_g
    return Directions(d_g, d_a, float(p[female >= 0.5].mean() - p[female < 0.5].mean()), float(beta_a @ d_a))


@torch.inference_mode()
def sample_labels(generator, judge, n: int, seed: int = 0, batch: int = 8, progress=print):
    """``n`` random codes (truncation 0.7) with the ``judge``'s (FairFace) P(female) and age of their faces."""
    from .gfpgan import to_bgr

    torch.manual_seed(seed)
    w_avg = generator.mean_latent(10000)
    lm = TEMPLATES["ffhq_512"] * generator.size
    ws, female, age = [], [], []
    start = time.perf_counter()
    for i in range(0, n, batch):
        w = w_avg + 0.7 * (generator.mapping(torch.randn(batch, 512, device=w_avg.device)) - w_avg)
        for r in judge.read_many(to_bgr(generator.synthesis(w)).astype(np.uint8), lm):
            female.append(r.female)
            age.append(r.age)
        ws.append(w.cpu().numpy())
        if progress and (i // batch) % 25 == 0:
            progress(f"[distill] labelled {i + batch}/{n} faces ({time.perf_counter() - start:.0f} s)")
    return np.concatenate(ws)[:n].astype(np.float64), np.array(female)[:n], np.array(age)[:n]


@torch.inference_mode()
def make_pairs(generator, directions: Directions, n: int, out_dir, seed: int = 1, batch: int = 8,
               progress=print) -> Path:
    """``n`` before/after pairs in ``out_dir``: ``before.npy`` and ``after.npy`` (n, 256, 256, 3, BGR uint8) and
    ``amounts.npy`` (n, 2): the gender amount (gaps, + towards female) and the years. A tenth of the pairs change
    only the gender, a tenth only the age; truncation varies from 0.5 to 1 for variety. Stopped runs resume."""
    from .gfpgan import to_bgr

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    done_file = out / "done.txt"  # how many pairs are complete: a stopped run goes on from there
    done = int(done_file.read_text()) if done_file.is_file() else 0

    def array(name, dtype, shape):
        path = out / name
        return np.lib.format.open_memmap(path, "r+" if done and path.is_file() else "w+", dtype, shape)

    before = array("before.npy", np.uint8, (n, SIZE, SIZE, 3))
    after = array("after.npy", np.uint8, (n, SIZE, SIZE, 3))
    amounts = array("amounts.npy", np.float32, (n, 2))  # replayed for the pairs already made
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    dev = generator.input.input.device
    w_avg = generator.mean_latent(10000)
    start, made = time.perf_counter(), 0
    for i in range(0, n, batch):
        b = min(batch, n - i)
        # every random number is drawn in the same order whether the batch is made or skipped, so a resumed run
        # makes exactly the pairs an uninterrupted one would
        psi = torch.as_tensor(rng.uniform(0.5, 1.0, (b, 1)), dtype=torch.float32, device=dev)
        z = torch.randn(b, 512, device=dev)
        g = rng.uniform(-GENDER_RANGE, GENDER_RANGE, b)
        a = rng.uniform(-AGE_RANGE, AGE_RANGE, b)
        kind = rng.random(b)
        a[kind < 0.1] = 0.0
        g[(kind >= 0.1) & (kind < 0.2)] = 0.0
        amounts[i:i + b] = np.stack([g, a], 1)
        if i + b <= done:
            continue
        w = w_avg + psi * (generator.mapping(z) - w_avg)
        delta = np.stack([directions.offset(gi, ai) for gi, ai in zip(g, a)])
        w2 = w + torch.as_tensor(delta, dtype=torch.float32, device=dev)
        images = to_bgr(generator.synthesis(torch.cat([w, w2]))).astype(np.uint8)
        small = [cv2.resize(x, (SIZE, SIZE), interpolation=cv2.INTER_AREA) for x in images]
        before[i:i + b] = np.stack(small[:b])
        after[i:i + b] = np.stack(small[b:])
        for arr in (before, after, amounts):
            arr.flush()
        done_file.write_text(str(i + b))
        made += b
        if progress and (i // batch) % 25 == 0:
            rate = made / (time.perf_counter() - start)
            progress(f"[distill] {i + b}/{n} pairs ({rate:.1f}/s, {(n - i - b) / rate / 60:.0f} min left)")
    amounts.flush()
    (out / "info.json").write_text(json.dumps({"pairs": n, "size": SIZE, "gender_range": GENDER_RANGE,
                                               "age_range": AGE_RANGE, "seed": seed}))
    return out


# --- training -------------------------------------------------------------------------------------------------

FEATURES = ["/model/model/relu/Relu_output_0", "/model/model/layer1/layer1.2/relu_1/Relu_output_0",
            "/model/model/layer2/layer2.3/relu_1/Relu_output_0", "/model/model/layer3/layer3.5/relu_1/Relu_output_0"]
LOGITS = "/model/model/fc/Gemm_output_0"


def _theta(m: np.ndarray, size_in: int, size_out: int) -> torch.Tensor:
    """``affine_grid``'s theta for the pixel map ``m`` (2, 3) from a ``size_in`` image to a ``size_out`` crop."""
    inv = np.vstack([cv2.invertAffineTransform(m), [0, 0, 1]])  # crop pixels -> image pixels
    to_norm = lambda s: np.array([[2 / s, 0, 1 / s - 1], [0, 2 / s, 1 / s - 1], [0, 0, 1]])  # noqa: E731
    theta = to_norm(size_in) @ inv @ np.linalg.inv(to_norm(size_out))
    return torch.tensor(theta[:2], dtype=torch.float32)


class Critic(torch.nn.Module):
    """FairFace's ResNet-34 as a fixed feature extractor for the training loss: four layers of features (a
    perceptual loss) and its gender, age and race logits, on the ArcFace-aligned 224 px crop of a 256 px face."""

    def __init__(self, fairface_path, device="cpu"):
        super().__init__()
        from .onnx_torch import load
        from .warp import matrix

        self.net = load(fairface_path, device, outputs=FEATURES + [LOGITS])
        for p in self.net.parameters():
            p.requires_grad_(False)
        m = matrix(TEMPLATES["ffhq_512"] * SIZE, "arcface_112_v2", 224)
        self.register_buffer("theta", _theta(m, SIZE, 224))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.to(device)

    def forward(self, x: torch.Tensor):
        """``x``: (B, 3, 256, 256) RGB in [-1, 1] -> (features, logits)."""
        grid = torch.nn.functional.affine_grid(self.theta.expand(len(x), 2, 3), (len(x), 3, 224, 224),
                                               align_corners=False)
        crop = torch.nn.functional.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
        out = self.net(((crop + 1) / 2 - self.mean) / self.std)
        return out[:-1], out[-1]


def _augment(a: torch.Tensor, b: torch.Tensor, gen: torch.Generator):
    """The same random flip, small shift / scale / turn and colour change for a pair (B, 3, H, W)."""
    n = len(a)
    flip = torch.rand(n, generator=gen) < 0.5
    a, b = a.clone(), b.clone()
    a[flip], b[flip] = a[flip].flip(-1), b[flip].flip(-1)
    ang = (torch.rand(n, generator=gen) - 0.5) * np.radians(8)
    sc = 1 + (torch.rand(n, generator=gen) - 0.5) * 0.1
    t = (torch.rand(n, 2, generator=gen) - 0.5) * 0.06
    theta = torch.zeros(n, 2, 3)
    theta[:, 0, 0], theta[:, 0, 1] = sc * torch.cos(ang), -sc * torch.sin(ang)
    theta[:, 1, 0], theta[:, 1, 1] = sc * torch.sin(ang), sc * torch.cos(ang)
    theta[:, :, 2] = t
    grid = torch.nn.functional.affine_grid(theta.to(a.device), a.shape, align_corners=False)
    a = torch.nn.functional.grid_sample(a, grid, padding_mode="reflection", align_corners=False)
    b = torch.nn.functional.grid_sample(b, grid, padding_mode="reflection", align_corners=False)
    gain = (1 + (torch.rand(n, 1, 1, 1, generator=gen) - 0.5) * 0.3).to(a.device)
    bias = ((torch.rand(n, 3, 1, 1, generator=gen) - 0.5) * 0.15).to(a.device)
    return (a * gain + bias).clamp(-1, 1), (b * gain + bias).clamp(-1, 1)


def train(pairs_dir, fairface_path, out_path, steps: int = 4000, batch: int = 8, lr: float = 2e-4, device="cpu",
          start=None, log_every: int = 50, save_every: int = 250, progress=print):
    """Fit a :class:`~.student.Student` to the pairs: L1 on pixels, plus FairFace features (perceptual) and its
    gender / age logits (so the change carries the cues a classifier reads). Saves the student to ``out_path``
    every ``save_every`` steps, with ``<out_path>.resume`` to go on from there if it is interrupted."""
    from .student import Student, load, save

    dev = torch.device(device)
    d = Path(pairs_dir)
    before, after = np.load(d / "before.npy", mmap_mode="r"), np.load(d / "after.npy", mmap_mode="r")
    amounts = np.load(d / "amounts.npy")
    model = load(start, dev).train() if start else Student().to(dev).train()
    critic = Critic(fairface_path, dev).eval()
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
    gen = torch.Generator().manual_seed(0)
    rng = np.random.default_rng(0)
    resume_path = Path(str(out_path) + ".resume")  # everything needed to go on after an interruption
    first = 1
    if resume_path.is_file():
        state = torch.load(resume_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        gen.set_state(state["gen"])
        rng.bit_generator.state = state["rng"]
        first = state["step"] + 1
        if progress:
            progress(f"[train] resuming after step {state['step']}")

    def tensor(x):
        return torch.from_numpy(np.ascontiguousarray(x[..., ::-1])).to(dev).permute(0, 3, 1, 2).float() / 127.5 - 1

    start_t, logs = time.perf_counter(), []
    for step in range(first, steps + 1):
        idx = np.sort(rng.choice(len(before), batch, replace=False))
        x, y = _augment(tensor(before[idx]), tensor(after[idx]), gen)
        g = torch.as_tensor(amounts[idx, 0], device=dev)
        a = torch.as_tensor(amounts[idx, 1], device=dev)
        out = model(x, g, a)
        l1 = (out - y).abs().mean()
        fo, lo = critic(out)
        with torch.no_grad():
            fy, ly = critic(y)
        perc = sum((p - q).abs().mean() / (q.abs().mean() + 1e-6) for p, q in zip(fo, fy)) / len(fo)
        attr = ((lo[:, 7:18] - ly[:, 7:18]) ** 2).mean()
        loss = l1 + 0.5 * perc + 0.05 * attr
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        logs.append([l1.item(), perc.item(), attr.item()])
        if progress and step % log_every == 0:
            m = np.mean(logs[-log_every:], 0)
            rate = (step - first + 1) / (time.perf_counter() - start_t)
            progress(f"[train] step {step}/{steps}: L1 {m[0]:.4f} perceptual {m[1]:.4f} attributes {m[2]:.3f} "
                     f"({rate:.2f} steps/s, {(steps - step) / rate / 60:.0f} min left)")
        if step % save_every == 0 or step == steps:
            save(model, out_path, step=step, pairs=str(d))
            tmp = resume_path.with_name(resume_path.name + ".tmp")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "gen": gen.get_state(),
                        "rng": rng.bit_generator.state, "step": step}, tmp)
            tmp.replace(resume_path)
    return model
