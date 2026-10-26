"""Which way in StyleGAN2's latent space a face gets older, and which way it gets more female or male.

The method is InterFaceGAN's (Shen et al., "Interpreting the Latent Space of GANs for Semantic Face Editing", 2020),
with MiVOLO as the judge:

1. draw a few thousand random faces with the generator and let MiVOLO say how old each looks and whether it looks
   male or female;
2. fit a linear model of those labels on the faces' 512-number codes ``w`` (ridge regression): its weights point
   the way the label grows, and that is the direction;
3. make each direction conditional: take out of it the part along the other one, so a step along "age" does not
   change the gender MiVOLO's linear model reads, and a step along "gender" does not change the age;
4. calibrate on faces held back from the fit: step their codes along "age" and measure with MiVOLO how many years
   one step adds; for gender, measure the distance between the average female and the average male code.

The result is small (two vectors and two numbers) and is saved next to the models, see :func:`Directions.save`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

from .e4e import to_images
from .mivolo import crop_box

AGE_END = 40.0  # years: the age slider goes from -AGE_END to +AGE_END
GENDER_END = 1.25  # the gender slider's ends move a face this many female-to-male gaps
TRUNCATION = 0.7  # random faces are drawn this close to the average face (the usual "truncation trick")


@dataclass
class Directions:
    """Unit directions in ``w`` space (512,): ``age`` towards older, ``gender`` towards male;
    ``years_per_unit``: years one unit along ``age`` adds; ``gender_gap``: units from the average female code to
    the average male code along ``gender``."""

    age: np.ndarray
    gender: np.ndarray
    years_per_unit: float
    gender_gap: float
    info: dict = field(default_factory=dict)

    def offset(self, years: float, gender: float) -> np.ndarray:
        """The change of ``w`` (512,) for the sliders: ``years`` older (negative: younger) and ``gender`` from -1
        (towards female, by :data:`GENDER_END` gaps) to +1 (towards male)."""
        return (years / self.years_per_unit) * self.age + (gender * GENDER_END * self.gender_gap) * self.gender

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez(tmp, age=self.age.astype(np.float32), gender=self.gender.astype(np.float32),
                 years_per_unit=self.years_per_unit, gender_gap=self.gender_gap, info=json.dumps(self.info))
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path) -> "Directions":
        with np.load(path) as data:
            return cls(data["age"].astype(np.float64), data["gender"].astype(np.float64),
                       float(data["years_per_unit"]), float(data["gender_gap"]), json.loads(str(data["info"])))


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


def fit(w: np.ndarray, age: np.ndarray, male: np.ndarray, alpha: float = 1.0) -> Directions:
    """Conditional age and gender directions from codes ``w`` (n, 512) and MiVOLO's labels (``age`` in years,
    ``male`` probabilities); ``years_per_unit`` is the linear model's until :func:`calibrate` measures it."""
    w, age, male = np.asarray(w, np.float64), np.asarray(age, np.float64), np.asarray(male, np.float64)
    is_male = male >= 0.5
    if min(is_male.sum(), (~is_male).sum()) < 10:
        raise ValueError(f"need at least 10 faces of each gender, got {is_male.sum()} male, {(~is_male).sum()} female")
    beta_age = ridge(w, age, alpha)
    beta_gender = ridge(w, logit(male), alpha)
    d_age = without(unit(beta_age), beta_gender)
    d_gender = without(unit(beta_gender), beta_age)
    proj = (w - w.mean(0)) @ d_gender
    gap = float(proj[is_male].mean() - proj[~is_male].mean())
    pred_age = (w - w.mean(0)) @ beta_age + age.mean()
    r2 = 1 - float(((age - pred_age) ** 2).sum() / max(((age - age.mean()) ** 2).sum(), 1e-12))
    pred_male = (w - w.mean(0)) @ beta_gender + logit(male).mean() > 0
    info = {"samples": int(len(w)), "male": int(is_male.sum()), "female": int((~is_male).sum()),
            "age_mean": float(age.mean()), "age_std": float(age.std()), "age_r2": r2,
            "gender_accuracy": float((pred_male == is_male).mean()), "age_gender_cos": float(
                unit(beta_age) @ unit(beta_gender))}
    return Directions(d_age, d_gender, float(beta_age @ d_age), gap, info)


# --- drawing and labelling faces --------------------------------------------------------------------------------

class FaceLab:
    """Draws faces from codes and labels them: the generator (an :class:`~.e4e.E4E`'s), YuNet and MiVOLO."""

    def __init__(self, e4e, mivolo, detector, device, size: int = 1024, batch: int = 4):
        self.e4e, self.mivolo, self.detector = e4e, mivolo, detector
        self.device, self.size, self.batch = torch.device(device), size, batch

    @torch.inference_mode()
    def random_w(self, n: int, seed: int, truncation: float = TRUNCATION) -> np.ndarray:
        g = torch.Generator().manual_seed(seed)
        z = torch.randn(n, 512, generator=g).to(self.device)
        w_avg = self.e4e.w_avg
        ws = [w_avg + truncation * (self.e4e.generator.mapping(z[i:i + 256]) - w_avg) for i in range(0, n, 256)]
        return torch.cat(ws).cpu().numpy().astype(np.float64)

    @torch.inference_mode()
    def images(self, w: np.ndarray, out: int = 512) -> list[np.ndarray]:
        """BGR uint8 faces (``out`` px) for codes ``w`` (n, 512) or (n, 18, 512)."""
        result = []
        for i in range(0, len(w), self.batch):
            codes = torch.as_tensor(w[i:i + self.batch], dtype=torch.float32, device=self.device)
            for rgb in to_images(self.e4e.generator.synthesis(codes, self.size)):
                if rgb.shape[0] != out:
                    rgb = cv2.resize(rgb, (out, out), interpolation=cv2.INTER_AREA)
                result.append(np.ascontiguousarray(rgb[..., ::-1]))
        return result

    def label(self, images: list[np.ndarray]):
        """MiVOLO's :class:`~.mivolo.AgeGender` for the biggest face in each image (``None`` where none is found)."""
        crops, where = [], []
        for i, img in enumerate(images):
            faces = self.detector.detect(img)
            face = crop_box(img, faces[0].box) if faces else None
            if face is not None:
                crops.append(face)
                where.append(i)
        labels = [None] * len(images)
        for i, result in zip(where, self.mivolo.predict(crops) if crops else []):
            labels[i] = result
        return labels

    def sample(self, n: int, seed: int = 0, progress=print) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``n`` random codes (fewer where no face was found) with MiVOLO's ages and male probabilities."""
        w = self.random_w(n, seed)
        keep, ages, males = [], [], []
        start = time.perf_counter()
        step = max(self.batch, 8)
        for i in range(0, n, step):
            for j, label in enumerate(self.label(self.images(w[i:i + step])), i):
                if label is not None:
                    keep.append(j)
                    ages.append(label.age)
                    males.append(label.male)
            done = min(i + step, n)
            if progress and (done % 100 < step or done == n):
                rate = done / (time.perf_counter() - start)
                progress(f"[directions] {done}/{n} faces drawn and labelled ({rate:.1f}/s, "
                         f"{(n - done) / rate / 60:.1f} min left)")
        return w[keep], np.array(ages), np.array(males)


def calibrate(directions: Directions, lab: FaceLab, w: np.ndarray, step: float = 2.0, progress=print) -> Directions:
    """Measure on held-back codes ``w`` what the directions do: years per unit along ``age`` (the median of MiVOLO's
    age difference between ``w + step`` and ``w - step`` units), and how often the gender slider's ends turn a face
    to the other gender. Returns the directions with the measured ``years_per_unit``."""
    older = lab.label(lab.images(w + step * directions.age))
    younger = lab.label(lab.images(w - step * directions.age))
    slopes = [(o.age - y.age) / (2 * step) for o, y in zip(older, younger) if o is not None and y is not None]
    info = dict(directions.info)
    years = directions.years_per_unit
    if len(slopes) >= max(5, len(w) // 4):
        years = float(np.median(slopes))
        info["years_per_unit_linear"] = directions.years_per_unit
    info["years_per_unit_measured"] = years
    if years < 1.0:
        raise ValueError(f"the age direction adds only {years:.2f} years per unit: the labels look broken")
    base = lab.label(lab.images(w))
    shift = GENDER_END * directions.gender_gap * directions.gender
    to_female = lab.label(lab.images(w - shift))
    to_male = lab.label(lab.images(w + shift))
    flips_f = [t.male < 0.5 for b, t in zip(base, to_female) if b is not None and t is not None and b.male >= 0.5]
    flips_m = [t.male >= 0.5 for b, t in zip(base, to_male) if b is not None and t is not None and b.male < 0.5]
    info["male_to_female"] = float(np.mean(flips_f)) if flips_f else None
    info["female_to_male"] = float(np.mean(flips_m)) if flips_m else None
    if progress:
        f = lambda v: "n/a" if v is None else f"{v:.0%}"  # noqa: E731
        progress(f"[directions] one unit along 'age' adds {years:.1f} years; at the gender slider's ends "
                 f"{f(info['male_to_female'])} of the men look female and {f(info['female_to_male'])} of the women "
                 "look male to MiVOLO")
    return Directions(directions.age, directions.gender, years, directions.gender_gap, info)


def compute(lab: FaceLab, samples: int = 3000, held_out: int = 48, seed: int = 0, progress=print) -> Directions:
    """Draw and label ``samples`` faces, fit the directions, and calibrate them on ``held_out`` more."""
    w, age, male = lab.sample(samples, seed, progress)
    if progress:
        progress(f"[directions] {len(w)} of {samples} faces labelled: mean age {age.mean():.0f} "
                 f"(sd {age.std():.0f}), {np.mean(male >= 0.5):.0%} male")
    directions = fit(w, age, male)
    if progress:
        i = directions.info
        progress(f"[directions] linear models: age R² {i['age_r2']:.2f}, gender accuracy {i['gender_accuracy']:.1%}; "
                 f"female-to-male gap {directions.gender_gap:.2f} units")
    directions.info.update(seed=seed, truncation=TRUNCATION, size=lab.size)
    return calibrate(directions, lab, lab.random_w(held_out, seed + 1), progress=progress)


def preview(lab: FaceLab, directions: Directions, faces: int = 4, seed: int = 7, side: int = 256) -> np.ndarray:
    """A grid (BGR) of ``faces`` random faces: younger, as drawn, older, towards female, towards male."""
    w = lab.random_w(faces, seed)
    columns = [(f"{-AGE_END:+.0f} years", directions.offset(-AGE_END, 0)), ("as drawn", np.zeros(512)),
               (f"{AGE_END:+.0f} years", directions.offset(AGE_END, 0)), ("female", directions.offset(0, -1)),
               ("male", directions.offset(0, 1))]
    header = 36
    grid = np.full((header + faces * side, len(columns) * side, 3), 255, np.uint8)
    for c, (title, delta) in enumerate(columns):
        cv2.putText(grid, title, (c * side + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)
        for r, img in enumerate(lab.images(w + delta, out=side)):
            grid[header + r * side:header + (r + 1) * side, c * side:(c + 1) * side] = img
    return grid
