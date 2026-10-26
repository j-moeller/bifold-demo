import cv2
import numpy as np
import torch

from bifold_morph import student as student_mod
from bifold_morph.distill import Directions, _augment, _theta, fit_directions, unit, without


def test_student_starts_as_the_identity_and_round_trips(tmp_path):
    torch.manual_seed(0)
    model = student_mod.Student(widths=(16, 32, 32))
    x = torch.rand(2, 3, 64, 64) * 2 - 1
    with torch.no_grad():
        y = model(x, torch.tensor([1.0, -1.0]), torch.tensor([30.0, 0.0]))
    assert y.shape == x.shape and torch.allclose(y, x)  # zero-initialised head: no change yet
    for p in model.parameters():
        p.data += torch.randn_like(p) * 0.01
    student_mod.save(model, tmp_path / "s.pt")
    back = student_mod.load(tmp_path / "s.pt")
    with torch.no_grad():
        a = model(x, torch.tensor([0.5, 0.5]), torch.tensor([10.0, -10.0]))
        b = back(x, torch.tensor([0.5, 0.5]), torch.tensor([10.0, -10.0]))
    assert torch.allclose(a, b, atol=2e-2)  # stored as 16-bit


def test_directions_fit_and_conditional():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(3000, 512)) * np.linspace(0.3, 2.0, 512)
    u, v = unit(rng.normal(size=512)), unit(rng.normal(size=512))
    female = 1 / (1 + np.exp(-3 * (w @ v)))
    age = 35 + 9 * (w @ u)
    d = fit_directions(w, female, age)
    assert d.gender @ v > 0.5 and d.age @ u > 0.5
    assert d.gap > 0 and d.years > 1
    assert np.allclose(d.offset(1.0, 0), d.gap * d.gender)
    assert np.allclose(without(np.array([1.0, 1.0]), np.array([1.0, 0.0])), [0, 1])


def test_directions_round_trip(tmp_path):
    d = Directions(unit(np.ones(512)), unit(np.arange(512.0)), 2.4, 10.7)
    d.save(tmp_path / "d.npz")
    back = Directions.load(tmp_path / "d.npz")
    assert np.allclose(back.gender, d.gender) and back.gap == 2.4 and back.years == 10.7


def test_theta_matches_opencv():
    """The differentiable crop of the training loss is the same crop OpenCV cuts."""
    rng = np.random.default_rng(2)
    img = cv2.GaussianBlur(rng.random((256, 256, 3)).astype(np.float32), (0, 0), 4)
    m = np.array([[0.8, 0.1, 10.0], [-0.1, 0.8, 20.0]])
    ref = cv2.warpAffine(img, m, (224, 224), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    x = torch.from_numpy(img).permute(2, 0, 1)[None]
    grid = torch.nn.functional.affine_grid(_theta(m, 256, 224)[None], (1, 3, 224, 224), align_corners=False)
    out = torch.nn.functional.grid_sample(x, grid, padding_mode="border", align_corners=False)[0].permute(1, 2, 0)
    assert np.abs(out.numpy() - ref)[8:-8, 8:-8].max() < 1e-3


def test_augment_moves_both_alike():
    a = torch.rand(4, 3, 32, 32) * 2 - 1
    x, y = _augment(a, a.clone(), torch.Generator().manual_seed(0))
    assert x.shape == a.shape and torch.allclose(x, y)
