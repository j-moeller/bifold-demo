import pytest
import torch
import torch.nn.functional as F

from bifold_morph.stylegan2 import Generator, make_kernel, upfirdn2d


def upfirdn2d_reference(x, kernel, up=1, down=1, pad=(0, 0)):
    """rosinality's ``upfirdn2d_native`` (stylegan2-pytorch, MIT), the CPU fallback of the CUDA op."""
    pad_x0, pad_x1, pad_y0, pad_y1 = pad[0], pad[1], pad[0], pad[1]
    up_x = up_y = up
    down_x = down_y = down
    _, channel, in_h, in_w = x.shape
    x = x.reshape(-1, in_h, in_w, 1)
    _, in_h, in_w, minor = x.shape
    kernel_h, kernel_w = kernel.shape
    out = x.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)
    out = F.pad(out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
    out = out[:, max(-pad_y0, 0):out.shape[1] - max(-pad_y1, 0), max(-pad_x0, 0):out.shape[2] - max(-pad_x1, 0), :]
    out = out.permute(0, 3, 1, 2)
    out = out.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(-1, minor, in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
                      in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1)
    out = out.permute(0, 2, 3, 1)[:, ::down_y, ::down_x, :]
    return out.reshape(-1, channel, out.shape[1], out.shape[2])


@pytest.mark.parametrize("up,down,pad", [(1, 1, (1, 1)), (2, 1, (2, 1)), (1, 2, (1, 1)), (1, 1, (-1, 2)),
                                         (2, 1, (1, 1))])
def test_upfirdn2d_matches_reference(up, down, pad):
    torch.manual_seed(0)
    x = torch.randn(2, 3, 9, 7)
    k = make_kernel([1, 3, 3, 1]) * (4 if up == 2 else 1)
    k = k + torch.randn_like(k) * 0.01  # not symmetric: the flip matters
    assert torch.allclose(upfirdn2d(x, k, up, down, pad), upfirdn2d_reference(x, k, up, down, pad), atol=1e-6)


def test_generator_shapes_and_fixed_noise():
    torch.manual_seed(0)
    g = Generator(size=32).eval()
    assert g.n_latent == 8 and g.num_layers == 7
    w = g.mapping(torch.randn(2, 512))
    with torch.no_grad():
        full = g.synthesis(w)
        again = g.synthesis(w[:, None].repeat(1, g.n_latent, 1))
        small = g.synthesis(w, size=16)
    assert full.shape == (2, 3, 32, 32) and small.shape == (2, 3, 16, 16)
    assert torch.equal(full, again)  # fixed noise: the same code gives the same image
    with pytest.raises(ValueError):
        g.synthesis(w, size=24)
    with pytest.raises(ValueError):
        g.synthesis(w, size=64)


def test_early_stop_is_the_partial_sum():
    """Stopping at a size gives exactly the RGB sum the full generator has added up by that resolution."""
    torch.manual_seed(1)
    g = Generator(size=32).eval()
    w = g.mapping(torch.randn(1, 512))[:, None].repeat(1, g.n_latent, 1)
    with torch.no_grad():
        small = g.synthesis(w, size=16)
        # the 32 px image is the 16 px sum upsampled plus the 32 px layers' own RGB
        out = g.conv1(g.input(w), w[:, 0], noise=g.noises.noise_0)
        skip = g.to_rgb1(out, w[:, 1])
        for i, (c1, c2, rgb) in enumerate(zip(g.convs[::2][:2], g.convs[1::2][:2], g.to_rgbs[:2])):
            j = 1 + 2 * i
            out = c2(c1(out, w[:, j], getattr(g.noises, f"noise_{j}")), w[:, j + 1],
                     getattr(g.noises, f"noise_{j + 1}"))
            skip = rgb(out, w[:, j + 2], skip)
    assert torch.allclose(small, skip, atol=1e-6)


def test_parameter_names_are_rosinalitys():
    keys = set(Generator(size=16).state_dict())
    for key in ("style.1.weight", "style.8.bias", "input.input", "conv1.conv.weight", "conv1.conv.modulation.bias",
                "conv1.noise.weight", "conv1.activate.bias", "to_rgb1.conv.weight", "to_rgb1.bias",
                "convs.0.conv.blur.kernel", "convs.1.activate.bias", "to_rgbs.0.upsample.kernel",
                "noises.noise_0", "noises.noise_4"):
        assert key in keys, key
