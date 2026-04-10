import os
import numpy as np
import pytest
from src.inference.brain_animator import BrainAnimator

@pytest.fixture(scope="module")
def animator():
    return BrainAnimator()

def test_animate_creates_gif_file(animator, tmp_path, synthetic_preds):
    # Use only first 5 frames to keep test fast
    output = str(tmp_path / "test_brain.gif")
    animator.animate(synthetic_preds[:5], output_path=output)
    assert os.path.exists(output)
    assert os.path.getsize(output) > 0

def test_animate_gif_is_valid(animator, tmp_path, synthetic_preds):
    import imageio
    output = str(tmp_path / "test_brain_valid.gif")
    animator.animate(synthetic_preds[:3], output_path=output)
    reader = imageio.get_reader(output)
    frames = list(reader)
    assert len(frames) == 3
    assert frames[0].ndim == 3  # (H, W, C)
