import numpy as np
import pytest
from src.inference.roi_extractor import ROIExtractor

@pytest.fixture(scope="module")
def extractor():
    """Module-scoped so Destrieux atlas is downloaded once per test run."""
    return ROIExtractor()

def test_extract_returns_pfc_and_stc_keys(extractor, synthetic_preds):
    result = extractor.extract(synthetic_preds)
    assert "PFC" in result
    assert "STC" in result

def test_extract_output_shape(extractor, synthetic_preds):
    result = extractor.extract(synthetic_preds)
    n_timesteps = synthetic_preds.shape[0]
    assert result["PFC"].shape == (n_timesteps,)
    assert result["STC"].shape == (n_timesteps,)

def test_pfc_vertices_are_non_empty(extractor):
    assert len(extractor.pfc_verts) > 0

def test_stc_vertices_are_non_empty(extractor):
    assert len(extractor.stc_verts) > 0

def test_vertex_indices_in_valid_range(extractor):
    from src.inference.engine import N_VERTICES
    assert extractor.pfc_verts.max() < N_VERTICES
    assert extractor.stc_verts.max() < N_VERTICES
