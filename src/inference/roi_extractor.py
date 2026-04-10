from __future__ import annotations
import numpy as np
from nilearn import datasets

N_VERTS_PER_HEMI = 10242  # fsaverage5 per hemisphere

PFC_LABEL_FRAGMENTS = [
    "G_front_inf-Opercular",
    "G_front_inf-Orbital",
    "G_front_inf-Triangul",
    "G_front_middle",
    "G_front_sup",
]
STC_LABEL_FRAGMENTS = [
    "G_temp_sup-G_T_transv",
    "G_temp_sup-Lateral",
    "G_temp_sup-Plan_polar",
    "G_temp_sup-Plan_tempo",
]


class ROIExtractor:
    """
    Maps fsaverage5 vertex predictions to mean activation per ROI.
    Downloads Destrieux atlas on first instantiation (cached by nilearn).
    """

    def __init__(self) -> None:
        destrieux = datasets.fetch_atlas_surf_destrieux()
        label_names = [
            lbl.decode() if isinstance(lbl, bytes) else lbl
            for lbl in destrieux.labels
        ]

        pfc_idx = [
            i for i, l in enumerate(label_names)
            if any(frag in l for frag in PFC_LABEL_FRAGMENTS)
        ]
        stc_idx = [
            i for i, l in enumerate(label_names)
            if any(frag in l for frag in STC_LABEL_FRAGMENTS)
        ]

        pfc_lh = np.where(np.isin(destrieux.map_left, pfc_idx))[0]
        pfc_rh = np.where(np.isin(destrieux.map_right, pfc_idx))[0] + N_VERTS_PER_HEMI
        self.pfc_verts: np.ndarray = np.concatenate([pfc_lh, pfc_rh])

        stc_lh = np.where(np.isin(destrieux.map_left, stc_idx))[0]
        stc_rh = np.where(np.isin(destrieux.map_right, stc_idx))[0] + N_VERTS_PER_HEMI
        self.stc_verts: np.ndarray = np.concatenate([stc_lh, stc_rh])

    def extract(self, preds: np.ndarray) -> dict[str, np.ndarray]:
        """
        preds: (n_timesteps, 20484) -> {"PFC": (n_timesteps,), "STC": (n_timesteps,)}
        """
        return {
            "PFC": preds[:, self.pfc_verts].mean(axis=1),
            "STC": preds[:, self.stc_verts].mean(axis=1),
        }
