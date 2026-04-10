from __future__ import annotations
import io
import numpy as np
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")  # headless — must be set before pyplot import
import matplotlib.pyplot as plt
from nilearn import plotting, datasets


class BrainAnimator:
    """
    Renders per-second brain activation frames from fsaverage5 predictions
    and assembles them into an animated GIF.
    """

    def __init__(self) -> None:
        fsaverage5 = datasets.fetch_surf_fsaverage("fsaverage5")
        self._meshes = {
            "left":  (fsaverage5.infl_left,  fsaverage5.sulc_left),
            "right": (fsaverage5.infl_right, fsaverage5.sulc_right),
        }

    def animate(
        self,
        preds: np.ndarray,
        output_path: str,
        fps: int = 1,
    ) -> str:
        """
        preds: (n_timesteps, 20484)
        Saves GIF to output_path and returns output_path.
        """
        vmax = float(np.percentile(np.abs(preds), 95)) or 1.0
        frames = [self._render_frame(preds[t], vmax) for t in range(preds.shape[0])]
        duration_ms = int(1000 / fps)
        imageio.mimsave(output_path, frames, duration=duration_ms, loop=0)
        return output_path

    def _render_frame(self, preds_t: np.ndarray, vmax: float) -> np.ndarray:
        """Render left + right hemisphere side-by-side, return (H, W, C) array."""
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), subplot_kw={"projection": "3d"})
        for ax, (hemi, (mesh, bg)) in zip(axes, self._meshes.items()):
            stat_map = preds_t[:10242] if hemi == "left" else preds_t[10242:]
            plotting.plot_surf_stat_map(
                surf_mesh=mesh,
                stat_map=stat_map,
                bg_map=bg,
                hemi=hemi,
                view="lateral",
                vmax=vmax,
                colorbar=False,
                axes=ax,
            )
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=72, bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)
        return imageio.imread(buf)
