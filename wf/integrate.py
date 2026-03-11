import math
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

from latch.resources.tasks import custom_task, large_gpu_task
from latch.types.file import LatchFile
from latch.types.directory import LatchDir, LatchOutputDir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class IntegrationInput:
    sample_h5ads: List[LatchFile]
    n_latent: int
    n_layers: int
    max_epochs: int
    n_top_genes: int          # 0 means skip HVG selection
    gene_likelihood: str      # "nb" or "zinb"
    output_dir: LatchDir



def _make_obs_names_unique(h5ad_path: str, sample_name: str) -> None:
    import anndata as ad

    adata = ad.read_h5ad(h5ad_path)
    adata.obs_names = [f"{sample_name}_{idx}" for idx in adata.obs_names]
    adata.write_h5ad(h5ad_path)
    logger.info(f"  [{sample_name}] {adata.n_obs} cells — obs_names rewritten")


def _generate_overview_coordinates(adata, sample_key: str = "sample",
                                   padding_factor: float = 1.1):
    import numpy as np

    samples = adata.obs[sample_key].unique().tolist()
    n = len(samples)
    ncols = math.ceil(math.sqrt(n))

    # Priority: spatial_fov (global coords) > spatial (local) > obs columns
    _OBSM_KEYS = ["spatial_fov", "spatial"]
    _OBS_PAIRS = [
        ("CenterX_global_px", "CenterY_global_px"),
        ("x_centroid", "y_centroid"),
        ("CenterX_local_px", "CenterY_local_px"),
    ]

    def _get_coords(ad_view):
        for key in _OBSM_KEYS:
            if key in ad_view.obsm:
                return ad_view.obsm[key][:, :2].copy()
        for cx, cy in _OBS_PAIRS:
            if cx in ad_view.obs.columns and cy in ad_view.obs.columns:
                return ad_view.obs[[cx, cy]].values.astype(float)
        return np.zeros((ad_view.n_obs, 2))

    sample_coords = {}
    max_w, max_h = 0.0, 0.0
    for s in samples:
        mask = adata.obs[sample_key] == s
        coords = _get_coords(adata[mask])
        sample_coords[s] = (mask, coords)
        if len(coords):
            max_w = max(max_w, coords[:, 0].ptp())
            max_h = max(max_h, coords[:, 1].ptp())

    tile_w = max_w * padding_factor
    tile_h = max_h * padding_factor

    overview = np.zeros((adata.n_obs, 2), dtype=np.float64)
    for i, s in enumerate(samples):
        mask, coords = sample_coords[s]
        mask_arr = mask.values if hasattr(mask, "values") else mask
        if len(coords):
            coords -= coords.min(axis=0)
        coords[:, 0] += (i % ncols) * tile_w
        coords[:, 1] += (i // ncols) * tile_h
        overview[mask_arr] = coords

    adata.obsm["X_overview_spatial"] = overview
    nrows = math.ceil(n / ncols)
    logger.info(
        f"Overview coords: {n} samples → {nrows}×{ncols} grid "
        f"(tile {tile_w:.0f}×{tile_h:.0f})"
    )
    return adata


def _compute_integration_metrics(
    adata,
    batch_key: str = "sample",
    embed_key: str = "X_scVI",
    n_neighbors: int = 30,
    subsample_n: int = 50_000,
) -> dict:
    import numpy as np
    import pandas as pd
    import scanpy as sc
    from sklearn.metrics import silhouette_samples

    n_batches = adata.obs[batch_key].nunique()
    if n_batches < 2:
        logger.info("  Single sample — skipping batch-mixing metrics")
        return {"note": "single_sample_no_batch_metrics"}

    if adata.n_obs > subsample_n:
        logger.info(f"  Subsampling {subsample_n} cells for metric computation")
        rng = np.random.default_rng(42)
        idx = rng.choice(adata.n_obs, subsample_n, replace=False)
        adata_sub = adata[idx].copy()
    else:
        adata_sub = adata.copy()

    X = adata_sub.obsm[embed_key]
    batch_labels = adata_sub.obs[batch_key].values

    metrics = {}

    logger.info("  Computing batch silhouette (ASW) …")
    sil_per_cell = silhouette_samples(X, batch_labels, metric="euclidean")
    # Perfect mixing → silhouette ≈ 0; perfect separation → ≈ 1
    # Rescale: batch_ASW = 1 − |mean_sil|  so 1 is best
    raw_mean = float(np.mean(sil_per_cell))
    metrics["batch_ASW"] = round(1.0 - abs(raw_mean), 4)
    metrics["batch_ASW_raw_mean"] = round(raw_mean, 4)

    # Per-sample breakdown
    per_sample = {}
    for s in adata_sub.obs[batch_key].unique():
        mask = batch_labels == s
        per_sample[s] = round(float(np.mean(np.abs(sil_per_cell[mask]))), 4)
    metrics["batch_ASW_per_sample"] = per_sample

    logger.info("  Computing entropy of batch mixing …")
    sc.pp.neighbors(adata_sub, use_rep=embed_key, n_neighbors=n_neighbors)
    conn = adata_sub.obsp["connectivities"]

    max_entropy = np.log(n_batches)
    batch_codes = pd.Categorical(batch_labels).codes
    n_cats = n_batches

    entropies = np.zeros(adata_sub.n_obs)
    for i in range(adata_sub.n_obs):
        row = conn[i]
        # Get neighbor indices
        if hasattr(row, "indices"):
            nbrs = row.indices
        else:
            nbrs = np.nonzero(row)[0]
        if len(nbrs) == 0:
            continue
        nbr_batches = batch_codes[nbrs]
        counts = np.bincount(nbr_batches, minlength=n_cats).astype(float)
        counts = counts[counts > 0]
        probs = counts / counts.sum()
        entropies[i] = -np.sum(probs * np.log(probs))

    metrics["entropy_of_batch_mixing"] = round(float(np.mean(entropies) / max_entropy), 4)

    logger.info("  Computing kNN batch purity …")
    purities = np.zeros(adata_sub.n_obs)
    for i in range(adata_sub.n_obs):
        row = conn[i]
        if hasattr(row, "indices"):
            nbrs = row.indices
        else:
            nbrs = np.nonzero(row)[0]
        if len(nbrs) == 0:
            purities[i] = 1.0
            continue
        same_batch = (batch_codes[nbrs] == batch_codes[i]).sum()
        purities[i] = same_batch / len(nbrs)

    metrics["knn_batch_purity"] = round(float(np.mean(purities)), 4)
    metrics["knn_batch_mixing"] = round(1.0 - float(np.mean(purities)), 4)

    logger.info("  Computing graph connectivity …")
    try:
        import scipy.sparse as sp
        from scipy.sparse.csgraph import connected_components

        adj = (conn > 0).astype(int)
        gc_scores = []
        for s in adata_sub.obs[batch_key].unique():
            mask = (batch_labels == s)
            idx_s = np.where(mask)[0]
            sub_adj = adj[np.ix_(idx_s, idx_s)]
            n_comp, labels_cc = connected_components(sub_adj, directed=False)
            largest_cc = np.bincount(labels_cc).max()
            gc_scores.append(largest_cc / len(idx_s))
        metrics["graph_connectivity"] = round(float(np.mean(gc_scores)), 4)
    except Exception as e:
        logger.warning(f"  Graph connectivity failed: {e}")
        metrics["graph_connectivity"] = None

    return metrics


@large_gpu_task
def scvi_integration_task(input: IntegrationInput) -> LatchOutputDir:
    import anndata as ad
    ad.settings.allow_write_nullable_strings = True
    import scanpy as sc
    import scvi
    import numpy as np
    import pandas as pd
    import json

    work = Path("/root/scvi_work")
    work.mkdir(parents=True, exist_ok=True)
    local_out = Path("/root/scvi_output")
    local_out.mkdir(parents=True, exist_ok=True)

    n_samples = len(input.sample_h5ads)
    is_multisample = n_samples > 1

    logger.info("=" * 60)
    logger.info(f"STEP 1  Download & rewrite obs_names ({n_samples} sample(s))")
    logger.info("=" * 60)

    local_paths = {}   # sample_name → local h5ad path
    for lf in input.sample_h5ads:
        lp = Path(lf.local_path)
        sample_name = lp.stem
        _make_obs_names_unique(str(lp), sample_name)
        local_paths[sample_name] = str(lp)

    logger.info("=" * 60)
    logger.info("STEP 2  On-disk concat (anndata.experimental.concat_on_disk)")
    logger.info("=" * 60)

    merged_path = str(work / "merged.h5ad")

    if is_multisample:
        # _make_obs_names_unique already wrote obs["sample"] into each file,
        # so we don't pass label= here to avoid conflict.  The column is
        # already present and consistent with the dict keys.
        ad.experimental.concat_on_disk(
            in_files=local_paths,
            out_file=merged_path,
            join="inner",
            merge="same",
        )
    else:
        # Single sample — just copy the one file
        import shutil
        only_path = list(local_paths.values())[0]
        shutil.copy2(only_path, merged_path)

    logger.info(f"  Written → {merged_path}")

    logger.info("=" * 60)
    logger.info("STEP 3  Load merged & generate overview coordinates")
    logger.info("=" * 60)

    adata = ad.read_h5ad(merged_path)
    logger.info(f"  {adata.n_obs} cells × {adata.n_vars} genes")
    logger.info(f"  Samples : {adata.obs['sample'].nunique()}")
    logger.info(f"  Layers  : {list(adata.layers.keys())}")

    if is_multisample:
        adata = _generate_overview_coordinates(adata, sample_key="sample")
    else:
        logger.info("  Single sample — skipping overview coordinate layout")

    logger.info("=" * 60)
    logger.info("STEP 4  scVI integration")
    logger.info("=" * 60)

    # Resolve where raw counts live
    count_layer = None  # type: str | None
    if "counts" in adata.layers:
        count_layer = "counts"
        logger.info("  Count source: adata.layers['counts']")
    else:
        logger.info("  Count source: adata.X (assuming raw counts)")

    # Optional HVG selection
    if input.n_top_genes > 0:
        logger.info(f"  Selecting {input.n_top_genes} HVGs …")
        tmp = adata.copy()
        if count_layer:
            tmp.X = tmp.layers[count_layer].copy()
        sc.pp.normalize_total(tmp, target_sum=1e4)
        sc.pp.log1p(tmp)
        hvg_kwargs = dict(
            n_top_genes=input.n_top_genes,
            flavor="seurat",
            subset=False,
        )
        if is_multisample:
            hvg_kwargs["batch_key"] = "sample"
        sc.pp.highly_variable_genes(tmp, **hvg_kwargs)
        hvg = tmp.var["highly_variable"]
        adata = adata[:, hvg].copy()
        del tmp
        logger.info(f"  After HVG: {adata.n_vars} genes")

    # Setup & train
    scvi.settings.seed = 42

    # batch_key is always "sample" — for single sample this creates a
    # single-category batch which is fine (scVI handles it gracefully)
    scvi.model.SCVI.setup_anndata(
        adata,
        layer=count_layer,
        batch_key="sample",
    )

    model = scvi.model.SCVI(
        adata,
        n_latent=input.n_latent,
        n_layers=input.n_layers,
        gene_likelihood=input.gene_likelihood,
    )
    logger.info(f"  Model: {model}")
    logger.info(f"  Training (max {input.max_epochs} epochs, early stopping) …")

    model.train(
        max_epochs=input.max_epochs,
        early_stopping=True,
        early_stopping_patience=20,
        early_stopping_monitor="elbo_validation",
        check_val_every_n_epoch=1,
        batch_size=256,
    )
    logger.info("  Training complete")

    # Latent representation
    adata.obsm["X_scVI"] = model.get_latent_representation()
    logger.info(f"  X_scVI shape: {adata.obsm['X_scVI'].shape}")

    logger.info("=" * 60)
    logger.info("STEP 5  Integration metrics")
    logger.info("=" * 60)

    metrics = _compute_integration_metrics(
        adata,
        batch_key="sample",
        embed_key="X_scVI",
    )

    # Log headline numbers
    for k, v in metrics.items():
        if not isinstance(v, dict):
            logger.info(f"  {k}: {v}")

    # Save metrics
    metrics_path = local_out / "integration_metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2, default=str)
    logger.info(f"  → integration_metrics.json")

    # Also save a flat CSV for easy consumption
    flat = {k: v for k, v in metrics.items() if not isinstance(v, dict)}
    pd.DataFrame([flat]).to_csv(
        local_out / "integration_metrics.csv", index=False
    )
    logger.info(f"  → integration_metrics.csv")

    # If per-sample breakdown exists, save separately
    if "batch_ASW_per_sample" in metrics and isinstance(metrics["batch_ASW_per_sample"], dict):
        pd.DataFrame(
            list(metrics["batch_ASW_per_sample"].items()),
            columns=["sample", "abs_silhouette"],
        ).to_csv(local_out / "batch_asw_per_sample.csv", index=False)
        logger.info(f"  → batch_asw_per_sample.csv")

    logger.info("=" * 60)
    logger.info("STEP 6  Save outputs")
    logger.info("=" * 60)

    # Integrated h5ad
    adata.write_h5ad(str(local_out / "integrated_scvi.h5ad"))
    logger.info(f"  → integrated_scvi.h5ad")

    # scVI model directory
    model.save(str(local_out / "scvi_model"), overwrite=True)
    logger.info(f"  → scvi_model/")

    # Training history
    hist = model.history
    if hist is not None:
        rows = {k: v.values.flatten() for k, v in hist.items() if hasattr(v, "values")}
        pd.DataFrame(rows).to_csv(str(local_out / "training_history.csv"), index=True)
        logger.info(f"  → training_history.csv")

    return LatchDir(
        str(local_out),
        input.output_dir.remote_directory,
    )