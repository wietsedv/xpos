import os
import sys
from functools import partial
from hashlib import md5
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics.pairwise import pairwise_distances_chunked
from tqdm import tqdm
from transformers.modeling_utils import PreTrainedModel

MARGIN_FN: Dict[str, Callable] = {"ratio": lambda a, b: a / b}


def _get_cache(key, cache_dir: Optional[Path], verbose: bool = False, **kwargs):
    if not cache_dir:
        return None, None
    os.makedirs(cache_dir, exist_ok=True)
    suffix = "_".join([f"{k}-{v}" for k, v in kwargs.items()])
    if len(suffix) > 200:
        suffix = md5(suffix.encode("utf-8")).hexdigest()
    path = cache_dir / f"{key}_{suffix}.npy"
    if path.exists():
        if verbose:
            print("Reusing cached", path, file=sys.stderr)
        return path, np.load(path)
    print("Not yet cached:", path, file=sys.stderr)
    return path, None


def _featurize_batches(
    model: PreTrainedModel,
    data_collator: Callable,
    dataset: Dataset,
    layers: List[int],
    batch_size: int,
):
    batches = list(range(0, dataset.num_rows, batch_size))
    for i in tqdm(batches, dynamic_ncols=True):
        batch = {k: t.to(model.device)
                 for k, t in data_collator(dataset[i:i + batch_size]).items() if k != "labels"}  # type: ignore
        batch_mask: torch.Tensor = batch["attention_mask"].unsqueeze(2).float()

        batch_layers_states: Dict[int, torch.Tensor]
        if layers == [-1]:
            batch_layers_states = {-1: model(**batch).last_hidden_state}
        else:
            hidden_states = model(**batch, output_hidden_states=True).hidden_states
            batch_layers_states = {l: hidden_states[l] for l in layers}

        batch_layers_feats: Dict[int, np.ndarray] = {
            l: ((batch_layers_states[l] * batch_mask).sum(dim=1) / batch_mask.sum(dim=1)).cpu().numpy()
            for l in layers
        }
        yield i, batch_layers_feats


@torch.inference_mode()
def featurize(
    model_fn: Callable[[], PreTrainedModel],
    data_collator: Callable,
    dataset: Dataset,
    layer: Union[int, List[int]] = -1,
    normalize: bool = True,
    batch_size: int = 8,
    cache_dir: Optional[Path] = None,
    cache_key: Optional[str] = None,
    verbose: bool = False,
) -> Union[np.ndarray, Dict[int, np.ndarray]]:
    model = None
    if layer == -1:
        model = model_fn()
        layer = model.config.num_hidden_layers  # type: ignore

    layers = [layer] if isinstance(layer, int) else layer

    get_cache = partial(_get_cache, cache_dir=cache_dir, verbose=verbose)
    cache_key = "features" if cache_key is None else f"{cache_key}_features"

    layers_paths: Dict[int, Optional[Path]] = {}
    layers_feats: Dict[int, np.ndarray] = {}

    for l in layers:
        layers_paths[l], layer_feats = get_cache(cache_key, layer=l)
        if layer_feats is not None:
            layers_feats[l] = layer_feats

    if len(layers_feats) < len(layers):
        layers_feats = {}

        if model is None:
            model = model_fn()

        for i, batch_layers_feats in _featurize_batches(model, data_collator, dataset, layers, batch_size):
            for l, batch_layer_feats in batch_layers_feats.items():
                if l not in layers_feats:
                    layers_feats[l] = np.empty((dataset.num_rows, batch_layer_feats.shape[1]),
                                               dtype=batch_layer_feats.dtype)
                layers_feats[l][i:i + batch_layer_feats.shape[0]] = batch_layer_feats

        for l in layers:
            if layers_paths[l] is not None:
                np.save(layers_paths[l], layers_feats[l])
                not verbose or print(f"Cached to {layers_paths[l]}", file=sys.stderr)

    if normalize:
        for l in layers:
            layers_feats[l] = (layers_feats[l] - layers_feats[l].mean(axis=0)) / layers_feats[l].std(axis=0)

    out = layers_feats[layer] if isinstance(layer, int) else layers_feats
    if "labels" in dataset:
        return out
    return out


def _knn_reduce(chunk: np.ndarray, _: int, k: int):
    indices = np.argpartition(chunk, k, axis=1)[:, :k]

    values = np.empty_like(indices, dtype=chunk.dtype)
    for i in range(indices.shape[0]):
        values[i] = chunk[i, indices[i]]

    return indices, values


def knn(
    X: np.ndarray,
    Y: np.ndarray,
    k: int,
    metric: Union[Callable, str] = "cosine",
) -> Tuple[np.ndarray, np.ndarray]:
    reduce_func = partial(_knn_reduce, k=k)
    indices = np.empty((X.shape[0], k), dtype=np.int32)
    distances = np.empty((X.shape[0], k), dtype=np.float32)

    i = 0
    with tqdm(total=X.shape[0], dynamic_ncols=True) as pbar:
        for chunk_indices, chunk_distances in pairwise_distances_chunked(X, Y, reduce_func=reduce_func, metric=metric):
            n = chunk_indices.shape[0]
            indices[i:i + n] = chunk_indices
            distances[i:i + n] = chunk_distances
            pbar.update(n)
            i += n

    return np.copy(indices), np.copy(distances)


def _cached_knn(
    X: np.ndarray,
    Y: np.ndarray,
    k: int,
    cache_fn: Callable,
    cache_key: str,
    verbose: bool = False,
    **knn_kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    indices_path, indices = cache_fn(f"{cache_key}_indices", k=k, **knn_kwargs)
    distances_path, distances = cache_fn(f"{cache_key}_distances", k=k, **knn_kwargs)
    if indices is None or distances is None:
        indices, distances = knn(X, Y, k, **knn_kwargs)
        if indices_path:
            np.save(indices_path, indices)
            if verbose:
                print("Cached to", indices_path, file=sys.stderr)
        if distances_path:
            np.save(distances_path, distances)
            if verbose:
                print("Cached to", distances_path, file=sys.stderr)
    return indices, distances


def _margin_score(
    x2y_indices: np.ndarray,
    x2y_distances: np.ndarray,
    x2y_kmean: np.ndarray,
    y2x_kmean: np.ndarray,
    margin: Callable,
):
    x_idx = np.arange(x2y_indices.shape[0])

    x2y_mdistances = np.empty_like(x2y_distances)
    for x_i in range(x2y_mdistances.shape[0]):
        for k_i in range(x2y_mdistances.shape[1]):
            y_i = x2y_indices[x_i, k_i]
            d = x2y_distances[x_i, k_i]
            x2y_mdistances[x_i, k_i] = margin(d, (x2y_kmean[x_i] + y2x_kmean[y_i]))

    best_margin_indices = x2y_mdistances.argmin(axis=1)
    best_indices = x2y_indices[x_idx, best_margin_indices]
    best_mdistances = x2y_mdistances[x_idx, best_margin_indices]
    return best_indices, best_mdistances


def search_pairs(
    X: np.ndarray,
    Y: np.ndarray,
    metric: str = "cosine",
    margin: Union[None, Callable, str] = None,
    margin_k: int = 4,
    # bidirectional: bool = False,
    cache_dir: Optional[Path] = None,
    verbose: bool = False,
):
    echo = partial(print, file=sys.stderr) if verbose else lambda *_: _

    if isinstance(margin, str):
        margin = MARGIN_FN[margin]
    if margin is None:
        margin_k = 1

    get_cache = partial(_get_cache, cache_dir=cache_dir, verbose=verbose)
    cached_knn = partial(_cached_knn, k=margin_k, metric=metric, cache_fn=get_cache, verbose=verbose)

    echo("Preparing x2y index")
    x2y_indices, x2y_distances = cached_knn(X, Y, cache_key="x2y")
    x2y_kmean: np.ndarray = x2y_distances.mean(axis=1)
    assert x2y_indices.shape == (X.shape[0], margin_k)

    if margin is None:
        assert x2y_indices.shape == (X.shape[0], 1)
        return x2y_indices[:, 0], x2y_distances[:, 0]

    echo("Preparing y2x index")
    y2x_indices, y2x_distances = cached_knn(Y, X, cache_key="y2x")
    y2x_kmean: np.ndarray = y2x_distances.mean(axis=1)
    assert y2x_indices.shape == (Y.shape[0], margin_k)

    echo("Calculating x2y margin distances")
    best_x2y_indices, best_x2y_mdistances = _margin_score(x2y_indices, x2y_distances, x2y_kmean, y2x_kmean, margin)

    # if bidirectional:
    #     echo("Calculating y2x margin distances")
    #     best_y2x_indices, best_y2x_mdistances = _margin_score(y2x_indices, y2x_distances, y2x_kmean, x2y_kmean, margin)
    #     return (best_x2y_indices, best_x2y_mdistances), (best_y2x_indices, best_y2x_mdistances)

    return best_x2y_indices, best_x2y_mdistances
