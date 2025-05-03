import itertools
import re
from collections.abc import Sequence
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import scipy.ndimage
import scipy.sparse
from anndata import AnnData
from scanpy import logging
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

from pscanalysis._util import _ensure_array



def infercnv(
    adata: AnnData,
    *,
    reference_key: str | None = None,
    reference_cat: None | str | Sequence[str] = None,
    reference: np.ndarray | None = None,
    normalization_mode: str = "reference",  # 🆕 ADD
    lfc_clip: float = 3,
    window_distance: float = 5e6,
    min_genes_per_window: int = 5,
    smooth: bool = True,
    dynamic_threshold: float | None = 1.5,
    exclude_chromosomes: Sequence[str] | None = ("chrX", "chrY"),
    chunksize: int = 5000,
    n_jobs: int | None = None,
    inplace: bool = True,
    layer: str | None = None,
    key_added: str = "cnv",
    calculate_gene_values: bool = False,
    two_step_refinement: bool = False,      # 🆕 ADD
    low_variance_quantile: float = 0.3,     # 🆕 ADD
)-> None | tuple[dict, scipy.sparse.csr_matrix, np.ndarray | None]:
    """Infer Copy Number Variation (CNV) by averaging gene expression over genomic regions.

    This method is heavily inspired by `infercnv <https://github.com/broadinstitute/inferCNV/>`_
    but more computationally efficient. The method is described in more detail
    in on the :ref:`infercnv-method` page.

    There, you can also find instructions on how to :ref:`prepare input data <input-data>`.

    Parameters
    ----------
    adata
        annotated data matrix
    reference_key
        Column name in adata.obs that contains tumor/normal annotations.
        If this is set to None, the average of all cells is used as reference.
    reference_cat
        One or multiple values in `adata.obs[reference_key]` that annotate
        normal cells.
    reference
        Directly supply an array of average normal gene expression. Overrides
        `reference_key` and `reference_cat`.
    normalization_mode
        Normalization mode. Options:
        - `"reference"`: Normalize using a supplied or inferred reference profile.
        - `"auto"`: Automatically calculate a reference as the mean expression across all cells.
    lfc_clip
        Clip log fold changes at this value
    window distance
    min_genes_per_window
    smooth
    dynamic_threshold
        Values `< dynamic threshold * STDDEV` will be set to 0, where STDDEV is
        the stadard deviation of the smoothed gene expression. Set to `None` to disable
        this step.
    exclude_chromosomes
        List of chromosomes to exclude. The default is to exclude genosomes.
    chunksize
        Process dataset in chunks of cells. This allows to run infercnv on
        datasets with many cells, where the dense matrix would not fit into memory.
    n_jobs
        Number of jobs for parallel processing. Default: use all cores.
        Data will be submitted to workers in chunks, see `chunksize`.
    inplace
        If True, save the results in adata.obsm, otherwise return the CNV matrix.
    layer
        Layer from adata to use. If `None`, use `X`.
    key_added
        Key under which the cnv matrix will be stored in adata if `inplace=True`.
        Will store the matrix in `adata.obsm["X_{key_added}"] and additional information
        in `adata.uns[key_added]`.
    calculate_gene_values
        If True per gene CNVs will be calculated and stored in `adata.layers["gene_values_{key_added}"]`.
        As many genes will be included in each segment the resultant per gene value will be an average of the genes included in the segment.
        Additionally not all genes will be included in the per gene CNV, due to the window size and step size not always being a multiple of
        the number of genes. Any genes not included in the per gene CNV will be filled with NaN.
        Note this will significantly increase the memory and computation time, it is recommended to decrease the chunksize to ~100 if this is set to True.
    two_step_refinement
        If True, perform an initial rough inferCNV run without a reference to select
        low-variance cells as an internal reference. Useful for heterogeneous datasets
        without a clearly annotated normal population.
    low_variance_quantile
        Quantile threshold (between 0 and 1) to select low-variance cells during two-step refinement.
        Only used if `two_step_refinement=True`.

    Returns
    -------
    Depending on inplace, either return the smoothed and denoised gene expression
    matrix sorted by genomic position, or add it to adata.
    """
    if not adata.var_names.is_unique:
        raise ValueError("Ensure your var_names are unique!")
    if {"chromosome", "start", "end"} - set(adata.var.columns) != set():
        raise ValueError(
            "Genomic positions not found. There need to be `chromosome`, `start`, and `end` columns in `adata.var`. "
        )
    # 🆕 Auto-add "chr" if missing
    if adata.var["chromosome"].dtype == object:
        unique_chroms = adata.var["chromosome"].dropna().unique()
        if all(not str(chrom).startswith("chr") for chrom in unique_chroms):
            logging.info("Auto-adding 'chr' prefix to chromosome names...")
            adata.var["chromosome"] = adata.var["chromosome"].apply(
                lambda x: f"chr{x}" if pd.notna(x) else x
            )

    var_mask = adata.var["chromosome"].isnull()
    if np.sum(var_mask):
        logging.warning(f"Skipped {np.sum(var_mask)} genes because they don't have a genomic position annotated. ")  # type: ignore
    if exclude_chromosomes is not None:
        var_mask = var_mask | adata.var["chromosome"].isin(exclude_chromosomes)

    # === Determine expression matrix and var ===
    tmp_adata = adata[:, ~var_mask]

    # call reference on filtered adata, so shapes match
    reference = _get_reference(tmp_adata, reference_key, reference_cat, reference)

    expr = tmp_adata.X if layer is None else tmp_adata.layers[layer]
    
    if scipy.sparse.issparse(expr):
        expr = expr.tocsr()
    else:
        expr = np.asarray(expr)

    var = tmp_adata.var.loc[:, ["chromosome", "start", "end"]]

    # === Reference selection and normalization ===
    if two_step_refinement:
        logging.info("Running two-step refinement using low-variance reference...")
        mean_expr = np.mean(expr, axis=0)
        std_expr = np.std(expr, axis=0) + 1e-6
        expr_z = (expr - mean_expr) / std_expr

        chr_pos, rough_cnv, _ = _infercnv_chunk(
            expr_z,
            var,
            reference,
            lfc_cap=lfc_clip,
            window_distance=window_distance,
            min_genes_per_window=min_genes_per_window,
            dynamic_threshold=dynamic_threshold,
            smooth=True
        )

        rough_cnv = rough_cnv.toarray()
        cell_vars = np.var(rough_cnv, axis=1)
        threshold = np.quantile(cell_vars, low_variance_quantile)
        stable_mask = cell_vars <= threshold

        if not np.any(stable_mask):
            raise ValueError("No low-variance cells selected. Try raising quantile.")

        reference = np.mean(expr[stable_mask, :], axis=0, keepdims=True)
        logging.info(f"Selected {np.sum(stable_mask)} cells for refined reference.")
    elif normalization_mode == "reference":
        reference = _get_reference(tmp_adata, reference_key, reference_cat, reference)
    elif normalization_mode == "auto":
        logging.info("Using in-built average reference from all cells.")
        reference = np.mean(expr, axis=0, keepdims=True)
    else:
        raise ValueError("normalization_mode must be 'reference' or 'auto'")

    chunks = []
    chr_pos_list = []
    convolved_dfs = []
    chr_pos_final = None

    for i in tqdm(range(0, adata.shape[0], chunksize), desc="Running inferCNV chunks"):
        chunk_expr = expr[i : i + chunksize, :]
        chr_pos, chunk, _ = _infercnv_chunk(
            chunk_expr,
            var,
            reference,
            lfc_clip,
            window_distance,
            min_genes_per_window,
            smooth,
            dynamic_threshold
        )
        if chr_pos_final is None:
            chr_pos_final = chr_pos  # use the first one

        chunks.append(chunk)
        convolved_dfs.append(pd.DataFrame())  # Placeholder if needed

    res = scipy.sparse.vstack(chunks)
    chr_pos = chr_pos_final

    if calculate_gene_values:
        per_gene_df = pd.concat(convolved_dfs, axis=0)
        # Ensure the DataFrame has the correct row index
        per_gene_df.index = adata.obs.index
        # Ensure the per gene CNV matches the adata var (genes) index, any genes
        # that are not included in the CNV will be filled with NaN
        per_gene_df = per_gene_df.reindex(columns=adata.var_names, fill_value=np.nan)
        # This needs to be a numpy array as colnames are too large to save in anndata
        per_gene_mtx = per_gene_df.values
    else:
        per_gene_mtx = None

    if inplace:
        adata.obsm[f"X_{key_added}"] = res
        adata.uns[key_added] = {"chr_pos": chr_pos}

        if calculate_gene_values:
            adata.layers[f"gene_values_{key_added}"] = per_gene_mtx

    else:
        return chr_pos, res, per_gene_mtx




def _natural_sort(l: Sequence):
    """Natural sort without third party libraries.

    Adapted from: https://stackoverflow.com/a/4836734/2340703
    """

    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        return [convert(c) for c in re.split("([0-9]+)", key)]

    return sorted(l, key=alphanum_key)


def _running_mean(
    x: np.ndarray | scipy.sparse.spmatrix,
    n: int = 50,
    step: int = 10,
    gene_list: list = None,
    calculate_gene_values: bool = False,
) -> tuple[np.ndarray, pd.DataFrame | None]:
    """
    Compute a pyramidially weighted running mean.

    Densifies the matrix. Use `step` and `chunksize` to save memory.

    Parameters
    ----------
    x
        matrix to work on
    n
        Length of the running window
    step
        only compute running windows every `step` columns, e.g. if step is 10
        0:99, 10:109, 20:119 etc. Saves memory.
    gene_list
        List of gene names to be used in the convolution
    calculate_gene_values
        If True per gene CNVs will be calculated and stored in `adata.layers["gene_values_{key_added}"]`.
    """
    if n < x.shape[1]:  # regular convolution: the filter is smaller than the #genes
        r = np.arange(1, n + 1)
        pyramid = np.minimum(r, r[::-1])
        smoothed_x = np.apply_along_axis(
            lambda row: np.convolve(row, pyramid, mode="valid"),
            axis=1,
            arr=x,
        ) / np.sum(pyramid)

        ## get the indices of the genes used in the convolution
        convolution_indices = get_convolution_indices(x, n)[np.arange(0, smoothed_x.shape[1], step)]
        ## Pull out the genes used in the convolution
        convolved_gene_names = gene_list[convolution_indices]
        smoothed_x = smoothed_x[:, np.arange(0, smoothed_x.shape[1], step)]

        if calculate_gene_values:
            convolved_gene_values = _calculate_gene_averages(convolved_gene_names, smoothed_x)
        else:
            convolved_gene_values = None

        return smoothed_x, convolved_gene_values

    else:  # If there is less genes than the window size, set the window size to the number of genes and perform a single convolution
        n = x.shape[1]  # set the filter size to the number of genes
        r = np.arange(1, n + 1)
        ## As we are only doing one convolution the values should be equal
        pyramid = np.array([1] * n)
        smoothed_x = np.apply_along_axis(
            lambda row: np.convolve(row, pyramid, mode="valid"),
            axis=1,
            arr=x,
        ) / np.sum(pyramid)

        if calculate_gene_values:
            ## As all genes are used the convolution the values are identical for all genes
            convolved_gene_values = pd.DataFrame(np.repeat(smoothed_x, len(gene_list), axis=1), columns=gene_list)
        else:
            convolved_gene_values = None

        return smoothed_x, convolved_gene_values


def _calculate_gene_averages(
    convolved_gene_names: np.ndarray,
    smoothed_x: np.ndarray,
) -> pd.DataFrame:
    """
    Calculate the average value of each gene in the convolution

    Parameters
    ----------
    convolved_gene_names
        A numpy array with the gene names used in the convolution
    smoothed_x
        A numpy array with the smoothed gene expression values

    Returns
    -------
    convolved_gene_values
        A DataFrame with the average value of each gene in the convolution
    """
    ## create a dictionary to store the gene values per sample
    gene_to_values = {}
    # Calculate the number of genes in each convolution, will be same as the window size default=100
    length = len(convolved_gene_names[0])
    # Convert the flattened convolved gene names to a list
    flatten_list = list(convolved_gene_names.flatten())

    # For each sample in smoothed_x find the value for each gene and store it in a dictionary
    for sample, row in enumerate(smoothed_x):
        # Create sample level in the dictionary
        if sample not in gene_to_values:
            gene_to_values[sample] = {}
        # For each gene in the flattened gene list find the value and store it in the dictionary
        for i, gene in enumerate(flatten_list):
            if gene not in gene_to_values[sample]:
                gene_to_values[sample][gene] = []
            # As the gene list has been flattend we can use the floor division of the index
            # to get the correct position of the gene to get the value and store it in the dictionary
            gene_to_values[sample][gene].append(row[i // length])

    for sample in gene_to_values:
        for gene in gene_to_values[sample]:
            gene_to_values[sample][gene] = np.mean(gene_to_values[sample][gene])

    convolved_gene_values = pd.DataFrame(gene_to_values).T
    return convolved_gene_values

def get_convolution_indices(x, n):
    indices = []
    for i in range(x.shape[1] - n + 1):
        indices.append(np.arange(i, i + n))
    return np.array(indices)


def _running_mean_by_chromosome(
    expr,
    var,
    window_distance,  # genomic bp
    step,  # unused, kept for compatibility
    calculate_gene_values,
    min_genes_per_window=5,
    smooth=True
) -> tuple[dict, np.ndarray, list[int]]:
    """
    Compute the running mean for each chromosome independently. Stack the resulting arrays ordered by chromosome.

    Returns
    -------
    chr_start_pos
        A Dictionary mapping each chromosome to the index of running_mean where
        this chromosome begins.
    running_mean
        A numpy array with the smoothed gene expression, ordered by chromosome
        and genomic position
    n_genes_per_window : list[int]
        Number of genes used in each window.
    """
    
    chromosomes = _natural_sort([x for x in var["chromosome"].unique() if x.startswith("chr") and x != "chrM"])

    smoothed_chunks = []
    chr_pos = {}
    total_windows = 0
    n_genes_per_window = []  # store number of genes per window

    for chrom in chromosomes:
        idxs = np.where(var['chromosome'] == chrom)[0]
        starts = var.iloc[idxs]['start'].values
        expr_chr = expr[:, idxs]

        sorted_order = np.argsort(starts)
        expr_chr = expr_chr[:, sorted_order]
        starts = starts[sorted_order]

        window_exprs = []
        win_start_idx = 0

        while win_start_idx < len(starts):
            win_start_pos = starts[win_start_idx]
            window_genes = []

            for j in range(win_start_idx, len(starts)):
                if starts[j] - win_start_pos <= window_distance:
                    window_genes.append(j)
                else:
                    break
            # only accept windows with enough genes
            if len(window_genes) >= min_genes_per_window:
                if smooth:
                    window_expr = expr_chr[:, window_genes].mean(axis=1)
                else:
                    window_expr = expr_chr[:, window_genes].sum(axis=1)

                window_exprs.append(window_expr)
                n_genes_per_window.append(len(window_genes))  # record number of genes used

            win_start_idx = window_genes[-1] + 1 if len(window_genes) > 0 else win_start_idx + 1

        if len(window_exprs) > 0:
            chunk = np.vstack(window_exprs).T
            smoothed_chunks.append(chunk)
            chr_pos[chrom] = total_windows
            total_windows += chunk.shape[1]

    running_mean = np.hstack(smoothed_chunks)

    return chr_pos, running_mean, n_genes_per_window  # return genes per window too


def _get_reference(
    adata: AnnData,
    reference_key: str | None,
    reference_cat: None | str | Sequence[str],
    reference: np.ndarray | None,
) -> np.ndarray:
    """Parameter validation extraction of reference gene expression.

    If multiple reference categories are given, compute the mean per
    category.

    Returns a 2D array with reference categories in rows, cells in columns.
    If there's just one category, it's still a 2D array.
    """
    if reference is None:
        if reference_key is None or reference_cat is None:
            logging.warning(
                "Using mean of all cells as reference. For better results, "
                "provide either `reference`, or both `reference_key` and `reference_cat`. "
            )  # type: ignore
            reference = np.mean(adata.X, axis=0)

        else:
            obs_col = adata.obs[reference_key]
            if isinstance(reference_cat, str):
                reference_cat = [reference_cat]
            reference_cat = np.array(reference_cat)
            reference_cat_in_obs = np.isin(reference_cat, obs_col)
            if not np.all(reference_cat_in_obs):
                raise ValueError(
                    "The following reference categories were not found in "
                    "adata.obs[reference_key]: "
                    f"{reference_cat[~reference_cat_in_obs]}"
                )

            reference = np.vstack([np.mean(adata.X[obs_col.values == cat, :], axis=0) for cat in reference_cat])

    if reference.ndim == 1:
        reference = reference[np.newaxis, :]

    if reference.shape[1] != adata.shape[1]:
        raise ValueError("Reference must match the number of genes in AnnData. ")

    return reference

def _infercnv_chunk(
    tmp_x,
    var,
    reference,
    lfc_cap,
    window_distance,
    min_genes_per_window,
    smooth,
    dynamic_threshold,
    calculate_gene_values=False
):
    """
    The actual infercnv work is happening here.

    Process chunks of serveral thousand genes independently since this
    leads to (temporary) densification of the matrix.

    Parameters see `infercnv`.
    """
    """Chunk-based adaptive CNV inference: log fold-change, smoothing, denoising."""

    if reference.shape[0] == 1:
        x_centered = tmp_x - reference[0, :]
    else:
        ref_min = np.min(reference, axis=0)
        ref_max = np.max(reference, axis=0)
        x_centered = np.zeros(tmp_x.shape, dtype=tmp_x.dtype)
        above_max = tmp_x > ref_max
        below_min = tmp_x < ref_min
        x_centered[above_max] = _ensure_array(tmp_x - ref_max)[above_max]
        x_centered[below_min] = _ensure_array(tmp_x - ref_min)[below_min]

    x_centered = _ensure_array(x_centered)
    x_clipped = np.clip(x_centered, -lfc_cap, lfc_cap)

    chr_pos, running_mean, _ = _running_mean_by_chromosome(
        x_clipped,
        var,
        window_distance=window_distance,
        step=1,
        calculate_gene_values=False,
        min_genes_per_window=min_genes_per_window,
        smooth=smooth
    )

    x_res = running_mean - np.median(running_mean, axis=1)[:, np.newaxis]

    if dynamic_threshold is not None:
        noise_thres = dynamic_threshold * np.std(x_res)
        x_res[np.abs(x_res) < noise_thres] = 0

    x_res = scipy.sparse.csr_matrix(x_res)

    return chr_pos, x_res, None
    
