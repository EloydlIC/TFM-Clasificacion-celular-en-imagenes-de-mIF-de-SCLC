from typing import Literal
import warnings
warnings.filterwarnings('ignore')

def aggregate(sdata, 
            method: Literal["median", "mean"] = "median", 
            channels = None, 
            shrink_distance=None, 
            image_key=None, 
            shapes_key=None, 
            table_key=None, 
            verbose = False):
    
    import spatialdata as sd
    import numpy as np
    import pandas as pd
    import rasterio.features
    from anndata import AnnData
    from scipy import ndimage
    from tqdm.auto import tqdm  
    from spatialdata.models import TableModel

    # Resolving default keys
    if image_key is None: image_key = list(sdata.images.keys())[0]
    if shapes_key is None: shapes_key = list(sdata.shapes.keys())[0]
    if table_key is None and len(sdata.tables.keys()) > 0: table_key = list(sdata.tables.keys())[0]

    img = sdata.images[image_key]
    shapes = sdata.shapes[shapes_key]

    # Extracting dimensions and data from the image
    try:
        img_node = img['scale0']
    except KeyError:
        img_node = img

    var_name = list(img_node.keys())[0]
    img_data = img_node[var_name]

    shape_y = img_data.sizes['y']
    shape_x = img_data.sizes['x']
    
    if verbose: print("aggregate: Phase 1/4: Loading image onto memory...")
    img_array = np.asarray(img_data)
    
    if len(img_array.shape) == 2:
        img_array = img_array[np.newaxis, :, :]
        
    if 'c' in img_data.coords:
        channel_names = img_data.coords['c'].values.astype(str)
    else:
        channel_names = [f"channel_{i}" for i in range(img_array.shape[0])]

    if channels is not None:
        if isinstance(channels, str): 
            channels = [channels]
            
        # Verify if all the channels to aggregate are on the image
        missing_channels = [c for c in channels if c not in channel_names]
        if missing_channels:
            raise ValueError(f"The following channels do not exist in the image: {missing_channels}. Available channels: {list(channel_names)}")
        
        # Get the indeces for the channels
        channel_indices = [list(channel_names).index(c) for c in channels]
        
        # Filter both the array and the channel names
        img_array = img_array[channel_indices]
        channel_names = [channel_names[idx] for idx in channel_indices]

        if verbose: print(f"Filtering channels completed. Processing only {len(channels)} channels.")

    # Rasterization
    if verbose: print("aggregate Phase 2/4: Rasterizing polygons on the image...")
    shape_indices = shapes.index.tolist()

    geom_val_pairs = []
    for i, geom in enumerate(shapes.geometry):
        if shrink_distance is not None:
            shrunk_geom = geom.buffer(shrink_distance)
            if not shrunk_geom.is_empty:
                geom_val_pairs.append((shrunk_geom, i + 1))
            else:
                geom_val_pairs.append((geom, i + 1))
        else:
            geom_val_pairs.append((geom, i + 1))

    labels_matrix = rasterio.features.rasterize(
        geom_val_pairs,
        out_shape=(shape_y, shape_x),
        fill=0,
        dtype=np.int32)

    unique_labels = np.unique(labels_matrix)
    unique_labels = unique_labels[unique_labels > 0]

    # Basic optimization: Precompute bounding boxes for each cell. Otherwise it would read the whole image for each cell
    if verbose: print("aggregate Phase 3/4: Precomputing pixel maps for each cell...")
    slices = ndimage.find_objects(labels_matrix)
    cell_geometry_cache = {}
    for label in unique_labels:
        sl = slices[label - 1]
        if sl is not None:
            cell_geometry_cache[label] = (sl, labels_matrix[sl] == label)

    # Compute the aggregated intensities
    if verbose: print(f"aggregate Phase 4/4: Calculating intensities using method '{method}'...")
    aggregated_values = np.zeros((len(unique_labels), img_array.shape[0]))

    for c in tqdm(range(img_array.shape[0]), desc="Processing channels"):
        channel_data = img_array[c]
        
        
        for i, label in enumerate(unique_labels):
            if label not in cell_geometry_cache: continue
            sl, mask = cell_geometry_cache[label]
            cell_pixels = channel_data[sl][mask]
               
            if len(cell_pixels) > 0:
                if method == "median":
                    aggregated_values[i, c] = np.median(cell_pixels)
                if method == "mean":
                    aggregated_values[i, c] = np.mean(cell_pixels)
           

    # Build the anndata object
    label_to_index = {i + 1: str(idx) for i, idx in enumerate(shapes.index.tolist())}
    valid_indices = [label_to_index[val] for val in unique_labels]

    agg_adata = AnnData(
        X=aggregated_values,
        obs=pd.DataFrame(index=valid_indices),
        var=pd.DataFrame(index=channel_names)
    )

    # Passing the metadata from the original table
    if table_key is not None:
        orig_adata = sdata.tables[table_key]
        common_obs = agg_adata.obs_names.intersection(orig_adata.obs_names)
        columns_to_keep = orig_adata.obs.columns.difference(agg_adata.obs.columns)
        
        for col in columns_to_keep:
            agg_adata.obs[col] = pd.Series(index=agg_adata.obs.index, dtype=orig_adata.obs[col].dtype)
            agg_adata.obs.loc[common_obs, col] = orig_adata.obs.loc[common_obs, col]
            
        table_name = table_key
    else:
        table_name = "table"
    
    # Extracting the previous configuration, if it exists
    if table_key is not None and "spatialdata_attrs" in orig_adata.uns:
        attrs_old = orig_adata.uns["spatialdata_attrs"]
        region_actual = attrs_old.get("region", shapes_key)
        region_key_actual = attrs_old.get("region_key", "region")
        instance_key_actual = attrs_old.get("instance_key", "instance_id")
    else:
        region_actual = shapes_key
        region_key_actual = "region"
        instance_key_actual = "instance_id"

    agg_adata.obs[region_key_actual] = agg_adata.obs[region_key_actual].astype(object)

    # Manage the 'region' column in obs
    if region_key_actual not in agg_adata.obs.columns:
        agg_adata.obs[region_key_actual] = region_actual if isinstance(region_actual, str) else shapes_key
    else:
        # If the column already existed and was inherited, fill new cells (NaN)
        if isinstance(region_actual, str):
            agg_adata.obs[region_key_actual] = agg_adata.obs[region_key_actual].fillna(region_actual)
        else:
            agg_adata.obs[region_key_actual] = agg_adata.obs[region_key_actual].fillna(shapes_key)
            # If the old 'region' was a multi-region list, ensure shapes_key is in it
            if shapes_key not in region_actual:
                region_actual = list(region_actual) + [shapes_key]

    agg_adata.obs[region_key_actual] = agg_adata.obs[region_key_actual].astype("category")
    
    if table_key is not None and instance_key_actual in orig_adata.obs:
        missing_obs = agg_adata.obs_names.difference(orig_adata.obs_names)
        if len(missing_obs) > 0:
            if isinstance(shapes.index[0], int) or np.issubdtype(shapes.index.dtype, np.integer):
                agg_adata.obs.loc[missing_obs, instance_key_actual] = missing_obs.astype(int)
            else:
                agg_adata.obs.loc[missing_obs, instance_key_actual] = missing_obs
    else:
        if isinstance(shapes.index[0], int) or np.issubdtype(shapes.index.dtype, np.integer):
            agg_adata.obs[instance_key_actual] = agg_adata.obs.index.astype(int)
        else:
            agg_adata.obs[instance_key_actual] = agg_adata.obs.index

    region_actual = agg_adata.obs[region_key_actual].unique().tolist()
    if len(region_actual) == 1:
        region_actual = region_actual[0]

    # Parsing the new metadata
    agg_adata = TableModel.parse(
        agg_adata,
        region=region_actual,
        region_key=region_key_actual,
        instance_key=instance_key_actual
    )

    # Build and return
    new_sdata = sd.SpatialData(
        images=dict(sdata.images),
        shapes=dict(sdata.shapes),
        points=dict(sdata.points) if hasattr(sdata, "points") else {},
        labels=dict(sdata.labels) if hasattr(sdata, "labels") else {},
        tables=dict(sdata.tables) if hasattr(sdata, "tables") else {}
    )
    
    if table_name in new_sdata.tables:
        del new_sdata.tables[table_name]
        
    new_sdata.tables[table_name] = agg_adata

    if verbose: print(f"aggregate completed")

    return new_sdata

from typing import Literal

def prefilter_polygons(
        sdata, 
        image_key=None, 
        shapes_key=None, 
        global_positive_strategy: Literal['rightmost', 'all_except_leftmost'] = 'rightmost',
        subsample_size=200000, 
        clipping_percentile=95,
        channels=None, 
        data_is_logarithmic=False, 
        verbose=False
        ):
    
    import spatialdata as sd
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import xarray as xr
    import rasterio.features
    import shapely.geometry
    import shapely.ops
    import shapely.affinity
    import matplotlib.pyplot as plt
    from scipy import ndimage
    from scipy.stats import gaussian_kde, norm
    from sklearn.mixture import GaussianMixture
    from tqdm.auto import tqdm  

    # Version control for native vs external DataTree
    try:
        from xarray import DataTree
    except ImportError:
        import datatree as dt
        DataTree = dt.DataTree

    # Resolve default keys
    if image_key is None: image_key = list(sdata.images.keys())[0]
    if shapes_key is None: shapes_key = list(sdata.shapes.keys())[0]

    img = sdata.images[image_key]
    shapes = sdata.shapes[shapes_key]

    try: img_node = img['scale0']
    except KeyError: img_node = img

    var_name = list(img_node.keys())[0]
    img_data = img_node[var_name]
    
    available_channels = [str(c) for c in img_data.coords['c'].values] if 'c' in img_data.coords else []
    print(f"Channels detected in the image: {available_channels}")

    if channels is not None:
        if isinstance(channels, str): channels = [channels]
        valid_channels = [c for c in channels if c in available_channels]
    else:
        valid_channels = available_channels

    print("Phase 1/5: Projecting maximum resolution channels into memory...")
    img_array_full = np.asarray(img_data.values if hasattr(img_data, 'values') else img_data).astype(np.float32)
    if len(img_array_full.shape) == 3:
        if channels is not None:
            indices = [available_channels.index(c) for c in valid_channels]
            sum_img = np.sum(img_array_full[indices, :, :], axis=0)
        else:
            sum_img = np.sum(img_array_full, axis=0)
    else:
        sum_img = img_array_full

    shape_y, shape_x = sum_img.shape

    pct = np.percentile(sum_img, clipping_percentile)
    print(f"Applying clipping at percentile {clipping_percentile} ({pct:.2f}) to mitigate artifacts before GMM.")
    sum_img = np.clip(sum_img, a_min=None, a_max=pct)

    # Select the threshold through GMM
    def calculate_robust_gmm_threshold(pixels_data, context_name, fallback_threshold=None, is_global=False):

        if len(pixels_data) < 5:
            return fallback_threshold if fallback_threshold is not None else float(np.mean(pixels_data))
            
        # Apply safe logarithmic transformation if original data is linear
        if not data_is_logarithmic:
            transformed_pixels = np.log1p(pixels_data)
        else:
            transformed_pixels = pixels_data
            
        if is_global and subsample_size and len(transformed_pixels) > subsample_size:
            rng = np.random.default_rng(42)
            X = rng.choice(transformed_pixels, size=subsample_size, replace=False).reshape(-1, 1)
        else:
            X = transformed_pixels.reshape(-1, 1)
            
        def evaluate_k(n_comp):
            try:
                gmm = GaussianMixture(n_components=n_comp, random_state=42, max_iter=100)
                gmm.fit(X)
                means = gmm.means_.flatten()
                idx_sorted = np.argsort(means)
                
                bic = gmm.bic(X)
                
                grid = np.linspace(np.min(X), np.max(X), 1000).reshape(-1, 1)
                probs = gmm.predict_proba(grid)
                
                if is_global:
                    if global_positive_strategy == 'rightmost':
                        # Only the rightmost component is signal
                        idx_pos = idx_sorted[-1]
                        prob_pos = probs[:, idx_pos]
                        condition = prob_pos > 0.5  

                        # Check valid_separation
                        m_neg, m_pos = means[idx_sorted[-2]], means[idx_sorted[-1]]
                        sigma_neg = np.sqrt(gmm.covariances_[idx_sorted[-2]].flatten()[0])
                        sigma_pos = np.sqrt(gmm.covariances_[idx_sorted[-1]].flatten()[0])
                        D = np.sqrt(2)* np.abs(m_pos - m_neg) / np.sqrt(sigma_neg**2 + sigma_pos**2)
                        valid_separation = D >= 2

                        # Find intersection to the right of the penultimate component
                        prev_m = means[idx_sorted[-2]]
                        valid_indices = np.where(condition & (grid.flatten() > prev_m))[0]
                        
                        if len(valid_indices) > 0:
                            thresh = float(grid[valid_indices[0]][0])
                        else:
                            thresh = float((prev_m + means[idx_pos]) / 2)
                    else:
                        # All except the leftmost are signal
                        idx_neg = idx_sorted[0]
                        prob_neg = probs[:, idx_neg]
                        condition = prob_neg < 0.5  
                        
                        neg_m = means[idx_neg]
                        valid_indices = np.where(condition & (grid.flatten() > neg_m))[0]
                        
                        if len(valid_indices) > 0:
                            thresh = float(grid[valid_indices[0]][0])
                        else:
                            thresh = float((neg_m + means[idx_sorted[1]]) / 2)
                else:
                    # Everything that is not the leftmost component is signal
                    m_neg, m_pos = means[idx_sorted[0]], means[idx_sorted[1]]
                    sigma_neg = np.sqrt(gmm.covariances_[idx_sorted[0]].flatten()[0])
                    sigma_pos = np.sqrt(gmm.covariances_[idx_sorted[1]].flatten()[0])
                    D = np.sqrt(2) * np.abs(m_pos - m_neg) / np.sqrt(sigma_neg**2 + sigma_pos**2)
                    valid_separation = D >= 2

                    bg_prob = probs[:, idx_sorted[0]]
                    rest_prob = 1.0 - bg_prob
                    condition = rest_prob > bg_prob
                    
                    valid_indices = np.where(condition & (grid.flatten() > m_neg))[0]
                    
                    if len(valid_indices) > 0:
                        thresh = float(grid[valid_indices[0]][0])
                    else:
                        thresh = float((m_neg + m_pos) / 2)
                    
                return {"passed": valid_separation, "bic": bic, "thresh": thresh, "means": (m_neg, m_pos), "sigma": sigma_neg, "model": gmm, "idx_sorted": idx_sorted}
            except Exception:
                return {"passed": False, "bic": np.inf, "thresh": float(np.mean(X)), "means": (0, 0), "sigma": 1, "model": None, "idx_sorted": None}
        
        res2 = evaluate_k(2)
        res3 = evaluate_k(3)
        res4 = evaluate_k(4)
        
        if res3["passed"]:
            best_res, best_k = res3, 3
        else:
            if res2["bic"] <= res4["bic"]:
                bic_winner, winner_k, bic_loser, loser_k = res2, 2, res4, 4
            else:
                bic_winner, winner_k, bic_loser, loser_k = res4, 4, res2, 2
                
            if bic_winner["passed"]:
                best_res, best_k = bic_winner, winner_k
            elif bic_loser["passed"]:
                best_res, best_k = bic_loser, loser_k
            else:
                best_res, best_k = None, "Fallback"

        # Extract the threshold in the scale the GMM worked with (useful for plotting)
        thresh_in_gmm_scale = best_res["thresh"] if best_res else float(np.percentile(X, 90))

        # KDE plot
        if is_global and verbose:
            print("Generating GMM diagnostic plots...")
            x_range = np.linspace(np.min(X), np.max(X), 1000)
            kde_real = gaussian_kde(X.flatten())(x_range)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Left Panel: Estimated distributions for the different values of k 
            ax1.plot(x_range, kde_real, color='black', lw=2, label='Real Density (KDE)')
            ax1.fill_between(x_range, 0, kde_real, color='black', alpha=0.05)
            if res2["model"]: ax1.plot(x_range, np.exp(res2["model"].score_samples(x_range.reshape(-1, 1))), '--', color='forestgreen', label=f'k=2 (BIC: {res2["bic"]:.0f})')
            if res3["model"]: ax1.plot(x_range, np.exp(res3["model"].score_samples(x_range.reshape(-1, 1))), '--', color='purple', label=f'k=3 (BIC: {res3["bic"]:.0f})')
            if res4["model"]: ax1.plot(x_range, np.exp(res4["model"].score_samples(x_range.reshape(-1, 1))), '--', color='dodgerblue', label=f'k=4 (BIC: {res4["bic"]:.0f})')
            ax1.axvline(thresh_in_gmm_scale, color='red', linestyle='-', lw=2, label=f'Gate: {thresh_in_gmm_scale:.2f}')
            
            title_suffix = " (Original k=3 rejected)" if best_k != 3 and best_k != "Fallback" else ""
            ax1.set_title(f"Scout Selection: Final winner k={best_k}{title_suffix}")
            ax1.set_xlabel("Log Intensity" if not data_is_logarithmic else "Intensity")
            ax1.set_ylabel("Density")
            ax1.legend(loc='upper right', fontsize='small')
            
            # Right Panel: Final applied components
            ax2.plot(x_range, kde_real, color='black', lw=2, label='Real Density (KDE)')
            ax2.fill_between(x_range, 0, kde_real, color='black', alpha=0.05)
            
            if best_res and best_res["model"]:
                gmm_plot = best_res["model"]
                m_scout = gmm_plot.means_.flatten()
                s_scout = np.sqrt(gmm_plot.covariances_.flatten())
                w_scout = gmm_plot.weights_
                idx_scout = best_res["idx_sorted"]
                
                for idx in idx_scout:
                    color = 'blue' if best_k == 1 else ('green' if idx == idx_scout[-1] else 'blue')
                    ax2.plot(x_range, w_scout[idx] * norm.pdf(x_range, m_scout[idx], s_scout[idx]), '--', color=color, alpha=0.6)
            
            ax2.axvline(thresh_in_gmm_scale, color='red', lw=2, label=f'Gate: {thresh_in_gmm_scale:.2f}')
            ax2.set_title(f"Active Final Components Analysis (k={best_k})")
            ax2.set_xlabel("Log Intensity" if not data_is_logarithmic else "Intensity")
            ax2.legend(loc='upper right', fontsize='small')
            
            plt.suptitle(f"Marker: {context_name}", fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()

        # If computed in log, return threshold in linear scale
        if best_res:
            if not data_is_logarithmic:
                return float(np.expm1(best_res["thresh"]))
            else:
                return float(best_res["thresh"])
        else:
            return fallback_threshold if fallback_threshold is not None else float(np.percentile(pixels_data, 90))

    print("Phase 2/5: Executing global GMM optimization...")
    global_threshold = calculate_robust_gmm_threshold(sum_img.ravel(), context_name="GLOBAL IMAGE", fallback_threshold=None, is_global=True)
    print(f"Final Global Threshold (Linear Scale): {global_threshold:.2f}\n")

    print("Phase 3/5: Rasterizing original polygons...")
    geom_val_pairs = [(geom, i + 1) for i, geom in enumerate(shapes.geometry)]
    labels_matrix = rasterio.features.rasterize(geom_val_pairs, out_shape=(shape_y, shape_x), fill=0, dtype=np.int32)
    unique_labels = np.unique(labels_matrix)[np.unique(labels_matrix) > 0]

    print("Phase 4/5: Precomputing bounding boxes...")
    slices = ndimage.find_objects(labels_matrix)
    cell_geometry_cache = {lbl: (slices[lbl-1], labels_matrix[slices[lbl-1]] == lbl) for lbl in unique_labels if slices[lbl-1] is not None}

    # Local filtering
    print("Phase 5/5: Evaluating and cleaning cell geometries...")
    new_geometries = list(shapes.geometry.values)
    stats = {"intact_ok": 0, "intact_dark": 0, "filtered": 0}

    for label in tqdm(unique_labels, desc="Filtering cells"):
        orig_idx = label - 1
        if label not in cell_geometry_cache: continue
        sl, mask = cell_geometry_cache[label]
        
        local_sum_img = np.array(sum_img[sl], copy=True)
        cell_pixels = local_sum_img[mask]
        if cell_pixels.size == 0: continue
        
        if np.all(cell_pixels > global_threshold):
            stats["intact_ok"] += 1
            geom_orig = shapes.geometry.values[orig_idx]
            new_geometries[orig_idx] = geom_orig.buffer(0.7, join_style=1).buffer(-0.7, join_style=1).simplify(0.5, preserve_topology=True)
            continue
        else:
            stats["filtered"] += 1
            local_threshold = calculate_robust_gmm_threshold(cell_pixels, context_name=f"Cell {label}", fallback_threshold=global_threshold, is_global=False)
            
            new_cell_mask = mask & (local_sum_img > local_threshold)
            if not np.any(new_cell_mask): continue
            
            shapes_generator = rasterio.features.shapes(new_cell_mask.astype(np.int32), mask=new_cell_mask)
            local_geoms = [shapely.geometry.shape(g) for g, val in shapes_generator if val == 1]
            
            if local_geoms:
                new_geom = shapely.ops.unary_union(local_geoms)
                new_geom = new_geom.buffer(0.7, join_style=1).buffer(-0.7, join_style=1)
                new_geom = new_geom.simplify(0.5, preserve_topology=True)
            
                if not new_geom.is_empty:
                    # Remove disconnected regions keeping only the one with the largest area
                    if isinstance(new_geom, shapely.geometry.MultiPolygon):
                        new_geom = max(new_geom.geoms, key=lambda p: p.area)
                
                    # Fill internal holes by taking only the coordinates of the exterior ring
                    if isinstance(new_geom, shapely.geometry.Polygon) and not new_geom.is_empty:
                        new_geom = shapely.geometry.Polygon(new_geom.exterior)
            
                new_geom = shapely.affinity.translate(new_geom, xoff=sl[1].start, yoff=sl[0].start)
                new_geometries[orig_idx] = new_geom


    # Rebuild the DataTree and the SpatialData
    print("Updating channels in the image structure...")
    transformations = sd.transformations.get_transformation(img, get_all=True)

    if hasattr(img, "children") and len(img.children) > 0:
        new_scales = {}
        for scale_name in img.children:
            scale_ds = img[scale_name]
            v_name = list(scale_ds.keys())[0]
            da_scale = scale_ds[v_name]
            
            valid_c_scale = [c for c in valid_channels if c in da_scale.coords['c'].values]
            da_sel = da_scale.sel(c=valid_c_scale) if valid_c_scale else da_scale
                
            sum_da = da_sel.sum(dim='c').expand_dims(dim={'c': ['channel_sum']})
            sum_da = sum_da.assign_coords(y=da_scale.y, x=da_scale.x)
            
            updated_da = xr.concat([da_scale, sum_da], dim='c')
            new_scales[scale_name] = xr.Dataset({v_name: updated_da})
            
        updated_img = DataTree.from_dict(new_scales)
    else:
        v_name = list(img.keys())[0] if hasattr(img, "keys") else None
        da_scale = img[v_name] if v_name else img
        
        valid_c_scale = [c for c in valid_channels if c in da_scale.coords['c'].values]
        da_sel = da_scale.sel(c=valid_c_scale) if valid_c_scale else da_scale
            
        sum_da = da_sel.sum(dim='c').expand_dims(dim={'c': ['channel_sum']})
        sum_da = sum_da.assign_coords(y=da_scale.y, x=da_scale.x)
        updated_da = xr.concat([da_scale, sum_da], dim='c')
        updated_img = xr.Dataset({v_name: updated_da}) if v_name else updated_da

    sd.transformations.set_transformation(updated_img, transformations, set_all=True)

    updated_shapes = gpd.GeoDataFrame(shapes.copy(), geometry=new_geometries, crs=shapes.crs)
    if hasattr(shapes, "attrs"): updated_shapes.attrs = shapes.attrs.copy()

    original_shapes_backup = gpd.GeoDataFrame(shapes.copy(), geometry=shapes.geometry.values, crs=shapes.crs)
    if hasattr(shapes, "attrs"): original_shapes_backup.attrs = shapes.attrs.copy()
    
    dict_images = dict(sdata.images)
    dict_images[image_key] = updated_img  
    
    dict_shapes = dict(sdata.shapes)
    dict_shapes[shapes_key] = updated_shapes                                  
    dict_shapes[f"{shapes_key}_original"] = original_shapes_backup       

    new_sdata = sd.SpatialData(
        images=dict_images,
        shapes=dict_shapes,
        points=dict(sdata.points) if hasattr(sdata, "points") else {},
        labels=dict(sdata.labels) if hasattr(sdata, "labels") else {},
        tables=dict(sdata.tables) if hasattr(sdata, "tables") else {}
    )
    return new_sdata

def separate_table_by_phase(df, phase_column="Phase", state_value="State"):

    # Verify if phase_column exists in the DataFrame
    if phase_column not in df.columns:
        raise KeyError(f"The DataFrame does not contain the column '{phase_column}'")
    
    # Rows where the phase is state_value
    df_state = df[df['Phase'] == state_value].drop(columns=['Phase']).copy()
    
    # The rest of the rows (lineage)
    df_rest = df[df['Phase'] != state_value].drop(columns=['Phase']).copy()
    
    return df_rest, df_state

def clean_and_repair_colors(sdata, column):
    """
    Systematically detects and repairs mismatches between the number of categories 
    of a column in `.obs` and its color palette in `.uns`.
    """
    import pandas as pd
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np

    # Find all associated tables in the SpatialData object
    tables = []
    if hasattr(sdata, 'tables'):
        for k in sdata.tables.keys():
            tables.append(sdata.tables[k])

    for adata in tables:
        if column in adata.obs.columns:
            # 1. Ensure the column is strictly of Categorical type
            if not isinstance(adata.obs[column].dtype, pd.CategoricalDtype):
                adata.obs[column] = adata.obs[column].astype('category')
            
            # 2. Get current categories and the corresponding color key
            cats = adata.obs[column].cat.categories
            n_cats = len(cats)
            colors_key = f"{column}_colors"
            
            # 3. If the saved palette size mismatches, it is removed to avoid a crash
            if colors_key in adata.uns:
                if len(adata.uns[colors_key]) != n_cats:
                    print(f"[FIX] Mismatch in '{colors_key}': {len(adata.uns[colors_key])} colors vs {n_cats} categories. Removing obsolete palette...")
                    del adata.uns[colors_key]
            
            # 4. Safely force the creation of a new palette of the exact size
            if colors_key not in adata.uns:
                if n_cats <= 10:
                    cmap = plt.get_cmap('tab10')
                elif n_cats <= 20:
                    cmap = plt.get_cmap('tab20')
                else:
                    cmap = plt.get_cmap('turbo')
                
                # Generate colors in HEX string format compatible with Scanpy/SpatialData
                adata.uns[colors_key] = [mpl.colors.to_hex(cmap(i)) for i in np.linspace(0, 1, n_cats)]
                print(f"Generated new clean palette with {n_cats} colors for '{column}'.")

def extract_nuclei(
        sdata, 
        dapi_channel='DAPI',
        image_key=None, 
        shapes_key=None, 
        output_shapes_key=None,
        global_positive_strategy: Literal['rightmost', 'all_except_leftmost'] = 'rightmost',
        subsample_size=200000, 
        clipping_percentile=95,
        data_is_logarithmic=False, 
        verbose=False
        ):
    """
    Takes a SpatialData object and generates a new polygon layer representing nuclei,
    using a GMM model fitted exclusively on the DAPI channel.
    """

    import numpy as np
    import spatialdata as sd
    import geopandas as gpd
    from scipy.stats import gaussian_kde, norm
    from sklearn.mixture import GaussianMixture
    from tqdm.auto import tqdm
    import matplotlib.pyplot as plt
    from scipy import ndimage
    import rasterio
    import shapely
    
    # Version control for native vs external DataTree
    try:
        from xarray import DataTree
    except ImportError:
        import datatree as dt
        DataTree = dt.DataTree

    # Resolve default keys
    if image_key is None: image_key = list(sdata.images.keys())[0]
    if shapes_key is None: shapes_key = list(sdata.shapes.keys())[0]
    if output_shapes_key is None: output_shapes_key = f"{shapes_key}_nuclei"

    img = sdata.images[image_key]
    shapes = sdata.shapes[shapes_key]

    try: img_node = img['scale0']
    except KeyError: img_node = img

    var_name = list(img_node.keys())[0]
    img_data = img_node[var_name]
    
    available_channels = [str(c) for c in img_data.coords['c'].values] if 'c' in img_data.coords else []
    print(f"[INFO] Channels detected in the image: {available_channels}")

    if dapi_channel not in available_channels:
        raise ValueError(f"Channel '{dapi_channel}' not found. Make sure to provide the exact name.")

    print(f"Phase 1/5: Extracting the {dapi_channel} channel at maximum resolution...")
    img_array_full = np.asarray(img_data.values if hasattr(img_data, 'values') else img_data).astype(np.float32)
    
    if len(img_array_full.shape) == 3:
        dapi_idx = available_channels.index(dapi_channel)
        dapi_img = img_array_full[dapi_idx, :, :]
    else:
        dapi_img = img_array_full

    shape_y, shape_x = dapi_img.shape

    pct = np.percentile(dapi_img, clipping_percentile)
    print(f"[INFO] Applying clipping at percentile {clipping_percentile} ({pct:.2f}) on DAPI.")
    dapi_img = np.clip(dapi_img, a_min=None, a_max=pct)

    # GMM model selection and threshold calculation
    def calculate_robust_gmm_threshold(pixels_data, context_name, fallback_threshold=None, is_global=False):

        if len(pixels_data) < 5:
            return fallback_threshold if fallback_threshold is not None else float(np.mean(pixels_data))
            
        # Apply safe logarithmic transformation if original data is linear
        if not data_is_logarithmic:
            transformed_pixels = np.log1p(pixels_data)
        else:
            transformed_pixels = pixels_data
            
        if is_global and subsample_size and len(transformed_pixels) > subsample_size:
            rng = np.random.default_rng(42)
            X = rng.choice(transformed_pixels, size=subsample_size, replace=False).reshape(-1, 1)
        else:
            X = transformed_pixels.reshape(-1, 1)
            
        def evaluate_k(n_comp):
            try:
                gmm = GaussianMixture(n_components=n_comp, random_state=42, max_iter=100)
                gmm.fit(X)
                means = gmm.means_.flatten()
                idx_sorted = np.argsort(means)
                
                m1, m2 = means[idx_sorted[0]], means[idx_sorted[1]]
                sigma_1 = np.sqrt(gmm.covariances_[idx_sorted[0]].flatten()[0])
                
                valid_separation = np.abs(m2 - m1) >= sigma_1
                bic = gmm.bic(X)
                
                grid = np.linspace(np.min(X), np.max(X), 1000).reshape(-1, 1)
                probs = gmm.predict_proba(grid)
                
                if is_global:
                    if global_positive_strategy == 'rightmost':
                        idx_pos = idx_sorted[-1]
                        prob_pos = probs[:, idx_pos]
                        condition = prob_pos > 0.5  
                        
                        prev_m = means[idx_sorted[-2]]
                        valid_indices = np.where(condition & (grid.flatten() > prev_m))[0]
                        
                        if len(valid_indices) > 0:
                            thresh = float(grid[valid_indices[0]][0])
                        else:
                            thresh = float((prev_m + means[idx_pos]) / 2)
                    else:
                        idx_neg = idx_sorted[0]
                        prob_neg = probs[:, idx_neg]
                        condition = prob_neg < 0.5  
                        
                        neg_m = means[idx_neg]
                        valid_indices = np.where(condition & (grid.flatten() > neg_m))[0]
                        
                        if len(valid_indices) > 0:
                            thresh = float(grid[valid_indices[0]][0])
                        else:
                            thresh = float((neg_m + means[idx_sorted[1]]) / 2)
                else:
                    bg_prob = probs[:, idx_sorted[0]]
                    rest_prob = 1.0 - bg_prob
                    condition = rest_prob > bg_prob
                    
                    valid_indices = np.where(condition & (grid.flatten() > m1))[0]
                    
                    if len(valid_indices) > 0:
                        thresh = float(grid[valid_indices[0]][0])
                    else:
                        thresh = float((m1 + m2) / 2)
                    
                return {"passed": valid_separation, "bic": bic, "thresh": thresh, "means": (m1, m2), "sigma": sigma_1, "model": gmm, "idx_sorted": idx_sorted}
            except Exception:
                return {"passed": False, "bic": np.inf, "thresh": float(np.mean(X)), "means": (0, 0), "sigma": 1, "model": None, "idx_sorted": None}
        
        res2 = evaluate_k(2)
        res3 = evaluate_k(3)
        res4 = evaluate_k(4)
        
        if res3["passed"]:
            best_res, best_k = res3, 3
        else:
            if res2["bic"] <= res4["bic"]:
                bic_winner, winner_k, bic_loser, loser_k = res2, 2, res4, 4
            else:
                bic_winner, winner_k, bic_loser, loser_k = res4, 4, res2, 2
                
            if bic_winner["passed"]:
                best_res, best_k = bic_winner, winner_k
            elif bic_loser["passed"]:
                best_res, best_k = bic_loser, loser_k
            else:
                best_res, best_k = None, "Fallback"

        thresh_in_gmm_scale = best_res["thresh"] if best_res else float(np.percentile(X, 90))

        if is_global and verbose:
            print("Generating GMM diagnostic plots...")
            x_range = np.linspace(np.min(X), np.max(X), 1000)
            kde_real = gaussian_kde(X.flatten())(x_range)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            ax1.plot(x_range, kde_real, color='black', lw=2, label='Real Density (KDE)')
            ax1.fill_between(x_range, 0, kde_real, color='black', alpha=0.05)
            if res2["model"]: ax1.plot(x_range, np.exp(res2["model"].score_samples(x_range.reshape(-1, 1))), '--', color='forestgreen', label=f'k=2 (BIC: {res2["bic"]:.0f})')
            if res3["model"]: ax1.plot(x_range, np.exp(res3["model"].score_samples(x_range.reshape(-1, 1))), '--', color='purple', label=f'k=3 (BIC: {res3["bic"]:.0f})')
            if res4["model"]: ax1.plot(x_range, np.exp(res4["model"].score_samples(x_range.reshape(-1, 1))), '--', color='dodgerblue', label=f'k=4 (BIC: {res4["bic"]:.0f})')
            ax1.axvline(thresh_in_gmm_scale, color='red', linestyle='-', lw=2, label=f'Gate: {thresh_in_gmm_scale:.2f}')
            
            title_suffix = " (Original k=3 rejected)" if best_k != 3 and best_k != "Fallback" else ""
            ax1.set_title(f"Scout Selection: Final winner k={best_k}{title_suffix}")
            ax1.set_xlabel("Log Intensity" if not data_is_logarithmic else "Intensity")
            ax1.set_ylabel("Density")
            ax1.legend(loc='upper right', fontsize='small')
            
            ax2.plot(x_range, kde_real, color='black', lw=2, label='Real Density (KDE)')
            ax2.fill_between(x_range, 0, kde_real, color='black', alpha=0.05)
            
            if best_res and best_res["model"]:
                gmm_plot = best_res["model"]
                m_scout = gmm_plot.means_.flatten()
                s_scout = np.sqrt(gmm_plot.covariances_.flatten())
                w_scout = gmm_plot.weights_
                idx_scout = best_res["idx_sorted"]
                
                for idx in idx_scout:
                    color = 'blue' if best_k == 1 else ('green' if idx == idx_scout[-1] else 'blue')
                    ax2.plot(x_range, w_scout[idx] * norm.pdf(x_range, m_scout[idx], s_scout[idx]), '--', color=color, alpha=0.6)
            
            ax2.axvline(thresh_in_gmm_scale, color='red', lw=2, label=f'Gate: {thresh_in_gmm_scale:.2f}')
            ax2.set_title(f"Active Final Components Analysis (k={best_k})")
            ax2.set_xlabel("Log Intensity" if not data_is_logarithmic else "Intensity")
            ax2.legend(loc='upper right', fontsize='small')
            
            plt.suptitle(f"Marker: {context_name}", fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()

        if best_res:
            if not data_is_logarithmic: return float(np.expm1(best_res["thresh"]))
            else: return float(best_res["thresh"])
        else:
            return fallback_threshold if fallback_threshold is not None else float(np.percentile(pixels_data, 90))

    # Calculate global DAPI threshold
    print("Phase 2/5: Executing global GMM optimization on DAPI...")
    global_threshold = calculate_robust_gmm_threshold(dapi_img.ravel(), context_name="GLOBAL DAPI", fallback_threshold=None, is_global=True)
    print(f"Final Global Threshold (Linear Scale): {global_threshold:.2f}\n")

    # Rasterize
    print("Phase 3/5: Rasterizing original cells...")
    geom_val_pairs = [(geom, i + 1) for i, geom in enumerate(shapes.geometry)]
    labels_matrix = rasterio.features.rasterize(geom_val_pairs, out_shape=(shape_y, shape_x), fill=0, dtype=np.int32)
    unique_labels = np.unique(labels_matrix)[np.unique(labels_matrix) > 0]

    print("Phase 4/5: Precomputing spatial bounding boxes...")
    slices = ndimage.find_objects(labels_matrix)
    cell_geometry_cache = {lbl: (slices[lbl-1], labels_matrix[slices[lbl-1]] == lbl) for lbl in unique_labels if slices[lbl-1] is not None}

    # Local nuclei filtering
    print("Phase 5/5: Evaluating intracellular nuclear contours...")
    new_geometries = list(shapes.geometry.values)
    stats = {"intact_nuclei": 0, "refined_nuclei": 0}

    for label in tqdm(unique_labels, desc="Extracting nuclei"):
        orig_idx = label - 1
        if label not in cell_geometry_cache: continue
        sl, mask = cell_geometry_cache[label]
        
        local_dapi_img = np.array(dapi_img[sl], copy=True)
        cell_pixels = local_dapi_img[mask]
        if cell_pixels.size == 0: continue
        
        if np.all(cell_pixels > global_threshold):
            stats["intact_nuclei"] += 1
            geom_orig = shapes.geometry.values[orig_idx]
            new_geometries[orig_idx] = geom_orig.buffer(0.7, join_style=1).buffer(-0.7, join_style=1).simplify(0.5, preserve_topology=True)
            continue
        else:
            stats["refined_nuclei"] += 1
            local_threshold = calculate_robust_gmm_threshold(cell_pixels, context_name=f"Cell {label}", fallback_threshold=global_threshold, is_global=False)
            
            new_cell_mask = mask & (local_dapi_img > local_threshold)
            if not np.any(new_cell_mask): 
                # Here we leave it with an empty geometry for safety.
                new_geometries[orig_idx] = shapely.geometry.Polygon() 
                continue
            
            shapes_generator = rasterio.features.shapes(new_cell_mask.astype(np.int32), mask=new_cell_mask)
            local_geoms = [shapely.geometry.shape(g) for g, val in shapes_generator if val == 1]
            
            if local_geoms:
                new_geom = shapely.ops.unary_union(local_geoms)
                new_geom = new_geom.buffer(0.7, join_style=1).buffer(-0.7, join_style=1)
                new_geom = new_geom.simplify(0.5, preserve_topology=True)
            
                if not new_geom.is_empty:
                    # Remove disconnected regions keeping only the one with the largest area
                    if isinstance(new_geom, shapely.geometry.MultiPolygon):
                        new_geom = max(new_geom.geoms, key=lambda p: p.area)
                
                    # Fill internal holes by taking only the coordinates of the exterior ring
                    if isinstance(new_geom, shapely.geometry.Polygon) and not new_geom.is_empty:
                        new_geom = shapely.geometry.Polygon(new_geom.exterior)
            
                new_geom = shapely.affinity.translate(new_geom, xoff=sl[1].start, yoff=sl[0].start)
                new_geometries[orig_idx] = new_geom

    # Update shapes on the SpatialData object
    print(f"Saving the '{output_shapes_key}' layer in the SpatialData object...")
    
    nuclei_shapes = gpd.GeoDataFrame(shapes.copy(), geometry=new_geometries, crs=shapes.crs)
    if hasattr(shapes, "attrs"): nuclei_shapes.attrs = shapes.attrs.copy()

    dict_shapes = dict(sdata.shapes)
    dict_shapes[output_shapes_key] = nuclei_shapes                                  

    new_sdata = sd.SpatialData(
        images=dict(sdata.images),
        shapes=dict_shapes,
        points=dict(sdata.points) if hasattr(sdata, "points") else {},
        labels=dict(sdata.labels) if hasattr(sdata, "labels") else {},
        tables=dict(sdata.tables) if hasattr(sdata, "tables") else {}
    )
    
    print(f"Extraction completed. Intact/complete cells: {stats['intact_nuclei']} | Nuclei refined by DAPI: {stats['refined_nuclei']}")
    return new_sdata

def _process_individual_cell(
    id_cell,
    polygon,
    image_np,
    k_min,
    k_max,
    channel_indices,
    log_transform=False,
    standardize_per_cell=True,
):
    """
    Optimized worker function for joblib with
    integrated transformations.
    """
    import numpy as np
    import rasterio
    from sklearn.feature_extraction.image import grid_to_graph
    from sklearn.cluster import AgglomerativeClustering

    try:
        # Get the limits for the polygon's bounding box
        minx, miny, maxx, maxy = polygon.bounds

        # Safety check to ensure staying within the image
        minx_idx = max(0, int(np.floor(minx)))
        miny_idx = max(0, int(np.floor(miny)))
        maxx_idx = min(image_np.shape[2], int(np.ceil(maxx)))
        maxy_idx = min(image_np.shape[1], int(np.ceil(maxy)))

        # Create a translation to bring the origin to the bounding box
        local_translation = rasterio.Affine(
            1.0, 0.0, float(minx_idx), 0.0, 1.0, float(miny_idx)
        )

        local_height = maxy_idx - miny_idx
        local_width = maxx_idx - minx_idx

        # Create a mask just for the bounding box
        local_mask = rasterio.features.geometry_mask(
            [polygon],
            out_shape=(local_height, local_width),
            transform=local_translation,
            invert=True,
        )

        local_indices = np.argwhere(local_mask)
        n_pixels = len(local_indices)

        if n_pixels < k_min:
            return None

        # Convert local indices to global indices on the original image
        global_indices = local_indices + np.array([miny_idx, minx_idx])

        # Extract intensities
        sub_image = image_np[:, miny_idx:maxy_idx, minx_idx:maxx_idx]
        intensities = sub_image[:, local_mask].T  # Shape: (n_pixels, n_selected_channels)

        if log_transform:
            intensities = np.log1p(intensities)

        # Z-score scaling per cell, to ensure comparability between variables
        if standardize_per_cell:
            means = np.mean(intensities, axis=0)
            std_devs = np.std(intensities, axis=0)
            std_devs[std_devs == 0] = 1.0 # Safety check
            intensities = (intensities - means) / std_devs

        # Build the adjacency graph
        connectivity = grid_to_graph(
            n_x=local_mask.shape[0],
            n_y=local_mask.shape[1],
            mask=local_mask,
        )

        # Safety adjustment for maximum k
        actual_k_max = min(k_max, n_pixels - 1)
        if actual_k_max < k_min:
            return global_indices, np.ones(n_pixels, dtype=np.int32)

        # Fit the complete tree to calculate distances (Auto-K)
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0,
            connectivity=connectivity,
            linkage="ward",
            compute_distances=True,
        )
        model.fit(intensities)

        # Find optimal k using the Maximum Lifespan Criterion
        lifespans = []
        k_values = list(range(k_min, actual_k_max + 1))

        for k in k_values:
            lifespan = model.distances_[-(k - 1)] - model.distances_[-k]
            lifespans.append(lifespan)

        optimal_k = k_values[np.argmax(lifespans)]

        # Final clustering with the optimal k
        final_model = AgglomerativeClustering(
            n_clusters=optimal_k, connectivity=connectivity, linkage="ward"
        )
        zone_labels = final_model.fit_predict(intensities)

        # Global unique label encoding
        unique_labels = (id_cell * 100) + (zone_labels + 1)

        return global_indices, unique_labels

    except Exception as e:
        print(f"Error in cell {id_cell}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
def calculate_subcellular_regions(
    sdata,
    image_key=None,
    shapes_key=None,
    output_key="subcellular_regions",
    filtered_channels=None,  
    log_transform=False, 
    standardize_per_cell=True,  
    k_min=2,
    k_max=10,
    n_jobs=-1
    ):

    """
    Executes aglomerative connected clustering, allowing channel selection,
    logarithmic transformation, and local normalization for subcellular phenotyping.
    """

    import numpy as np
    import rasterio
    from joblib import Parallel, delayed
    from tqdm import tqdm
    from spatialdata.models import Labels2DModel
    from spatialdata.transformations import get_transformation

    if image_key is None:
        image_key = list(sdata.images.keys())[0]
    if shapes_key is None:
        shapes_key = list(sdata.shapes.keys())[0]

    image_element = sdata.images[image_key]
    shapes_element = sdata.shapes[shapes_key]

    # Unpack multiscale in a structured manner
    if hasattr(image_element, "keys") and "scale0" in image_element:
        variable_name = list(image_element["scale0"].data_vars)[0]
        image_dataarray = image_element["scale0"][variable_name]
    elif hasattr(image_element, "data_vars"):
        variable_name = list(image_element.data_vars)[0]
        image_dataarray = image_element[variable_name]
    else:
        image_dataarray = image_element

    # Search for the channels in the image
    if "c" in image_dataarray.coords:
        total_channel_names = list(image_dataarray.coords["c"].values)
    elif hasattr(image_dataarray, "dataset") and "c" in image_dataarray.dataset.coords:
        total_channel_names = list(image_dataarray.dataset.coords["c"].values)
    else:
        total_channel_names = [str(i) for i in range(image_dataarray.shape[0])]

    # Map requested channels to numerical positions
    if filtered_channels is None:
        selected_channels = total_channel_names
        print(f"Using all available channels ({len(selected_channels)})")
    else:
        selected_channels = [
            c for c in filtered_channels if c in total_channel_names
        ]
        invalid_channels = set(filtered_channels) - set(total_channel_names)
        if invalid_channels:
            print(f"Warning: These channels do not exist and will be ignored: {invalid_channels}")
        print(f"Selected channels for clustering ({len(selected_channels)}): {selected_channels}")

    channel_indices = [
        total_channel_names.index(c) for c in selected_channels
    ]

    # Get the actual dimensions of the image
    if "y" in image_dataarray.dims and "x" in image_dataarray.dims:
        height_y = image_dataarray.shape[image_dataarray.dims.index("y")]
        width_x = image_dataarray.shape[image_dataarray.dims.index("x")]
    else:
        height_y, width_x = image_dataarray.shape[-2], image_dataarray.shape[-1]

    print("\nLoading image NumPy...")
    if hasattr(image_dataarray, "transpose"):
        image_dataarray = image_dataarray.transpose('c', 'y', 'x')
    
    if filtered_channels:
        # Cross-reference to ensure we don't look up non-existent channels
        available_channels = image_dataarray.coords['c'].values
        valid_channels = [ch for ch in filtered_channels if ch in available_channels]
        
        if valid_channels:
            image_dataarray = image_dataarray.sel(c=valid_channels)
            
    # If the image is trapped in an oversized uint64 format, downcast it
    import numpy as np
    if image_dataarray.dtype == np.uint64:
        # float32 is safer if you plan to do log transforms or scaling next
        image_dataarray = image_dataarray.astype(np.float32)

    raw_numpy_image = image_dataarray.compute().values
    global_mask = np.zeros((height_y, width_x), dtype=np.int32)
    rasterio_transform = rasterio.transform.from_origin(0, 0, 1, 1)

    polygons = shapes_element.geometry
    cell_ids = shapes_element.index

    print(f"Processing {len(polygons)} cells in parallel with {n_jobs} threads...")
    # Passing normalization flags and mapped index vector to the threads
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_process_individual_cell)(
            idx,
            poly,
            raw_numpy_image,
            k_min,
            k_max,
            channel_indices,
            log_transform=log_transform,
            standardize_per_cell=standardize_per_cell,
        )
        for idx, (idx_cell, poly) in enumerate(
            tqdm(
                zip(cell_ids, polygons),
                total=len(polygons),
                desc="Subcellular segmenting",
            )
        )
    )

    print("\nAssembling global labels mask...")
    completed_cells = 0
    for res in results:
        if res is not None:
            indices, labels = res
            global_mask[indices[:, 0], indices[:, 1]] = labels
            completed_cells += 1

    print(f"Mask generated. {completed_cells} of {len(polygons)} cells segmented into micro-regions.")

    print("Saving to SpatialData object...")
    transformations_dictionary = get_transformation(image_element, get_all=True)

    encapsulated_labels = Labels2DModel.parse(
        global_mask,
        dims=("y", "x"),
        transformations=transformations_dictionary,
    )

    sdata.labels[output_key] = encapsulated_labels
    print(f"Labels layer '{output_key}' successfully added.")
    return sdata

def extract_regional_intensities(
    sdata,
    image_key=None,
    labels_key=None,
    output_table_key="table_subcellular_clusters",
    aggregation_mode = "mean",
    clipping_percentile=0.99
):
    """
    Extracts the mean intensities of each marker for each mask,
    applying an upper percentile-based clip, and saves the result 
    natively as an indexed table inside the SpatialData object.
    """

    import numpy as np
    import pandas as pd
    import anndata as ad
    from spatialdata.models import TableModel

    if image_key is None:
        image_key = list(sdata.images.keys())[0]
    if labels_key is None:
        labels_key = list(sdata.labels.keys())[0]
    image_element = sdata.images[image_key]

    if hasattr(image_element, "keys") and "scale0" in image_element:
        # Extract the maximum resolution DataArray
        variable_name = list(image_element["scale0"].data_vars)[0]
        image_dataarray = image_element["scale0"][variable_name]
    else:
        image_dataarray = image_element

    print("Extracting image and mask matrices...")
    # Read values and coordinates directly from the extracted DataArray
    image_np = image_dataarray.compute().values
    masks_np = np.array(sdata.labels[labels_key].values)
    channel_names = image_dataarray.coords["c"].values
    
    # Flatten the matrices (Ravel) and filter the background
    flat_masks = masks_np.ravel()
    valid_pixels = flat_masks > 0
    valid_labels = flat_masks[valid_pixels]
    
    # Now that image_np has a true shape of (C, Y, X), flatten it across Y and X axes
    flattened_image = image_np.reshape(image_np.shape[0], -1)
    
    # Filter channels based on pixels that belong to a cell
    valid_intensities = flattened_image[:, valid_pixels].T

    # Create a Pandas DataFrame
    df_pixels = pd.DataFrame(valid_intensities, columns=channel_names)
    df_pixels["Label_ID"] = valid_labels

    # Clipping (Outlier removal)
    print(f"Applying clipping to the top {clipping_percentile*100}%...")
    upper_limits = df_pixels[channel_names].quantile(clipping_percentile)
    df_pixels[channel_names] = df_pixels[channel_names].clip(
        upper=upper_limits, axis=1
    )

    # Grouping and aggregating means per mask (Region)
    print("Calculating means per zone...")
    if aggregation_mode == "mean":
        annotation_table = df_pixels.groupby("Label_ID").mean().reset_index()
    elif aggregation_mode == "median":
        annotation_table = df_pixels.groupby("Label_ID").mean().reset_index()

    # Unpack Label_ID to recover original parentage
    annotation_table["Cell_ID"] = annotation_table["Label_ID"] // 100
    annotation_table["Zone_ID"] = annotation_table["Label_ID"] % 100

    print("Structuring AnnData object for SpatialData...")
    # Separate the expression matrix (X) from the metadata (obs)
    # Convert Label_ID to string to use as a unique index (obs_names)
    annotation_table["Label_ID_str"] = annotation_table["Label_ID"].astype(str)
    annotation_table.set_index("Label_ID_str", inplace=True)

    X_matrix = annotation_table[channel_names].values.astype(np.float32)
    df_obs = annotation_table[["Cell_ID", "Label_ID", "Zone_ID"]].copy()

    # Ensure data types are compatible with AnnData/Zarr
    df_obs["Cell_ID"] = df_obs["Cell_ID"].astype(np.int64)
    df_obs["Label_ID"] = df_obs["Label_ID"].astype(np.int64)
    df_obs["Zone_ID"] = df_obs["Zone_ID"].astype(np.int32)
    df_obs["region"] = labels_key
    df_obs["region"] = df_obs["region"].astype("category")


    # Create the basic AnnData object
    adata = ad.AnnData(
        X=X_matrix, obs=df_obs, var=pd.DataFrame(index=channel_names)
        )

    # Parse with the SpatialData TableModel to link it to the masks
    adata_spatial = TableModel.parse(
        adata,
        region=[labels_key],  
        region_key="region", 
        instance_key="Label_ID"
        )

    # Save to the sdata object
    print(f"Saving table to sdata.tables['{output_table_key}']...")
    sdata.tables[output_table_key] = adata_spatial

    print(f"Regional intensities table linked with {len(annotation_table)} regions.")

    return sdata

def plot_density_w_gates(adata, vars_list, gates, gates_are_logarithmic = True, figsize_unit=(4, 4)):
    """
    Given an AnnData object, a list of markers and a DataFrame with the thresholds, this function
    plots the distributions of the markers with their positivity thresholds.
    """

    import math 
    import seaborn as sns 
    import matplotlib.pyplot as plt
    import numpy as np

    # Subplots configuration
    n_vars = len(vars_list)
    n_cols = 4
    n_rows = math.ceil(n_vars / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, 
                             figsize=(figsize_unit[0] * n_cols, figsize_unit[1] * n_rows))
    
    axes_flat = axes.flatten()
    gates_df = gates.copy()

    if gates_are_logarithmic:
        gates_df["gates"] = np.expm1(gates_df["gates"])

    for i, var in enumerate(vars_list):
        ax = axes_flat[i]
        
        # Data extraction from adata.X
        if var in adata.var_names:
            data = adata[:, var].X
            data = data.flatten()
            data = np.log1p(data)
        # If the column isn't there, check in adata.obs
        elif var in adata.obs.columns:
            data = adata.obs[var].values
        else:
            print(f"{var} marker not found.")
            continue

        # Density graph (KDE)
        sns.kdeplot(data[data <= np.percentile(data, 99.5)], ax=ax, fill=True, color='steelblue', alpha=0.5)
        
        # Vertical line in the threshold
        try:
            if 'markers' in gates_df.columns:
                thresh = gates_df.loc[gates_df['markers'] == var, 'gates'].item()
            else:
                thresh = np.log1p(gates_df.loc[var, 'gates'])
            ax.axvline(x=thresh, color='red', linestyle='--', linewidth=2, label=f'Threshold: {thresh:.2f}')
        except Exception as e:
            print(f"No gate given for {var}.")

        # Axes and title of the subplot
        ax.set_title(f"{var} density", fontsize=14, fontweight='bold')
        ax.set_xlabel("Marker intensity")
        ax.set_ylabel("Density")
        ax.legend()

    # Deleting extra axes
    for j in range(i + 1, len(axes_flat)):
        fig.delaxes(axes_flat[j])
    return fig

def calculate_gates_gmm_bic(adata, markers, gmm_components="auto",
                             max_k=3,
                             unimodal_sigma_coefficient = 1.5,
                             verbose=False):
    """
    Optimized Version with Cascading Biological Separation Filter:
    1. Evaluates k=1 up to k=max_k dynamically via BIC.
    2. Sorts models from best to worst according to their BIC.
    3. Evaluates the winner: if k>1, verifies that the distance between components is real.
    4. If separation fails, moves to the next best model according to BIC.
    5. If no multimodal complies, safely falls back to k=1.
    """
    
    import numpy as np
    from sklearn.mixture import GaussianMixture
    import pandas as pd

    new_gates = {}

    for var in markers:
        # Data preparation
        data = adata[:, var].X.flatten() if var in adata.var_names else adata.obs[var].values.flatten()
        data = np.log1p(data[data > 0])
        data_reshaped = data.reshape(-1, 1)
        x_range = np.linspace(data.min(), data.max(), 1000)

        prob_req = 0.5

        # SCOUT: Dynamic Model Training and BIC Calculation up to max_k
        gmm_dict = {}
        bic_dict = {}
        
        for k_val in range(1, max_k + 1):
            # k=1 does not benefit from multiple initializations, multi-component models do
            n_init_val = 1 if k_val == 1 else 10
            gmm = GaussianMixture(n_components=k_val, n_init=n_init_val, random_state=42, max_iter=200).fit(data_reshaped)
            gmm_dict[k_val] = gmm
            bic_dict[k_val] = gmm.bic(data_reshaped)

        if gmm_components == "auto":
            # Sort k values based on their BIC (from lowest/best to highest/worst)
            sorted_k = sorted(bic_dict.keys(), key=lambda k: bic_dict[k])
        else:
            sorted_k = [gmm_components] if isinstance(gmm_components, int) else list(gmm_components)

        # Cascading separation criterion
        best_k = None
        best_gmm = None
        
        for k in sorted_k:
            candidate_gmm = gmm_dict[k]
            m_cand = candidate_gmm.means_.flatten()
            s_cand = np.sqrt(candidate_gmm.covariances_.flatten())
            idx_cand = np.argsort(m_cand)
            
            if k == 1:
                # If k=1 is the current turn, it is automatically accepted without checking
                best_k = 1
                best_gmm = candidate_gmm
                break
            else:
                m_noise = m_cand[idx_cand[-2]]   
                m_pos = m_cand[idx_cand[-1]]   
                s_noise = s_cand[idx_cand[-2]]  
                s_pos = s_cand[idx_cand[-1]] 
                
                # Separation criterion: Ashman's D
                D = np.sqrt(2) * (np.abs(m_pos - m_noise)) / (np.sqrt(s_noise**2 + s_pos**2))
                if D >= 2:
                    # Meets criteria, keep this k
                    best_k = k
                    best_gmm = candidate_gmm
                    break
                # If it doesn't meet criteria, the loop continues evaluating the next best k
                
        if best_k is None:
            best_k = 1
            best_gmm = gmm_dict[1]

        # Extract final parameters of the winning model
        m_scout = best_gmm.means_.flatten()
        s_scout = np.sqrt(best_gmm.covariances_.flatten())
        w_scout = best_gmm.weights_.flatten()
        idx_scout = np.argsort(m_scout)

        # Calculate the gate
        if best_k == 1:
            # Outlier logic for negative / unimodal distributions
            multiplier = unimodal_sigma_coefficient
            target_x = m_scout[0] + (multiplier * s_scout[0])
        else:
            resp_scout = best_gmm.predict_proba(x_range.reshape(-1, 1))
            prob_pos = resp_scout[:, idx_scout[-1]]
            crossing_points = np.where(np.diff(np.sign(prob_pos - prob_req)))[0]
            
            m1 = m_scout[idx_scout[-2]]
            m_last = m_scout[idx_scout[-1]]
            s1 = s_scout[idx_scout[-2]]

            if len(crossing_points) > 0:
                valid_crossings = [x_range[c] for c in crossing_points if m1 < x_range[c] < m_last]
                target_x = valid_crossings[0] if valid_crossings else m1 + 1.5 * s1
            else:
                target_x = m1 + 2 * s1

        new_gates[var] = target_x

        # Visualization
        if verbose:
            import matplotlib.pyplot as plt
            from scipy.stats import gaussian_kde, norm

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            kde_real = gaussian_kde(data)(x_range)
            
            # Left Panel: Original BIC competition
            ax1.plot(x_range, kde_real, color='black', lw=2, label='Real Density (KDE)')
            ax1.fill_between(x_range, 0, kde_real, color='black', alpha=0.05)
            
            # Dynamic color cycling palette for competitive GMM models
            colors = ['forestgreen', 'purple', 'dodgerblue', 'darkorange', 'magenta', 'crimson', 'sienna']
            for i, k_val in enumerate(range(1, max_k + 1)):
                col_idx = i % len(colors)
                ax1.plot(
                    x_range, 
                    np.exp(gmm_dict[k_val].score_samples(x_range.reshape(-1, 1))), 
                    '--', 
                    color=colors[col_idx], 
                    label=f'k={k_val} (BIC: {bic_dict[k_val]:.0f})'
                )
                
            ax1.axvline(target_x, color='red', linestyle='-', lw=2, label=f'Gate: {target_x:.2f}')
            
            # Notice if there was a rejection
            original_winner = sorted_k[0] if gmm_components == "auto" else best_k
            title_suffix = f" (Original k={original_winner} rejected)" if best_k != original_winner else ""
            ax1.set_title(f"Scout Selection: Final winner k={best_k}{title_suffix}")
            ax1.set_xlabel("Log Intensity")
            ax1.set_ylabel("Density")
            ax1.legend(loc='upper right', fontsize='small')
            
            # Right Panel: Final applied components
            ax2.plot(x_range, kde_real, color='black', lw=2, label='Real Density (KDE)')
            ax2.fill_between(x_range, 0, kde_real, color='black', alpha=0.05)

            for idx in idx_scout:
                color = 'blue' if best_k == 1 else ('green' if idx == idx_scout[-1] else 'blue')
                ax2.plot(x_range, w_scout[idx] * norm.pdf(x_range, m_scout[idx], s_scout[idx]), '--', color=color, alpha=0.6)

            gate_label = f'Gate (Outliers {multiplier}σ): {target_x:.2f}' if best_k == 1 else f'Gate: {target_x:.2f}'
            ax2.axvline(target_x, color='red', lw=2, label=gate_label)
            ax2.set_title(f"Active Final Components Analysis (k={best_k})")
            ax2.set_xlabel("Log Intensity")
            ax2.legend(loc='upper right', fontsize='small')
            
            plt.suptitle(f"Marker: {var}", fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()

    return pd.DataFrame.from_dict(new_gates, orient='index', columns=['gates'])

def phenotyping_on_leiden(adata, phenotype, gates, resolution=1.5, n_pca= 50, n_neighbors=5,
                         phenotype_column='phenotype_subtipo', verbose=False, name=None):
    
    """
    Executes phenotyping in an adata stage according to the phenotype table.
    Runs Leiden clustering with the given parameters and phenotypes the 
    centroids of each cluster. Returns the phenotypes in output_column, 
    and if gates are not provided, uses KDE thresholding (calculate_gates).
    """

    import numpy as np
    import scanpy as sc
    import pandas as pd
    import anndata as ad

    from rescale_mod import rescale
    from phenotype_cells_mod import phenotype_cells

    # Selection of adata variables needed for phenotyping
    clustering_variables = [col for col in phenotype.columns[2:] if ("Unnamed" not in col) and (col in adata.var_names)]
    adata_aux = adata[:, clustering_variables]

    # Data preprocessing and filtering
    mask = ~np.isnan(adata_aux.X).any(axis=1)
    adata_aux = adata_aux[mask, :].copy()
    if adata_aux.raw is not None:
        del adata_aux.raw
    adata_aux.X[np.isnan(adata_aux.X)] = 0
    adata_aux.X[np.isinf(adata_aux.X)] = 0
    adata_aux.X = np.clip(adata_aux.X, a_min=0, a_max=None)

    # Leiden Clustering: neighbor graph and Leiden
    sc.pp.neighbors(adata_aux, n_neighbors=n_neighbors, use_rep='X')

    clustering_column = f'clustering_res{resolution}_knn{n_neighbors}_pca{n_pca}'

    sc.tl.leiden(
        adata_aux, 
        resolution=resolution, 
        key_added=clustering_column, 
        flavor='igraph',
        n_iterations=2,
        directed=False
        )
            
    adata.obs[clustering_column] = adata_aux.obs[clustering_column]

    if verbose: 
        if name == None :
            print(f"\n{len(adata.obs[clustering_column].unique())} clusters generated")
        else: 
            print(f"\n{len(adata.obs[clustering_column].unique())} clusters generated for {name}")

    # Cluster phenotyping: threshold generation, cluster centroid generation,
    # cluster labeling, and label induction

    adata_aux.obs['imageid'] = 'placeholder'

    # Variables are rescaled so the gate goes to 0.5 and everything is between 0 and 1. 
    adata_aux = rescale(adata_aux,
                        gate=gates, 
                        verbose=False)

    adata_aux.uns['gates'] = gates

    # Extract the already rescaled adata_aux data and save the mean per cluster
    large_rescaled_df = pd.DataFrame(adata_aux.X, index=adata_aux.obs_names, columns=adata_aux.var_names)
    large_rescaled_df[clustering_column] = adata_aux.obs[clustering_column].values
    cluster_mean_matrix = large_rescaled_df.groupby(clustering_column, observed=False).mean()

    # Convert the mean intensity data per cluster to AnnData format
    adata_clusters = ad.AnnData(X=cluster_mean_matrix.values)
    adata_clusters.obs_names = cluster_mean_matrix.index.astype(str)
    adata_clusters.var_names = cluster_mean_matrix.columns.astype(str)

    # Apply the phenotyping table to the cluster data
    adata_clusters, cluster_scores = phenotype_cells(
        adata_clusters, 
        phenotype=phenotype,
        label=phenotype_column,
        return_score=True,  
        verbose=False            
    )

    # Create a dictionary relating cluster and phenotype
    adata_clusters.obs[phenotype_column] = adata_clusters.obs[phenotype_column].astype(str).str.replace("likely-", "")
    cluster_phenotype_map = dict(zip(adata_clusters.obs.index, adata_clusters.obs[phenotype_column]))
    
    # Dictionary to relate the cluster with its score
    cluster_score_map = dict(zip(adata_clusters.obs.index, cluster_scores.values))

    # Map this dictionary to the original large adata based on its cluster.
    adata.obs[phenotype_column] = adata.obs[clustering_column].astype(str).map(cluster_phenotype_map)
    
    # Map the node score to each cell
    adata.obs["node_score"] = adata.obs[clustering_column].astype(str).map(cluster_score_map)

    parent_lineage = phenotype.iloc[0, 0]
    fallback_label = f"Undefined_{parent_lineage}"

    # Cells that Scimap could not classify ('Unknown') are renamed
    adata.obs[phenotype_column] = adata.obs[phenotype_column].replace('Unknown', fallback_label)
    adata.obs[phenotype_column] = adata.obs[phenotype_column].fillna(fallback_label)

    return adata

def RF_Undefined(adata, 
                 scores_df=None, 
                 phenotype_column="base_HSC_phenotype", 
                 output_column="phenotype_HSC", 
                 RF_majority_column=None, 
                 class_weight = "balanced_subsample",
                 n_folds=10, n_estimators=100, verbose=False):
    """
    Ensemble-based relabeling of undefined cells using Random Forest cross-validation.

    Trains an ensemble of Random Forest classifiers on explicitly labeled cells across stratified 
    folds, then predicts and assigns consensus phenotypes to cells flagged as 'Undefined'. 
    Optionally updates an external score matrix with the model's mean classification probabilities.
    """

    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    from scipy import stats

    mask_undefined = adata.obs[phenotype_column].str.contains("Undefined_", na=False)
    
    if not mask_undefined.any():
        if verbose:
            print("There are no Undefined cells to relabel")
        adata.obs[output_column] = adata.obs[phenotype_column]
        return adata

    X_train = adata[~mask_undefined].X
    y_train = adata[~mask_undefined].obs[phenotype_column].values
    X_predict = adata[mask_undefined].X

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    all_predictions = []
    all_probabilities = [] 

    if verbose:
        print(f"Training Random Forest ({n_folds} folds) to relabel Undefined cells.")

    for train_index, _ in kf.split(X_train, y_train):
        X_fold, y_fold = X_train[train_index], y_train[train_index]
        
        rf_model = RandomForestClassifier(
                        n_estimators=n_estimators, 
                        random_state=42, 
                        n_jobs=-1,
                        max_depth=20,
                        min_samples_leaf=5,
                        class_weight=class_weight)
        rf_model.fit(X_fold, y_fold)
        
        # Predictions and probabilities
        fold_preds = rf_model.predict(X_predict)
        fold_probs = rf_model.predict_proba(X_predict) 
        
        all_predictions.append(fold_preds)
        all_probabilities.append(fold_probs)     

    # Random Forest consensus for the labels 
    df_preds = pd.DataFrame(np.array(all_predictions).T, index=adata.obs[mask_undefined].index)
    
    # Create a unanimity mask
    unanimity_mask = (df_preds.nunique(axis=1) == 1)
    
    # Extract the unanimous predictions
    unanimous_predictions = df_preds.loc[unanimity_mask, 0]
    
    # Assign the final phenotype
    adata.obs[output_column] = adata.obs[phenotype_column].copy()
    
    # Only update cells that reached unanimity
    if not unanimous_predictions.empty:
        adata.obs.loc[unanimous_predictions.index, output_column] = unanimous_predictions
    
    if RF_majority_column:
        adata.obs[RF_majority_column] = adata.obs[phenotype_column].copy()
        if not unanimous_predictions.empty:
            adata.obs.loc[unanimous_predictions.index, RF_majority_column] = unanimous_predictions

    if scores_df is not None:
        if verbose:
            print("Updating the score matrix with the Random Forest probabilities...")
            
        mean_probs = np.mean(all_probabilities, axis=0)
        rf_labels = rf_model.classes_
        
        # Overwrite old scores for each class seen by the model
        for i, label in enumerate(rf_labels):
            if label in scores_df.columns:
                scores_df.loc[mask_undefined, label] = mean_probs[:, i]

    if verbose:
        relabeled_count = unanimity_mask.sum()
        left_undefined_count = (~unanimity_mask).sum()
        print(f"{relabeled_count} cells have been successfully relabeled by unanimous consensus.")
        print(f"{left_undefined_count} cells remain unchanged due to lack of unanimity.")
        
    return adata

def subcellular_HSC_phenotyping(adata, phenotype_table, leiden=True, final_column = "subcellular_phenotype", user_gates = {}, 
                                verbose = False, plot_distributions = False):
    
    """
    Based on the mean intensity data (in anndata format) and the phenotyping table,
    performs successive labeling using scimap. 
    
    If leiden=True: uses Leiden clustering at each node of the decision tree.
    If leiden=False: bypasses Leiden and evaluates intensities on a cell-by-cell basis.
    
    Thresholds are calculated using a GMM with an adaptive number of components based on BIC.

    The remaining Undefined cells are labeled using RF_n_folds Random Forests under 
    the condition that the classification must be unanimous. Otherwise, the Undefined 
    label is retained.
    """

    import warnings
    warnings.filterwarnings('ignore')
    import numpy as np
    import pandas as pd
    import anndata as ad
    import matplotlib.pyplot as plt
    from rescale_mod import rescale
    from phenotype_cells_mod import phenotype_cells

    # Internal variables
    pre_filtering_column = "base_phenotype"
    preRF_column = "filtered_phenotype"
    RF_majority_column = "RF_majority_phenotype"

    # Data loading (if necessary)
    if type(adata) == str:
        adata = ad.read_h5ad(adata)
    if type(phenotype_table) == str:
        phenotype_table = pd.read_csv(phenotype_table)

    phenotype_table.dropna(axis = 1, how = "all", inplace= True)
    markers = list(phenotype_table.columns[2:])

    # Gates are defined using the whole population
    # They will be used for the marker positivity columns
    markers_to_calculate = markers

    if isinstance(user_gates, pd.DataFrame):
        user_gates_dict = user_gates.iloc[:, 0].to_dict()
    elif isinstance(user_gates, dict):
        user_gates_dict = user_gates
    else:
        user_gates_dict = {}

    gates_dict = {}

    if user_gates_dict:
        # Load the custom gates provided by the user
        for m in markers:
            if m in user_gates_dict:
                gates_dict[m] = user_gates_dict[m]
        
        # Filter out markers that already have custom gates
        markers_to_calculate = [m for m in markers if m not in user_gates_dict]
        
        if verbose and len(gates_dict) > 0:
            print(f"Using provided custom global gates for: {list(gates_dict.keys())}")

    # Calculate GMM gates only for the markers missing from custom_gates
    if len(markers_to_calculate) > 0:
        auto_gates_df = calculate_gates_gmm_bic(
            adata, 
            gmm_components="auto", 
            markers=markers_to_calculate, 
            verbose=False
        )
        gates_dict.update(auto_gates_df.iloc[:, 0].to_dict())

    # Format the global gates object back to the standard DataFrame output format
    gates = pd.DataFrame.from_dict(gates_dict, orient='index', columns=['gates'])

    for marker in [col for col in phenotype_table.columns[2:] if "Unnamed" not in col and col in adata.var_names]:              
        table_marker = pd.DataFrame({
            "Input_node": ["All", "All"], 
            "Output_node": [f"positive_{marker}", f"negative_{marker}"], 
            f"{marker}" : ["pos", "neg"]
            })
                        
        adata_marker = adata[:, [marker]].copy()
        if "imageid" not in adata_marker.obs:
            adata_marker.obs["imageid"] = "placeholder"
        rescale(adata_marker, gate = gates, verbose= False)

        phenotype_cells(adata_marker, table_marker, label = f"{marker}_positivity", verbose = False)

        adata.obs[f"{marker}_positivity"] = adata_marker.obs[f"{marker}_positivity"].astype('category')

    if plot_distributions:
        plot_density_w_gates(adata, 
            [col for col in phenotype_table.columns[2:] if "Unnamed" not in col and col in adata.var_names], 
            gates, gates_are_logarithmic = True, figsize_unit=(4, 4))
        plt.show()

    # Initializing variables to loop through the decision tree
    input_node  = phenotype_table.columns[0]
    output_node = phenotype_table.columns[1]
    
    for col in phenotype_table.columns:
        if phenotype_table[col].isna().all(): phenotype_table.drop(columns=col, inplace=True)

    starting_nodes = ["All"]
    end_loop = False
    adata.obs["phenotype_level0"] = "All"
    phenotype_level = 0

    cumulative_scores_df = pd.DataFrame(index=adata.obs.index)
    cumulative_scores_df["All"] = 1.0

    parent_nodes = phenotype_table[input_node].unique()

    while not end_loop:
        nodes_to_process = [n for n in starting_nodes if n in parent_nodes]
        if len(nodes_to_process) == 0:
            end_loop = True
            break

        adata.obs[f"phenotype_level{phenotype_level + 1}"] = np.nan

        #Building the reduced table for this phenotype level
        phenotype_phase = phenotype_table[phenotype_table[input_node].isin(starting_nodes)]
        for node in starting_nodes:

            mask_node = adata.obs[f"phenotype_level{phenotype_level}"] == node
            if not mask_node.any(): continue

            if node in parent_nodes:

                # For each parent node, the corresponding sub-table is used to phenotype
                phenotype_node = phenotype_phase[phenotype_phase[input_node]== node]
                phenotype_node = phenotype_node.dropna(axis = 1, how = "all")

                adata_node = adata[adata.obs[f"phenotype_level{phenotype_level}"] == node].copy()
                
                if True:
                    # The gates are generated only using the elements (generally cells) that are already at the parent node
                    if verbose: 
                        print(f"\nCalculating gates for {node}:")
                   
                    node_markers = [m for m in phenotype_node.columns[2:]]
                    node_gates_dict = {}
                    node_markers_to_calculate = []

                    
                    # Prioritize the user's gates
                    for m in node_markers:
                        if m in user_gates_dict.keys():
                            node_gates_dict[m] = user_gates_dict[m]
                        else:
                            node_markers_to_calculate.append(m)

                    if verbose and len(node_gates_dict) > 0:
                        print(f"Bypassing GMM for {node}: Using user's gates for {list(node_gates_dict.keys())}")

                    # Calculate local node gates only for markers the user didn't explicitly provide
                    if len(node_markers_to_calculate) > 0:
                        if verbose: 
                            print(f"\nCalculating local GMM gates for {node} on markers: {node_markers_to_calculate}")
                        
                        auto_node_gates_df = calculate_gates_gmm_bic(
                            adata_node, 
                            gmm_components="auto", 
                            markers=node_markers_to_calculate,
                            verbose=verbose
                        )
                        node_gates_dict.update(auto_node_gates_df.iloc[:, 0].to_dict())

                    # Convert node gates dictionary to standard DataFrame format
                    node_gates = pd.DataFrame.from_dict(node_gates_dict, orient='index', columns=['gates'])
                    
                    if plot_distributions:
                        plot_density_w_gates(adata_node, phenotype_node.columns[2:], node_gates, gates_are_logarithmic = True, figsize_unit=(4, 4))
                        plt.show()
                
                if leiden:
                    # Capture node_scores
                    adata_node, node_scores = phenotyping_on_leiden(
                                    adata_node,
                                    phenotype_node,
                                    gates=node_gates,
                                    phenotype_column = f"phenotype_level{phenotype_level+1}", 
                                    verbose = verbose,
                                    n_neighbors = 3,
                                    resolution = 1,
                                    name = node
                                    )
                else:
                    if verbose: print(f"Phenotyping on node {node}")
                    
                    adata_subset = adata_node[:, node_markers].copy()
                    
                    if "imageid" not in adata_subset.obs:
                        adata_subset.obs["imageid"] = "placeholder"
                        
                    rescale(adata_subset, gate=node_gates, verbose=False)
                
                    adata_subset, node_scores = phenotype_cells(
                        adata_subset, 
                        phenotype_node, 
                        label=f"phenotype_level{phenotype_level+1}", 
                        return_score=True, 
                        verbose=False
                    )
                    
                    adata_subset.obs[f"phenotype_level{phenotype_level+1}"] = adata_subset.obs[f"phenotype_level{phenotype_level+1}"].astype(str).replace("Unknown", f"Undefined_{node}")
                    adata_node.obs[f"phenotype_level{phenotype_level+1}"] = adata_subset.obs[f"phenotype_level{phenotype_level+1}"]

                # Results are written on the corresponding column of the original adata
                adata.obs.loc[mask_node, f'phenotype_level{phenotype_level +1}'] = adata_node.obs[f"phenotype_level{phenotype_level+1}"]

                for col in node_scores.columns:

                    if phenotype_level == 0:
                        # Level 1: The cumulative score is the score of the node
                        cumulative_scores_df.loc[mask_node, col] = node_scores.loc[mask_node, col].values
                    else:
                        # Level > 1: The cumulative score is the mean of all the edges the element has gone through
                        parent_score = cumulative_scores_df.loc[mask_node, node].values
                        new_score = node_scores.loc[mask_node, col].values
                        
                        updated_average = ((parent_score * phenotype_level) + new_score) / (phenotype_level + 1)
                        cumulative_scores_df.loc[mask_node, col] = updated_average
            else: 
                # If the node is childless, the phenotypes are just copied to the next phenotyping level
                adata.obs.loc[mask_node, f'phenotype_level{phenotype_level +1}'] = node

        # Update internal variables and print a value count
        phenotype_level += 1

        starting_nodes = adata.obs[f'phenotype_level{phenotype_level}'].dropna().unique()

        if verbose: 
            print(f"\nPhenotypes on level {phenotype_level} : ")
            print(adata.obs[f"phenotype_level{phenotype_level}"].value_counts(dropna=False))
            print("-" * 30)

    # Define the needed columns
    adata.obs[pre_filtering_column] = adata.obs[f'phenotype_level{phenotype_level}']
    adata.obs[preRF_column] = adata.obs[pre_filtering_column].copy()

    # Apply Random Forests to relabel the Undefined cells 
    adata = RF_Undefined(adata, scores_df=cumulative_scores_df, phenotype_column = preRF_column, 
                         output_column= pre_filtering_column, 
                         RF_majority_column = RF_majority_column, verbose = verbose)

    adata.obs[final_column] = adata.obs[pre_filtering_column].astype(str)

    # Use the phenotyping table to induce a score to the child nodes that have not been evaluated
    changes = True
    while changes:
        changes = False
        for idx, row in phenotype_table.iterrows():
            parent = row.iloc[0]
            child = row.iloc[1]
            
            if parent in cumulative_scores_df.columns:
                # If the child column does not exist, we create it empty
                if child not in cumulative_scores_df.columns:
                    cumulative_scores_df[child] = np.nan
                    
                # Search rows where the child node's score doesn't exist but the parent node's does
                mask_inherit = cumulative_scores_df[child].isna() & cumulative_scores_df[parent].notna()
                
                if mask_inherit.any():
                    # The child node inherits the parent's score
                    cumulative_scores_df.loc[mask_inherit, child] = cumulative_scores_df.loc[mask_inherit, parent]
                    changes = True

    # Now that the end nodes have a score, we can delete the scores for the rest of the nodes
    intermediate_nodes = phenotype_table.iloc[:, 0].dropna().unique().tolist()
    
    columns_to_drop = [node for node in intermediate_nodes if node in cumulative_scores_df.columns]
    if columns_to_drop:
        cumulative_scores_df.drop(columns=columns_to_drop, inplace=True)
        
    if "phenotype_level0" in adata.obs.columns:
        adata.obs.drop(columns=["phenotype_level0"], inplace=True)

    final_phenotypes = adata.obs[final_column].values

    # Extract the actual scores of the winning phenotypes
    col_indices = cumulative_scores_df.columns.get_indexer(final_phenotypes)
    row_indices = np.arange(len(cumulative_scores_df))
    
    # Create an array to hold the winner's score for each cell
    winner_scores = np.zeros(len(cumulative_scores_df))
    valid_mask = col_indices >= 0
    winner_scores[valid_mask] = cumulative_scores_df.values[row_indices[valid_mask], col_indices[valid_mask]]

    # Calculate the maximum allowed score for losers
    loser_cap = np.maximum(0.0, winner_scores - 0.01)

    # Cap all losing phenotypes
    for col in cumulative_scores_df.columns:

        is_not_winner = adata.obs[final_column] != col
        
        # Apply the cap only to the losers that exceeded the winner's score
        cumulative_scores_df.loc[is_not_winner, col] = np.minimum(
            cumulative_scores_df.loc[is_not_winner, col], 
            loser_cap[is_not_winner]
        )

    # Save the whole matrix on the original adata
    adata.obsm[f"{final_column}_scores"] = cumulative_scores_df.fillna(0.0)

    # Formatting for graphs later
    phenotypes = adata.obs[final_column].dropna().unique().tolist()
    undefined = [phenotype for phenotype in phenotypes if "Undefined_" in phenotype]
    good_ones = [phenotype for phenotype in phenotypes if phenotype not in undefined]

    good_ones.sort()
    undefined.sort()
    new_order = good_ones + undefined

    adata.obs[final_column] = pd.Categorical(
        adata.obs[final_column], 
        categories=new_order, 
        ordered=True
        )
    
    if verbose:
        percentages = adata.obs[final_column].value_counts(dropna=False, normalize = True) * 100
        print(percentages)
    
    return adata

def phenotype_cells_from_regions(
    sdata,
    marker_places,
    labels_key=None,
    shapes_key=None,
    input_table_key="table_subcellular_clusters",
    region_phenotype_column="final_subcellular_phenotype",
    output_table_key="table",
    output_column="final_phenotype_through_subcellular",
    verbose=True
    ):

    """
    Assign a global phenotype to each cell using the score matrix of its zones,
    their areas and their position.
    """
    import numpy as np
    import pandas as pd
    from shapely.geometry import Point
    from skimage.measure import regionprops_table
    import anndata as ad
    from spatialdata.models import TableModel

    if shapes_key is None:
        shapes_key = list(sdata.shapes.keys())[0]
    if labels_key is None:
        labels_key = list(sdata.labels.keys())[0]

    # Extract the zones' data
    if verbose:
        print(f"Reading subcellular phenotypes from sdata.tables['{input_table_key}']...")
    
    adata_zones = sdata.tables[input_table_key]
    df_zones = adata_zones.obs.copy()
    
    # Get the real alphanumeric IDs from the table or the shapes
    if output_table_key in sdata.tables:
        real_ids_list = sdata.tables[output_table_key].obs.index.tolist()
    else:
        real_ids_list = sdata.shapes[shapes_key].index.tolist()
    
    # Convert the integer to the alphanumeric ID
    df_zones["Cell_ID"] = df_zones["Cell_ID"].apply(
        lambda x: real_ids_list[int(x)] if int(x) < len(real_ids_list) else str(x)
    )

    df_zones["zone_index"] = df_zones.index
    df_zones["Label_ID"] = df_zones["Label_ID"].astype(np.int64)

    if verbose:
        print("Extracting centroids and areas of the subcellular regions...")
    mask_np = np.array(sdata.labels[labels_key].values)
    props = regionprops_table(mask_np, properties=("label", "centroid", "area"))
    df_centroids = pd.DataFrame(props)
    df_centroids.rename(columns={"label": "Label_ID"}, inplace=True)

    df_master = pd.merge(df_zones, df_centroids, on="Label_ID")

    if verbose:
        print("Precomputing geometrical measures...")
    cell_polygons = sdata.shapes[shapes_key]
    
    dict_geometries = cell_polygons["geometry"].to_dict()
    dict_centroids = {k: geom.centroid for k, geom in dict_geometries.items()}
    dict_borders = {k: geom.boundary for k, geom in dict_geometries.items()}

    calculated_weights = []

    if verbose:
        print("Calculating position and area coefficients")
        
    for idx, row in df_master.iterrows():
        cell_id_str = str(row["Cell_ID"])
        assigned_phenotype = str(row[region_phenotype_column])
        region_area = row["area"]

        if cell_id_str not in dict_geometries:
            calculated_weights.append(0.0)
            continue

        phenotype_place = marker_places.get(assigned_phenotype, "neutral")

        cell_center = dict_centroids[cell_id_str]
        cell_border = dict_borders[cell_id_str]

        region_center = Point(row["centroid-1"], row["centroid-0"])

        d_center = region_center.distance(cell_center)
        d_border = region_center.distance(cell_border)

        denominator = d_center + d_border
        i_pos = d_center / denominator if denominator > 0 else 0.5

        if phenotype_place == "nuclear":
            place_weight = 1.0 - i_pos
        elif phenotype_place == "membrane":
            place_weight = i_pos
        else:
            place_weight = 1.0

        calculated_weights.append(place_weight * region_area)

    df_master["peso_espacial"] = calculated_weights

    subcellular_scores_key = f"{region_phenotype_column}_scores"
    
    if subcellular_scores_key in adata_zones.obsm:
        df_scores_zones = pd.DataFrame(
            adata_zones.obsm[subcellular_scores_key], 
            index=adata_zones.obs.index
        )
        #Generate a score for the Undefined class
        df_scores_zones['Undefined_'] = 1 - df_scores_zones.max(axis=1)

    else:
        raise KeyError(f"The '{subcellular_scores_key}' matrix was not found in adata_zones.obsm.")

    # Allign the dscores to df_faster's index
    df_scores_alligned = df_scores_zones.loc[df_master["zone_index"]].copy()
    df_scores_alligned.index = df_master.index 

    # Multiply the phenotype score times the region's area
    df_scores_weighted = df_scores_alligned.multiply(df_master["peso_espacial"], axis=0)
    df_scores_weighted["Cell_ID"] = df_master["Cell_ID"].values

    # Add the scores for each region of the cell
    if verbose:
        print("Computing the cell phenotype...")
    df_cellular_consensus = df_scores_weighted.groupby("Cell_ID").sum()

    # The phenotype with the highest score is selected
    final_cells = pd.DataFrame(index=df_cellular_consensus.index)
    final_cells[output_column] = df_cellular_consensus.idxmax(axis=1)
    
    # Save the final score of the winning phenotype
    final_cells["global_phenotype_confidence"] = df_cellular_consensus.max(axis=1)
    final_cells["n_zones_evaluated"] = df_master.groupby("Cell_ID").size()
    final_cells.index = final_cells.index.astype(str)

    # Including the information into sdata
    if output_table_key in sdata.tables:
        if verbose: 
            print(f"Updating sdata.tables['{output_table_key}']...")
        adata_cells = sdata.tables[output_table_key]
        
        for col in [output_column, "global_phenotype_confidence", "n_zones_evaluated"]:
            if col in adata_cells.obs.columns:
                adata_cells.obs.drop(columns=col, inplace=True)

        adata_cells.obs = adata_cells.obs.join(final_cells, how="left")
        adata_cells.obs[output_column] = (
            adata_cells.obs[output_column].astype(str).replace("nan", "Undefined_")
        )
        adata_cells.obs["n_zones_evaluated"] = adata_cells.obs["n_zones_evaluated"].fillna(0).astype(np.int32)
        adata_cells.obs[output_column] = adata_cells.obs[output_column].astype("category")
        
        # Saving the whole profile into the .obsm
        adata_cells.obsm[f"{output_column}_profiles"] = df_cellular_consensus.reindex(adata_cells.obs.index).fillna(0.0)

    else:
        if verbose: 
            print(f"Building new '{output_table_key}' table...")
        lista_cell_ids = cell_polygons.index.astype(str)
        df_obs = pd.DataFrame(index=lista_cell_ids).join(final_cells, how="left")
        
        df_obs[output_column] = df_obs[output_column].astype(str).replace("nan", "Undefined_no_zones")
        df_obs["n_zones_evaluated"] = df_obs["n_zones_evaluated"].fillna(0).astype(np.int32)
        df_obs[output_column] = df_obs[output_column].astype("category")
        df_obs["cell_id"] = df_obs.index.values
        df_obs["region"] = shapes_key
        
        X_dummy = np.zeros((len(df_obs), 0), dtype=np.float32)
        adata_cells = ad.AnnData(X=X_dummy, obs=df_obs)
        adata_cells.obsm[f"{output_column}_profiles"] = df_cellular_consensus.reindex(df_obs.index).fillna(0.0)
        
        sdata.tables[output_table_key] = TableModel.parse(
            adata_cells, region=shapes_key, region_key="region", instance_key="cell_id"
        )
        
    return sdata

def functional_state_phenotyping(adata, functional_table, 
                                 base_phenotype_column="subcellular_phenotype", final_column="functional_phenotype",
                                 user_gates = None, 
                                 verbose=False, plot_distributions=False):
    """
    Based on the data (already enriched with nuclear/functional markers) and a scimap-like
    states table, performs Hierarchical Gating.
    
    Unlike base phenotyping, it does not start at "All", but rather at the phenotypes 
    defined in the first column (Input_node). Cells that do not satisfy any functional 
    marker criteria simply retain their base phenotype.
    """
    
    import warnings
    warnings.filterwarnings('ignore')
    import numpy as np
    import pandas as pd
    import anndata as ad
    import matplotlib.pyplot as plt
    from rescale_mod import rescale
    from phenotype_cells_mod import phenotype_cells

    # Loading the data (if necessary)
    if type(adata) == str:
        adata = ad.read_h5ad(adata)
    if type(functional_table) == str:
        functional_table = pd.read_csv(functional_table)

    functional_table.dropna(axis=1, how="all", inplace=True)
    
    # Initialization: all cells start with their previous phenotype
    adata.obs[final_column] = adata.obs[base_phenotype_column].astype(str)
    
    input_node = functional_table.columns[0]
    output_node = functional_table.columns[1]
    
    parent_nodes = functional_table[input_node].dropna().unique()
    
    # Matrix to keep track of the phenotype scores
    functional_scores_df = pd.DataFrame(index=adata.obs.index)

    for node in parent_nodes:
        mask_node = adata.obs[base_phenotype_column] == node
        
        if not mask_node.any():
            if verbose: print(f"Skipping '{node}': No cells on this node.")
            continue
            
        if verbose: print(f"\n Evaluating state markers for {node} ({mask_node.sum()} cells)")
        
        # Subtable for this precise node
        phenotype_node = functional_table[functional_table[input_node] == node].dropna(axis=1, how="all")
        adata_node = adata[mask_node].copy()
        
        # Extract the markers to evaluate
        markers_in_node = [m for m in phenotype_node.columns[2:] if m in adata_node.var_names]
        
        if not markers_in_node:
            if verbose: print(f"The markers {list(phenotype_node.columns[2:])} are needed for {node} and aren't found in adata.")
            continue

        gmm_markers = []
        manual_markers = []
        
        # Normalize user_gates into a standard DataFrame with a 'gates' column
        if user_gates is not None:
            if isinstance(user_gates, pd.DataFrame):
                user_gates_df = user_gates.copy()
                if 'gates' not in user_gates_df.columns and user_gates_df.shape[1] > 0:
                    user_gates_df.columns = ['gates'] + list(user_gates_df.columns[1:])
            elif isinstance(user_gates, dict):
                user_gates_df = pd.DataFrame.from_dict(user_gates, orient='index', columns=['gates'])
            else:
                user_gates_df = pd.DataFrame(columns=['gates'])
        else:
            user_gates_df = pd.DataFrame(columns=['gates'])
            
        # Split markers into automated (GMM) and manual gating lists
        for m in markers_in_node:
            if m in user_gates_df.index:
                manual_markers.append(m)
            else:
                gmm_markers.append(m)

        # Initialize the final node gates container with the exact 'gates' column layout
        gates = pd.DataFrame(index=markers_in_node, columns=['gates'], dtype=float)
        
        # 1. Calculate automated gates via GMM
        if gmm_markers:
            gates_gmm = calculate_gates_gmm_bic(
                adata_node, 
                gmm_components="auto", 
                markers=gmm_markers,
                verbose=verbose
            )
            gates.loc[gmm_markers, 'gates'] = gates_gmm.loc[gmm_markers, 'gates']
            
        # 2. Extract manual gates from user input
        if manual_markers:
            gates.loc[manual_markers, 'gates'] = user_gates_df.loc[manual_markers, 'gates']
            if verbose:
                print(f"   Using manual gates for: {manual_markers}")
            
        # Ensure the final combined matrix contains pure numeric float values
        gates = gates.astype(float)
        
        if plot_distributions:
            plot_density_w_gates(adata_node, markers_in_node, gates, gates_are_logarithmic=True, figsize_unit=(4, 4))
            plt.show()

        # Rescaling the data to phenotype
        adata_subset = adata_node[:, markers_in_node].copy()
        if "imageid" not in adata_subset.obs:
            adata_subset.obs["imageid"] = "placeholder"
            
        rescale(adata_subset, gate=gates, verbose=False)

        # Phenotype based on the rescaled data
        adata_subset, node_scores = phenotype_cells(
            adata_subset, 
            phenotype_node, 
            label="temp_state", 
            return_score=True, 
            verbose=False
        )
        
        # Include the new phenotype on the adata object
        new_phenotype = adata_subset.obs["temp_state"].astype(str)
        idx_changes = new_phenotype[~new_phenotype.str.contains("Undefined_")].index
        if len(idx_changes) > 0:
            adata.obs.loc[idx_changes, final_column] = new_phenotype.loc[idx_changes]
        
        # Save the scores of the new phenotypes
        for col in node_scores.columns:
            functional_scores_df.loc[idx_changes, col] = node_scores.loc[idx_changes, col].values

    adata.obsm[f"{final_column}_scores"] = functional_scores_df.fillna(0.0)
    
    # Format for latter graphs
    phenotypes = adata.obs[final_column].dropna().unique().tolist()
    phenotypes.sort()
    
    undefined = [f for f in phenotypes if "Undefined" in f]
    good_ones = [f for f in phenotypes if f not in undefined]
    
    new_order = good_ones + undefined

    adata.obs[final_column] = pd.Categorical(
        adata.obs[final_column], 
        categories=new_order, 
        ordered=True
    )
    
    if verbose:
        print("Functional state phenotype results")
        percentages = adata.obs[final_column].value_counts(dropna=False, normalize=True) * 100
        print(percentages.round(2).astype(str) + " %")
        
    return adata

def phenotype_full_pipeline(
                    pheno_both,
                    sdata,
                    IMAGE_KEY = None,  
                    CELLS_KEY = None,
                    NUCLEUS_KEY = "nucleus",
                    TABLE_KEY = "table",
                    SUBCELLULAR_REGIONS_KEY = "subcellular_regions",
                    SUBCELLULAR_REGIONS_TABLE = "table_subcellular_regions",
                    LINEAGE_PHENOTYPE_COLUMN = "lineage_phenotype",
                    FINAL_PHENOTYPE_COLUMN = "final_phenotype",
                    SAVE_ROUTE = None,
                    channels = None,
                    user_gates = None,
                    places_map = None,
                    global_positive_strategy="rightmost",
                    data_is_logarithmic = False,
                    n_jobs = 8,
                    clipping_percentile = 99,
                    verbose = True,
                    plot_distributions = False
                    ):
    
    import matplotlib.pyplot as plt
    
    phenotype_table, functional_table = separate_table_by_phase(pheno_both)

    if IMAGE_KEY is None: 
        IMAGE_KEY = list(sdata.images.keys())[0]

    if NUCLEUS_KEY in list(sdata.shapes.keys()):
        nucleus_are_already_there = True
    else:
        nucleus_are_already_there = False

    if CELLS_KEY is None: 
        CELLS_KEY = list(sdata.shapes.keys())[0]

    if channels is None: 
        img_data = sdata.images[IMAGE_KEY]['scale0'][list(sdata.images[IMAGE_KEY]['scale0'].keys())[0]]
        channels = [str(c) for c in img_data.coords['c'].values]
        if verbose: 
            print(f"Channels were not given, the following have been extracted from image {IMAGE_KEY}:")
            print(channels)

    if places_map is None: 
        places_map = { f : "neutral" for f in phenotype_table.iloc[1:, 1].unique()}

    sdata = prefilter_polygons(sdata=sdata, 
                            global_positive_strategy=global_positive_strategy, 
                            clipping_percentile=99, 
                            data_is_logarithmic=data_is_logarithmic, 
                            verbose = verbose
                            )
    if verbose:
        sdata.pl.render_shapes(CELLS_KEY).pl.show(figsize = (18, 18))
        plt.show()

    sdata = calculate_subcellular_regions(sdata, 
                                       image_key= IMAGE_KEY, 
                                       shapes_key=CELLS_KEY, 
                                       output_key=SUBCELLULAR_REGIONS_KEY, 
                                       filtered_channels=channels,
                                       n_jobs = n_jobs)
    if verbose:
        sdata.pl.render_labels().pl.show(figsize = (18,18))
        plt.show()

    sdata = extract_regional_intensities(
        sdata,
        image_key=IMAGE_KEY,  
        labels_key=SUBCELLULAR_REGIONS_KEY,  
        output_table_key=SUBCELLULAR_REGIONS_TABLE,
        clipping_percentile=clipping_percentile/100
        )


    sdata.tables[SUBCELLULAR_REGIONS_TABLE] = subcellular_HSC_phenotyping(sdata.tables[SUBCELLULAR_REGIONS_TABLE], 
                                                                        phenotype_table, 
                                                                        final_column=LINEAGE_PHENOTYPE_COLUMN,
                                                                        verbose = True, 
                                                                        leiden = False, 
                                                                        user_gates = user_gates,
                                                                        plot_distributions = plot_distributions)

    if verbose:
        sdata.pl.render_labels(SUBCELLULAR_REGIONS_KEY, color = LINEAGE_PHENOTYPE_COLUMN).pl.show(figsize = (18,18))
        plt.show()

    phenotype_cells_from_regions(
        sdata,
        places_map,
        labels_key = None,
        shapes_key = None,
        input_table_key=SUBCELLULAR_REGIONS_TABLE,
        region_phenotype_column=LINEAGE_PHENOTYPE_COLUMN,
        output_table_key=TABLE_KEY,
        output_column=LINEAGE_PHENOTYPE_COLUMN
        )
    
    clean_and_repair_colors(sdata, column=LINEAGE_PHENOTYPE_COLUMN)

    if verbose: 
        sdata.pl.render_shapes(CELLS_KEY, color = LINEAGE_PHENOTYPE_COLUMN).pl.show(figsize = (18,18))
        plt.show()

    import warnings
    warnings.filterwarnings('ignore')

    if not nucleus_are_already_there:
        sdata = extract_nuclei(
            sdata=sdata,
            dapi_channel = "DAPI",
            image_key = IMAGE_KEY,
            shapes_key = CELLS_KEY,
            output_shapes_key=NUCLEUS_KEY,
            global_positive_strategy='rightmost',
            clipping_percentile=clipping_percentile,
            data_is_logarithmic = False, 
            verbose = True
            )

    sdata = aggregate(
            sdata=sdata,
            method="median",
            shrink_distance=None, 
            image_key=IMAGE_KEY,
            shapes_key=NUCLEUS_KEY, 
            table_key=TABLE_KEY      
            )

    sdata.tables[TABLE_KEY] = functional_state_phenotyping(
        adata=sdata.tables[TABLE_KEY],
        functional_table=functional_table,
        base_phenotype_column=LINEAGE_PHENOTYPE_COLUMN, 
        final_column=FINAL_PHENOTYPE_COLUMN, 
        verbose=True,
        user_gates = user_gates,
        plot_distributions=plot_distributions
        )

    clean_and_repair_colors(sdata, column=FINAL_PHENOTYPE_COLUMN)

    if verbose:
        sdata.pl.render_shapes(CELLS_KEY, color=FINAL_PHENOTYPE_COLUMN).pl.show(figsize = (10, 10))
        plt.show()

    if SAVE_ROUTE is not None:
        print(f"Saving the SpatialData object to {SAVE_ROUTE}")
        sdata.write(SAVE_ROUTE, overwrite = True)

    return sdata