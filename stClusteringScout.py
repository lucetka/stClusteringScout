# this is _NutellaPeanut version


import io
import zipfile
import colorcet as cc

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import umap
from hdbscan import HDBSCAN

import json

st.set_page_config(layout="wide")
np.random.RandomState(42)


@st.cache_data
def build_export_table(results, data, number_dimensions, min_samples):

    export_df = pd.DataFrame({"ID": data.index})

    # STEP 1: Build ALL clustering columns
    for nn in results.keys():
        for mcs in results[nn].keys():
            for eps, labels in results[nn][mcs].items():
                col_name = f"{number_dimensions}D_nn{nn}_mins{min_samples}_MSC{mcs}_eps{eps}"
                export_df[col_name] = labels

    # STEP 2: Add ONE reference color column
    if (
        "color_map" in st.session_state and
        "color_ref_nn" in st.session_state and
        "color_ref_mcs" in st.session_state
    ):
        try:
            nn = st.session_state["color_ref_nn"]
            mcs = st.session_state["color_ref_mcs"]

            # ✅ SAFE handling of eps key
            # eps=0 reference (with fallback safety)
            eps_dict = results[nn][mcs]
            ref_labels = eps_dict[0.0] if 0.0 in eps_dict else eps_dict[0]

            #export_df["Color_ref_eps0"] = [
            #    st.session_state["color_map"].get(str(int(lbl)), "#000000")
            #    for lbl in ref_labels
            #]
            nn = st.session_state["color_ref_nn"]
            mcs = st.session_state["color_ref_mcs"]

            col_name = f"Color_{number_dimensions}D_nn{nn}_mins{min_samples}_MSC{mcs}_eps0"

            export_df[col_name] = [
                st.session_state["color_map"].get(str(int(lbl)), "#000000")
                for lbl in ref_labels
            ]

        except Exception as e:
            print("Color column failed:", e)

    # STEP 3: Return final table
    return export_df

@st.cache_data
def load_data(uploaded_file, has_index_col):
    if has_index_col == "Yes":
        data_local = pd.read_csv(uploaded_file, index_col=0)
    else:
        data_local = pd.read_csv(uploaded_file)
    if data_local.isnull().sum().sum() > 0:
        st.error("Your dataset has missing values. Please fix them (e.g., fill with 0).")
        return None
    return data_local

def build_color_map(cluster_labels):
    unique = sorted(set(cluster_labels))
    non_noise = [c for c in unique if c != -1]

    palette_raw = list(cc.glasbey) + list(cc.glasbey_cool)

    def is_bad_color(hex_color, low=15, high=240):
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return (
            (r > high and g > high and b > high) or  # avoid white/near-white
            (r < low and g < low and b < low)        # avoid near-black
        )

    palette = [c for c in palette_raw if not is_bad_color(c)]

    color_map = {}

    for i, cluster_id in enumerate(non_noise):
        color_map[str(cluster_id)] = palette[i % len(palette)]

    # noise always white
    if -1 in unique:
        color_map["-1"] = "#FFFFFF"

    return color_map




#def build_color_map(cluster_labels):
#    import colorcet as cc

 #   unique_clusters = sorted(set(cluster_labels))
 #   palette = list(cc.glasbey) + list(cc.glasbey_cool)

 #   color_map = {}
 #   color_idx = 0

  #  for c in unique_clusters:
  #      if c == -1:
  #          color_map[c] = "#FFFFFF"   # white for noise
  #      else:
  #          color_map[c] = palette[color_idx % len(palette)]
  #          color_idx += 1

   # return color_map




def _exemplar_indices_from_tree(clusterer):
    """
    Return a list of 1D integer index arrays, one per selected cluster,
    computed from the condensed tree + prediction data.

    This mirrors the approach discussed by HDBSCAN maintainers:
    - iterate selected clusters
    - for each leaf in the cluster's DFS, find points at the leaf's max lambda
    - collect their 'child' indices (these are data row indices)
    """
    try:
        selected_clusters = clusterer.condensed_tree_._select_clusters()
        raw = clusterer.condensed_tree_._raw_tree  # structured np array with fields incl. 'parent','child','lambda_val'
        ex_list = []

        # Requires prediction_data=True on the model (your code already sets this)
        recurse = clusterer._prediction_data._recurse_leaf_dfs

        for cluster in selected_clusters:
            cluster_ex = np.array([], dtype=np.int64)
            # DFS over leaves that belong to this selected cluster
            for leaf in recurse(cluster):
                mask = (raw['parent'] == leaf)
                if not np.any(mask):
                    continue
                # For this leaf, take children at max lambda (most persistent members)
                leaf_max_lambda = raw['lambda_val'][mask].max()
                pts_mask = mask & (raw['lambda_val'] == leaf_max_lambda)
                points = raw['child'][pts_mask]
                # Append
                if points.size > 0:
                    cluster_ex = np.hstack([cluster_ex, points.astype(np.int64, copy=False)])
            ex_list.append(cluster_ex)
        return ex_list
    except Exception:
        # If anything goes wrong (e.g., prediction_data missing), return None
        return None


def cluster_with_all_models_incl_eps(nn, data_nd, nested_dict, data_index, id_col_name="ID"):
    """
    For each min_cluster_size in 'nested_dict' run all HDBSCAN models (varying eps),
    return per-mcs a 3-part structure:
      [0] dict: eps -> labels (np.array)
      [1] dict: {"Useful eps range": (min_eps, second_largest_eps)}
      [2] dict (only for eps=0): {
            "condensed_tree_df": DataFrame,
            "exemplars_df": DataFrame with columns [cluster, row_index, <id_col_name>]
          }
    """
    clustered = {}
    for mcs in nested_dict.keys():
        clustered[mcs] = [{}, {}, {}]
        my_range = "not applicable"

        for eps_val, model in nested_dict[mcs].items():
            labels = model.fit_predict(data_nd)
            clustered[mcs][0][eps_val] = labels

            if eps_val == 0:
                # 1) Condensed tree export
                condensed_tree_df = model.condensed_tree_.to_pandas()

                # 2) Useful epsilon range (from selected cluster branches)
                tree_df_clusters = condensed_tree_df[condensed_tree_df.child_size > 1]
                selected = model.condensed_tree_._select_clusters()
                eps_df = tree_df_clusters.loc[tree_df_clusters["child"].isin(selected)].copy()
                eps_df["eps"] = eps_df["lambda_val"].apply(lambda x: 1 / x)
                my_range = (eps_df["eps"].min(), eps_df["eps"].nlargest(2).min())

                # 3) Exemplars (indices), then map to IDs
                exemplar_idx_list = _exemplar_indices_from_tree(model)  # list of arrays of integer indices
                ex_records = []
                if exemplar_idx_list is not None:
                    for cluster_label, idx_array in enumerate(exemplar_idx_list):
                        # Ensure 1-D int64 array
                        idx_array = np.asarray(idx_array).ravel().astype(np.int64, copy=False)
                        for idx in idx_array:
                            # idx is guaranteed an integer scalar here
                            # Map to original ID via 'data_index'
                            ex_records.append({
                                "cluster": int(cluster_label),
                                "row_index": int(np.asarray(idx).item()),
                                id_col_name: data_index[int(np.asarray(idx).item())]
                            })
                exemplars_df = pd.DataFrame(ex_records)

                clustered[mcs][2] = {
                    "condensed_tree_df": condensed_tree_df,
                    "exemplars_df": exemplars_df
                }

        clustered[mcs][1] = {"Useful eps range": my_range}

    return clustered


# ---------------- UI ----------------
st.title("Compare clusterings with multiple UMAP and HDBSCAN models")
st.sidebar.title("CLUSTERING SCOUT")

st.markdown("First, please load your data.")
has_index_col = st.radio(
    "Use the 1st column as index? (say 'Yes' if the 1st column is NOT a variable, e.g., an ID)",
    ("Yes", "No ( ALL columns will be considered features (variables)! ). If your 2D plot looks odd, check this again."),
    key="has_index_col",
)
uploaded_file = st.file_uploader("Select your CSV file with the multidimensional dataset you wish to cluster:")

# --- Sidebar (ALWAYS visible), with helpful tooltips ---
st.sidebar.subheader("UMAP hyperparameters")

# Helpful texts restored
n_neighbors_help = (
    "Smaller values emphasize local structure (but avoid <5). Prefer values in the ballpark of expected cluster sizes."
)
umap_metric_help = "Distance metric used by UMAP to compute nearest neighbors. Cosine/correlation can be good for text-like features."
min_dist_help = "Controls how tightly points are packed in the low-dimensional embedding. Smaller values preserve local detail; larger values make clusters more spread out."
random_state_help = "Seed for reproducibility. Use the same value to get repeatable embeddings."

min_samples_help = (
    "Higher value => more points marked as noise. min_samples controls how conservative clustering is. "
    "Decrease if you have too much noise; increase if you want to avoid dubious assignments."
)
mcs_help = "Minimal cluster size that HDBSCAN will consider as a cluster. Larger values yield fewer, bigger clusters."
hdbscan_metric_help = "Distance metric used by HDBSCAN (limited options)."
cluster_sel_help = "HDBSCAN selection method: 'eom' (excess of mass) vs 'leaf'. 'eom' is default and generally recommended."

mineps_help = "Lower eps bound. Values up to this often have no effect (see Tab 2 recommendations)."
maxeps_help = "Upper eps bound. Try values well below the upper threshold where the dataset collapses to a few clusters."
eps_step_help = "Step size between epsilon values. If 0 with max>0, we take max as the only value."

constraints_help = "These constraints only affect recommendations; they do not change the clustering itself."

# Use data length if available; else default values (sidebar visible pre-upload)
if uploaded_file is not None:
    _data_preview = load_data(uploaded_file, has_index_col)
    data_len = len(_data_preview) if _data_preview is not None else 0
else:
    _data_preview = None
    data_len = 0

try:
    max_n_neighbors = round(data_len / (1 / 0.8)) if data_len > 0 else 500
except Exception:
    max_n_neighbors = 500

st.sidebar.write("Dataset length = ", data_len)
st.sidebar.write("As max n_neighbors we will allow ", max_n_neighbors)

n0 = st.sidebar.number_input("Minimal n_neighbors to try:", help=n_neighbors_help, min_value=1, max_value=max_n_neighbors, step=1, value=10, key="n_neighbors_0")
n1 = st.sidebar.number_input("Maximum n_neighbors to try:", help=n_neighbors_help, min_value=1, max_value=max_n_neighbors, step=1, value=30, key="n_neighbors_1")
nn_step = st.sidebar.number_input("Select the step for n_neighbors:", min_value=1, max_value=max_n_neighbors, step=1, value=int(n0), key="nn_step")
number_dimensions = st.sidebar.select_slider("Number of dimensions for clustering", options=[i for i in range(2, 51)], value=5, key="number_dimensions")
umap_metric = st.sidebar.selectbox("UMAP metric", (
    "euclidean","manhattan","chebyshev","minkowski","canberra","braycurtis","haversine","mahalanobis",
    "wminkowski","seuclidean","cosine","correlation","hamming","jaccard","dice","russellrao","kulsinski",
    "rogerstanimoto","sokalmichener","sokalsneath","yule"
), index=10, key="umap_metric", help=umap_metric_help)
min_dist = st.sidebar.number_input("UMAP min_dist", min_value=0.0, step=0.01, value=0.0, key="min_dist", help=min_dist_help)
random_state_val = st.sidebar.number_input("random_state", min_value=0, step=1, value=42, key="random_state", help=random_state_help)

# Sidebar: HDBSCAN hyperparameters with tooltips
st.sidebar.subheader("HDBSCAN hyperparameters")
min_samples = st.sidebar.number_input("min_samples", min_value=1, max_value=max_n_neighbors, value=3, key="min_samples", help=min_samples_help)
mcs0 = st.sidebar.number_input("Smallest min_cluster_size", min_value=3, max_value=max_n_neighbors, value=5, key="min_cluster_size_0", help=mcs_help)
mcs1 = st.sidebar.number_input("Largest  min_cluster_size", min_value=3, max_value=max_n_neighbors, value=max_n_neighbors, key="min_cluster_size_1", help=mcs_help)
mcs_step = st.sidebar.number_input("Step for min_cluster_size", min_value=1, max_value=int(mcs1), value=int(mcs0), key="mcs_step")
cluster_selection_method = st.sidebar.radio("Cluster selection method", ("eom", "leaf"), index=0, key="cluster_selection_method", help=cluster_sel_help)

st.sidebar.write("If unsure about epsilon range, run first with max eps=0.00 to get a recommendation, then re-run with a range.")
mineps = st.sidebar.number_input("Minimum epsilon", min_value=0.0, value=0.010, format="%.8f", key="mineps", help=mineps_help)
maxeps = st.sidebar.number_input("Maximum epsilon", min_value=0.0, value=0.030, format="%.8f", key="maxeps", help=maxeps_help)
eps_step = st.sidebar.number_input("Epsilon step", min_value=0.0, value=0.03, format="%.8f", key="eps_step", help=eps_step_help)

st.sidebar.subheader("Recommendation constraints")
max_acceptable_n_clusters = st.sidebar.number_input("Max acceptable #clusters", min_value=1, value=350, key="max_n_clusters", help=constraints_help)
min_acceptable_n_clusters = st.sidebar.number_input("Min acceptable #clusters", min_value=1, value=100, key="min_n_clusters", help=constraints_help)
plot_connectivity = st.sidebar.radio("Connectivity plot?", ("No", "Yes"), index=0, key="plot_connectivity")
include_umap2d_all = st.sidebar.checkbox(
    "Include all 2D UMAP embeddings (per n_neighbors) in ZIP",
    value=True,
    help=("Exports one CSV per n_neighbors with columns: ID, UMAP_2D_X, UMAP_2D_Y. This does not affect clustering; it only controls ZIP contents."),
    key="include_umap2d_all",
)

# Buttons always visible; Calculate disabled until models exist
create_models_clicked = st.sidebar.button("Create models with the selected hyperparameter ranges to compare", key="create_models")
calculate_clicked = st.sidebar.button("Calculate and plot results", key="calculate_and_plot")

# If no file, inform but still allow exploring sidebar
if uploaded_file is None:
    st.info("Upload a dataset to run models. The sidebar stays available to explore parameters and tooltips.")

# Create models
if create_models_clicked and uploaded_file is not None:
    try:
        epsstep = eps_step
        if (maxeps > 0) and (epsstep == 0):
            epsstep = maxeps
        if ((mineps == int(mineps)) and (maxeps == int(maxeps)) and (epsstep == int(epsstep))) or (maxeps == 0):
            multiplier = 1
        else:
            if epsstep <= 1:
                multiplier = int(np.power(10, (np.ceil(-np.log10(epsstep)))))
            else:
                multiplier = int(np.power(10, (np.ceil(np.log10(max(mineps, 1e-12))))))
        if maxeps == 0:
            epsstep = 0

        n_neighbors_min, n_neighbors_max = int(n0), int(n1)
        n_neighbors_step = int(nn_step)

        st.subheader("UMAP models created")
        umap_models = {
            nn: umap.UMAP(
                n_neighbors=nn,
                n_components=int(number_dimensions),
                min_dist=float(min_dist),
                metric=umap_metric,
                random_state=int(random_state_val),
            )
            for nn in range(n_neighbors_min, n_neighbors_max + n_neighbors_step, n_neighbors_step)
        }
        st.write(umap_models)

        min_cluster_size_min, min_cluster_size_max = int(mcs0), int(mcs1)
        cluster_step = int(mcs_step)
        configurations = {}
        if epsstep != 0:
            for i in range(min_cluster_size_min, min_cluster_size_max + cluster_step, cluster_step):
                configurations[i] = {
                    eps_val / multiplier: HDBSCAN(
                        min_cluster_size=i,
                        min_samples=int(min_samples),
                        cluster_selection_method=cluster_selection_method,
                        cluster_selection_epsilon=eps_val / multiplier,
                        gen_min_span_tree=True,
                        memory=r"./tmp_hdbscan_cache/",
                        prediction_data=True,
                    )
                    for eps_val in [0]
                    + list(
                        range(
                            int(multiplier * mineps),
                            int(multiplier * maxeps) + int(multiplier * epsstep),
                            int(multiplier * epsstep),
                        )
                    )
                }
        else:
            for i in range(min_cluster_size_min, min_cluster_size_max + cluster_step, cluster_step):
                configurations[i] = {
                    0.0: HDBSCAN(
                        min_cluster_size=i,
                        min_samples=int(min_samples),
                        cluster_selection_method=cluster_selection_method,
                        cluster_selection_epsilon=0.0,
                        gen_min_span_tree=True,
                        memory=r"./tmp_hdbscan_cache/",
                        prediction_data=True,
                    )
                }
        # RESTORED explicit print of HDBSCAN configs
        st.subheader("HDBSCAN model grid created")
        st.write(configurations)
        st.write(f"#UMAP={len(umap_models)}, #HDBSCAN grids={len(configurations)}")

        # Persist under non-widget keys
        st.session_state["models_umap"] = umap_models
        st.session_state["models_hdbscan_configs"] = configurations
        st.session_state["models_n_neighbors_min"] = n_neighbors_min
        st.session_state["models_n_neighbors_max"] = n_neighbors_max
        st.session_state["models_n_neighbors_step"] = n_neighbors_step
        st.session_state["models_min_cluster_size_min"] = min_cluster_size_min
        st.session_state["models_min_cluster_size_max"] = min_cluster_size_max
        st.session_state["models_cluster_step"] = cluster_step
        st.session_state["models_eps_max"] = float(maxeps)
        st.session_state["models_eps_step"] = float(epsstep)
        st.session_state["models_umap_metric"] = umap_metric
        st.session_state["models_random_state"] = int(random_state_val)
        st.session_state["models_include_umap2d_all"] = bool(include_umap2d_all)
        st.success("Models created. You can now click ‘Calculate and plot results’.")
    except Exception as e:
        st.exception(e)
        st.error("Failed to create models. Please review parameters above.")

# Calculate
if calculate_clicked and uploaded_file is not None:
#if (
#    (calculate_clicked or st.session_state.get("calculation_done", False))
#    and uploaded_file is not None):
    try:
        if "models_umap" not in st.session_state or len(st.session_state["models_umap"]) == 0:
            st.warning("Please create models first.")
            st.stop()
        data = load_data(uploaded_file, has_index_col)
        if data is None:
            st.warning("Uploaded data has missing values; please fix and retry.")
            st.stop()


        # Resolve the display name for the ID column (used in exemplars export)
        id_col_name = data.index.name or "ID"


        dimred = {
            nn: umap.UMAP(
                n_neighbors=nn,
                n_components=int(number_dimensions),
                min_dist=float(min_dist),
                metric=st.session_state["models_umap_metric"],
                random_state=st.session_state["models_random_state"],
            ).fit_transform(data)
            for nn in range(
                st.session_state["models_n_neighbors_min"],
                st.session_state["models_n_neighbors_max"] + st.session_state["models_n_neighbors_step"],
                st.session_state["models_n_neighbors_step"],
            )
        }

        if int(number_dimensions) > 2:
            umap2d_all = {
                nn: umap.UMAP(
                    n_neighbors=nn,
                    n_components=2,
                    min_dist=float(min_dist),
                    metric=st.session_state["models_umap_metric"],
                    random_state=st.session_state["models_random_state"],
                ).fit_transform(data)
                for nn in range(
                    st.session_state["models_n_neighbors_min"],
                    st.session_state["models_n_neighbors_max"] + st.session_state["models_n_neighbors_step"],
                    st.session_state["models_n_neighbors_step"],
                )
            }
        else:
            umap2d_all = dimred
        st.session_state["umap2d_all_store"] = umap2d_all

        nn_min = st.session_state["models_n_neighbors_min"]
        nn_max = st.session_state["models_n_neighbors_max"]
        vizred = {nn_min: umap2d_all[nn_min], nn_max: umap2d_all[nn_max]}

      
        results_incl_eps = {
            nn: cluster_with_all_models_incl_eps(
                nn,
                dimred[nn],
                st.session_state["models_hdbscan_configs"],
                data.index,          # <— pass original index so we can map exemplars
                id_col_name          # <— pass the name for the ID column
            )
            for nn in range(
                st.session_state["models_n_neighbors_min"],
                st.session_state["models_n_neighbors_max"] + st.session_state["models_n_neighbors_step"],
                st.session_state["models_n_neighbors_step"],
            )
        }

        results = {nn: {mcs: dlist[0] for mcs, dlist in results_incl_eps[nn].items()} for nn in results_incl_eps}
        useful_eps_ranges = {nn: {mcs: dlist[1] for mcs, dlist in results_incl_eps[nn].items()} for nn in results_incl_eps}

        dataframes = {}
        for nn in results:
            dfm = pd.DataFrame(results[nn]).T
            dataframes[nn] = pd.DataFrame(dfm.stack())
        df = pd.concat(dataframes.values(), axis=1)
        df.columns = list(results.keys())
        df = pd.DataFrame(df.unstack(level=[-2]).T.stack())
        df["n_clusters"] = df[0].apply(lambda x: (len(np.unique(x))) - 1)
        df["percent_unclustered"] = df[0].apply(lambda x: 100 * np.count_nonzero(x == -1) / len(x))
        df = df.reset_index().rename(columns={"level_0": "(UMAP) n_neighbors", "level_1": "min_cluster_size", "level_2": "eps"})

        low_color = float(df["percent_unclustered"].min())
        if round(low_color) > low_color:
            low_color = round(low_color) - 1
        high_color = float(df["percent_unclustered"].max())
        high_color = round(high_color)
        hyperparameters_plot = px.scatter(
            df,
            x="min_cluster_size",
            y="eps",
            size="n_clusters",
            size_max=100,
            color="percent_unclustered",
            color_continuous_scale="Rainbow",
            animation_frame="(UMAP) n_neighbors",
            height=900,
            range_color=[low_color, high_color],
            title="Clusters and % unclustered across UMAP/HDBSCAN grids",
        )
        hyperparameters_plot["layout"].pop("updatemenus", None)

        compliant_results_eps0 = df.loc[((df["n_clusters"] >= max(1, min_acceptable_n_clusters)) & (df["n_clusters"] <= max_acceptable_n_clusters)) & (df["eps"] == 0.00)].sort_values(by=["percent_unclustered"]) 
        compliant_results_all = df.loc[(df["n_clusters"] >= max(1, min_acceptable_n_clusters)) & (df["n_clusters"] <= max_acceptable_n_clusters)].sort_values(by=["percent_unclustered"]) 



    
        # ✅ PERSIST RESULTS (THIS IS WHAT YOU WERE MISSING)
        #st.session_state["results_incl_eps"] = results_incl_eps
        #st.session_state["results"] = results
        #st.session_state["df"] = df
        #st.session_state["useful_eps_ranges"] = useful_eps_ranges
        #st.session_state["calculation_done"] = True
        #st.session_state["hyperparameters_plot"] = hyperparameters_plot
        st.session_state["results_incl_eps"] = results_incl_eps
        st.session_state["results"] = results
        st.session_state["df"] = df
        st.session_state["useful_eps_ranges"] = useful_eps_ranges
        st.session_state["hyperparameters_plot"] = hyperparameters_plot
        st.session_state["compliant_results_eps0"] = compliant_results_eps0
        st.session_state["compliant_results_all"] = compliant_results_all
        st.session_state["calculation_done"] = True


    except Exception as e:
        st.exception(e)
        st.error("Calculation failed. Review parameters and try again.")
    
# ✅ DISPLAY BLOCK (persistent, no recompute)

if st.session_state.get("calculation_done", False):

    results_incl_eps = st.session_state["results_incl_eps"]
    results = st.session_state["results"]
    df = st.session_state["df"]
    useful_eps_ranges = st.session_state["useful_eps_ranges"]
    hyperparameters_plot = st.session_state["hyperparameters_plot"]
    
    compliant_results_eps0 = st.session_state["compliant_results_eps0"]
    compliant_results_all = st.session_state["compliant_results_all"]

    data = load_data(uploaded_file, has_index_col)
    nn_min = st.session_state["models_n_neighbors_min"]
    nn_max = st.session_state["models_n_neighbors_max"]
    vizred = st.session_state["umap2d_all_store"]

    # 👉 TABS 
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "2D plots", "Cluster_selection_epsilon range recommendations", "Summary plot", "Recommendations",
            "Clusterings vs hyperparameters", "Diagnostic plots", "Models in use", "Download results"])
    with tab1:
            st.header("2D plots using min and max n_neighbors")
            fig_1 = px.scatter(data, x=vizred[nn_min][:,0], y=vizred[nn_min][:,1], hover_data={data.index.name or "ID": data.index}, title=f"n_neighbors {nn_min}")
            fig_2 = px.scatter(data, x=vizred[nn_max][:,0], y=vizred[nn_max][:,1], hover_data={data.index.name or "ID": data.index}, title=f"n_neighbors {nn_max}")
            st.plotly_chart(fig_1, use_container_width=True)
            st.plotly_chart(fig_2, use_container_width=True)
    with tab2:
        st.header("Recommended ranges of cluster_selection_epsilon")
        eps_ranges_df = pd.DataFrame(useful_eps_ranges)
        st.write(eps_ranges_df.add_prefix("n_neighbors(UMAP)_").rename_axis("Min cluster size"))
    with tab3:
        st.header("Summary plot")
        st.plotly_chart(hyperparameters_plot, use_container_width=True)

        st.subheader("🎨 Color mapping (reference clustering)")

        # Get available runs from results structure
        available_nns = sorted(results.keys())
        selected_nn = st.selectbox("Select n_neighbors for reference clustering", available_nns, key="color_ref_nn")

        available_mcs = sorted(results[selected_nn].keys())
        selected_mcs = st.selectbox("Select min_cluster_size for reference clustering", available_mcs, key="color_ref_mcs")

        if 0.0 in results[selected_nn][selected_mcs]:
            ref_labels = results[selected_nn][selected_mcs][0.0]

            color_map = build_color_map(ref_labels)

            st.success(f"Generated color map with {len(color_map)} clusters (eps=0 reference)")

            # store for later use
            st.session_state["color_map"] = color_map
            #st.session_state["color_ref_nn"] = selected_nn
            #st.session_state["color_ref_mcs"] = selected_mcs
        else:
            st.warning("eps=0 not available for selected run — cannot generate color map.")
        
        # ---------------------------
# REFERENCE CLUSTER SCATTER
# ---------------------------

        st.subheader("Reference clustering (2D view)")

        umap_2d = st.session_state["umap2d_all_store"][selected_nn]


        if ref_labels is not None and umap_2d is not None:

            df_plot = pd.DataFrame({
                "UMAP1": umap_2d[:, 0],
                "UMAP2": umap_2d[:, 1],
                "cluster": ref_labels
            })

            color_map = build_color_map(df_plot["cluster"])

            fig = px.scatter(
                df_plot,
                x="UMAP1",
                y="UMAP2",
                color=df_plot["cluster"].astype(str),
                color_discrete_map={str(k): v for k, v in color_map.items()},
                title="Reference clustering (colored by cluster)",
                height = 600,
            )

            fig.update_traces(marker=dict(size=5))
            st.plotly_chart(fig, use_container_width=True)



    with tab4:
        st.header("Recommendations")
        st.dataframe(compliant_results_eps0[["(UMAP) n_neighbors", "min_cluster_size", "eps", "n_clusters", "percent_unclustered"]])
        st.dataframe(compliant_results_all[["(UMAP) n_neighbors", "min_cluster_size", "eps", "n_clusters", "percent_unclustered"]])
    with tab5:
        st.header("Clusterings vs hyperparameters")
        unclustered_vs_mcs_line = px.line(df, x="min_cluster_size", y="percent_unclustered", animation_frame="(UMAP) n_neighbors", height=500, color="eps", range_y=[0, 1.05*(df["percent_unclustered"].max())])
        st.plotly_chart(unclustered_vs_mcs_line)
        unclustered_vs_mcs = px.scatter(df, x="min_cluster_size", y="percent_unclustered", size="n_clusters", size_max=50, animation_frame="(UMAP) n_neighbors", height=900, color="eps", color_continuous_scale="Turbo", range_y=[-5, 1.05*(df["percent_unclustered"].max())])
        st.plotly_chart(unclustered_vs_mcs, use_container_width=True)
    with tab6:
        st.subheader("Connectivity plots")
        if plot_connectivity == "Yes":
            st.write("Connectivity plot is temporarily disabled in this version.")
        else:
            st.write("You did not choose a connectivity plot.")
    with tab7:
        st.header("UMAP and HDBSCAN models in use & results")
        st.write(st.session_state["models_umap"])  # UMAP models
        st.write(st.session_state["models_hdbscan_configs"])  # HDBSCAN configs
        st.write(results_incl_eps)  # full results incl ranges
    with tab8:
        st.header("Download all results as ZIP")
        try:
            export_df = build_export_table(results, data, int(number_dimensions), int(min_samples))
        except NameError:
            export_df = pd.DataFrame({"ID": data.index})
            for nn in results:
                for mcs in results[nn]:
                    for eps_val, labels in results[nn][mcs].items():
                        col_name = f"{number_dimensions}D_nn{nn}_mins{min_samples}_MSC{mcs}_eps{eps_val}"
                        export_df[col_name] = labels
        st.write("Preview of clustering results:")
        st.dataframe(export_df.head())

        def to_html_safe(fig):
            try:
                return fig.to_html()
            except Exception:
                return None
        def to_csv_safe(df_in, index=False):
            try:
                return df_in.to_csv(index=index)
            except Exception:
                return None
        def embedding_to_csv_bytes(embedding, index):
            try:
                df_embed = pd.DataFrame(embedding, columns=["UMAP_2D_X", "UMAP_2D_Y"])
                idx_name = index.name if index.name is not None else "ID"
                df_embed.insert(0, idx_name, index)
                return df_embed.to_csv(index=False)
            except Exception:
                return None

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as z:
            csv = to_csv_safe(export_df, index=False)
            if csv:
                z.writestr("clustering_results.csv", csv)
            # --- NEW: export color map ---
            if "color_map" in st.session_state:
                try:
                    color_json = json.dumps(st.session_state["color_map"], indent=2)
                    z.writestr("color_map.json", color_json)
                except Exception as e:
                    print("Failed to export color_map.json:", e)
            try:
                eps_raw = pd.DataFrame(useful_eps_ranges)
                raw_csv = to_csv_safe(eps_raw, index=True)
                if raw_csv:
                    z.writestr("eps_recommendations_raw.csv", raw_csv)
                eps_fmt = eps_raw.add_prefix("n_neighbors(UMAP)_")
                eps_fmt.index.names = ["Min cluster size"]
                fmt_csv = to_csv_safe(eps_fmt, index=True)
                if fmt_csv:
                    z.writestr("eps_recommendations_formatted.csv", fmt_csv)
            except Exception:
                pass
            csv_eps0 = to_csv_safe(compliant_results_eps0, index=False)
            if csv_eps0:
                z.writestr("recommendations_eps0.csv", csv_eps0)
            csv_all = to_csv_safe(compliant_results_all, index=False)
            if csv_all:
                z.writestr("recommendations_all.csv", csv_all)
            html_summary = to_html_safe(hyperparameters_plot)
            if html_summary:
                z.writestr("summary_plot.html", html_summary)
            fig_1 = px.scatter(data, x=vizred[nn_min][:,0], y=vizred[nn_min][:,1], hover_data={data.index.name or "ID": data.index}, title=f"n_neighbors {nn_min}")
            fig_2 = px.scatter(data, x=vizred[nn_max][:,0], y=vizred[nn_max][:,1], hover_data={data.index.name or "ID": data.index}, title=f"n_neighbors {nn_max}")
            html_min = to_html_safe(fig_1)
            if html_min:
                z.writestr("2D_plot_min_neighbors.html", html_min)
            html_max = to_html_safe(fig_2)
            if html_max:
                z.writestr("2D_plot_max_neighbors.html", html_max)
            html_line = to_html_safe(unclustered_vs_mcs_line)
            if html_line:
                z.writestr("unclustered_vs_mcs_line.html", html_line)
            html_scatter = to_html_safe(unclustered_vs_mcs)
            if html_scatter:
                z.writestr("unclustered_vs_mcs_scatter.html", html_scatter)
            try:
                if st.session_state.get("models_include_umap2d_all", True):
                    for nn_val, emb in st.session_state["umap2d_all_store"].items():
                        csv_bytes = embedding_to_csv_bytes(emb, data.index)
                        if csv_bytes:
                            z.writestr(f"umap_2d_embedding_nn{nn_val}.csv", csv_bytes)
            
            except Exception:
                pass
            # --- NEW: export condensed trees & exemplars for eps=0 only ---
            try:
                for nn_val, per_mcs in results_incl_eps.items():
                    for mcs_val, triple in per_mcs.items():
                        # triple is [labels_by_eps, {"Useful eps range": ...}, extras_dict]
                        if isinstance(triple, list) and len(triple) > 2 and isinstance(triple[2], dict):
                            extras = triple[2]
                            tree_df = extras.get("condensed_tree_df", None)
                            ex_df = extras.get("exemplars_df", None)

                            if tree_df is not None and not tree_df.empty:
                                csv_tree = to_csv_safe(tree_df, index=False)
                                if csv_tree:
                                    z.writestr(f"eps0_condensed_tree_nn{nn_val}_mcs{mcs_val}.csv", csv_tree)

                            if ex_df is not None and not ex_df.empty:
                                csv_ex = to_csv_safe(ex_df, index=False)
                                if csv_ex:
                                    z.writestr(f"eps0_exemplars_nn{nn_val}_mcs{mcs_val}.csv", csv_ex)
            except Exception:
                # Don't fail the whole ZIP if these optional artifacts can't be generated
                pass
            

        st.download_button(
            label="Download ALL results as ZIP",
            data=buffer.getvalue(),
            file_name="all_results.zip",
            mime="application/zip",
        )
