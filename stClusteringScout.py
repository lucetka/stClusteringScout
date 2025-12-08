
# This is stClusteringScout_Spekulatius.py version from Mon Dec 8th 2025

# Cookie was a working and deployed version with exporting HDBSCAN results but without UMAP results exported.
# Friday Dec 5th I added UMAP saving but messed up model writing and the sidebar behavior was messed up, it appeared only AFTER data upload.
# Also help tooltips were gone. I named this Spekulatius. Monday Dec 8th I fixed Spekulatius but created more errors and finally I restarted from from Cookie
# with co-pilot using some of the fixes I manually tested while fixing Spekulatius. So I renamed Spekulatius to Spekulatius v0 and the latest, "Best" version
# which restarted from Cookie was renamed to Spekulatius. It's not confusing at all!!!!! (and I hope I recapped it right)


import io
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import umap
from hdbscan import HDBSCAN

st.set_page_config(layout="wide")
np.random.RandomState(42)

@st.cache_data
def build_export_table(results, data, number_dimensions, min_samples):
    export_df = pd.DataFrame({"ID": data.index})
    for nn in results.keys():
        for mcs in results[nn].keys():
            for eps, labels in results[nn][mcs].items():
                col_name = f"{number_dimensions}D_nn{nn}_mins{min_samples}_MSC{mcs}_eps{eps}"
                export_df[col_name] = labels
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


def cluster_with_all_models_incl_eps(nn, data_nd, nested_dict):
    clustered = {}
    for mcs in nested_dict.keys():
        clustered[mcs] = [{}, {}]
        my_range = "not applicable"
        for eps_val, model in nested_dict[mcs].items():
            labels = model.fit_predict(data_nd)
            clustered[mcs][0][eps_val] = labels
            if eps_val == 0:
                condensed_tree_df = model.condensed_tree_.to_pandas()
                tree_df_clusters = condensed_tree_df[condensed_tree_df.child_size > 1]
                selected = model.condensed_tree_._select_clusters()
                eps_df = tree_df_clusters.loc[tree_df_clusters["child"].isin(selected)]
                eps_df["eps"] = eps_df["lambda_val"].apply(lambda x: 1 / x)
                my_range = (eps_df["eps"].min(), eps_df["eps"].nlargest(2).min())
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
    try:
        if "models_umap" not in st.session_state or len(st.session_state["models_umap"]) == 0:
            st.warning("Please create models first.")
            st.stop()
        data = load_data(uploaded_file, has_index_col)
        if data is None:
            st.warning("Uploaded data has missing values; please fix and retry.")
            st.stop()

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
            nn: cluster_with_all_models_incl_eps(nn, dimred[nn], st.session_state["models_hdbscan_configs"])
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

        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "2D plots", "Cluster_selection_epsilon range recommendations", "Summary plot", "Recommendations",
            "Clusterings vs hyperparameters", "Diagnostic plots", "Models in use", "Download results"
        ])
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

            st.download_button(
                label="Download ALL results as ZIP",
                data=buffer.getvalue(),
                file_name="all_results.zip",
                mime="application/zip",
            )
    except Exception as e:
        st.exception(e)
        st.error("Calculation failed. Review parameters and try again.")
