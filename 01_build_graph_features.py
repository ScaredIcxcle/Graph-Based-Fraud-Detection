"""
01_build_graph_features.py (non-temporal version)

Builds ONE static graph per split (train, test) from ALL of that split's
transactions at once -- no daily/incremental processing. Train and test
graphs are built completely independently (no carry-over), so each split
only ever uses its own data -- no cross-split leakage.

Output: outputs/train_features.csv, outputs/test_features.csv
"""

import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain
from collections import defaultdict

import config


def load_data(path):
    print(f"Loading {path} ...")
    df = pd.read_csv(path)
    if df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=[df.columns[0]])
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df = df.sort_values("trans_date_trans_time").reset_index(drop=True)
    print(f"  {len(df):,} rows")
    return df


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def compute_baseline_features(df):
    print("Computing baseline (non-graph) features...")
    df = df.copy()
    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
    df["dob"] = pd.to_datetime(df["dob"])
    df["age"] = (df["trans_date_trans_time"] - df["dob"]).dt.days / 365.25
    df["distance_km"] = haversine_km(
        df["lat"].values, df["long"].values,
        df["merch_lat"].values, df["merch_long"].values,
    )
    df = df.sort_values(["cc_num", "trans_date_trans_time"])
    grp = df.groupby("cc_num")["amt"]
    df["cust_txn_count_so_far"] = grp.cumcount()
    df["cust_avg_amt_so_far"] = grp.apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=0, drop=True)
    df["cust_avg_amt_so_far"] = df["cust_avg_amt_so_far"].fillna(df["amt"].median())
    df["prev_txn_time"] = df.groupby("cc_num")["trans_date_trans_time"].shift(1)
    df["minutes_since_prev_txn"] = (
        (df["trans_date_trans_time"] - df["prev_txn_time"]).dt.total_seconds() / 60.0
    )
    df["minutes_since_prev_txn"] = df["minutes_since_prev_txn"].fillna(-1)
    df = df.sort_values("trans_date_trans_time").reset_index(drop=True)
    return df


def tag_chains(df):
    print("Tagging chain patterns...")
    df = df.sort_values(["cc_num", "trans_date_trans_time"]).copy()
    df["tag_chain"] = 0
    for cc_num, g in df.groupby("cc_num"):
        times = g["trans_date_trans_time"].values
        idxs = g.index.values
        n = len(g)
        i = 0
        while i < n:
            j = i
            while (j + 1 < n and
                   (pd.Timestamp(times[j + 1]) - pd.Timestamp(times[j])).total_seconds() / 60.0
                   <= config.CHAIN_TIME_DELTA_MINUTES):
                j += 1
            run_len = j - i + 1
            if run_len >= config.CHAIN_MIN_LENGTH:
                df.loc[idxs[i:j + 1], "tag_chain"] = 1
            i = j + 1
    df = df.sort_values("trans_date_trans_time").reset_index(drop=True)
    return df


def compute_community_density(G, partition):
    """Bipartite-correct: max possible internal edges = n_cust * n_merch."""
    members = defaultdict(lambda: {"cust": 0, "merch": 0})
    for node, comm in partition.items():
        if node.startswith("C_"):
            members[comm]["cust"] += 1
        elif node.startswith("M_"):
            members[comm]["merch"] += 1
    internal_edges = defaultdict(int)
    for u, v in G.edges():
        cu, cv = partition.get(u), partition.get(v)
        if cu is not None and cu == cv:
            internal_edges[cu] += 1
    density = {}
    for comm, counts in members.items():
        possible = counts["cust"] * counts["merch"]
        density[comm] = internal_edges[comm] / possible if possible > 0 else 0.0
    return density


def build_static_graph_features(df):
    print("Building static graph from all transactions in this split...")
    G = nx.Graph()
    cust_node_all = "C_" + df["cc_num"].astype(str)
    merch_node_all = "M_" + df["merchant"].astype(str)

    for cn, mn in zip(cust_node_all, merch_node_all):
        if G.has_edge(cn, mn):
            G[cn][mn]["weight"] += 1
        else:
            G.add_edge(cn, mn, weight=1)

    total_edges = G.number_of_edges()
    total_weight = sum(w for _, _, w in G.edges(data="weight"))
    print(f"  graph: {G.number_of_nodes():,} nodes, {total_edges:,} edges, {total_weight:,} total weight")

    print("  computing PageRank...")
    pagerank = nx.pagerank(G, weight="weight")

    print("  computing Louvain communities (one-time, may take a while)...")
    partition = community_louvain.best_partition(G, weight="weight", random_state=config.RANDOM_SEED, resolution=10.0)
    community_density = compute_community_density(G, partition)

    print("  computing community fraud rates...")
    df = df.copy()
    df["_cust_community_tmp"] = cust_node_all.map(partition).fillna(-1).astype(int)
    comm_fraud = df.groupby("_cust_community_tmp")["is_fraud"].agg(["sum", "count"])
    global_rate = df["is_fraud"].mean()
    prior_weight = config.FRAUD_RATE_PRIOR_WEIGHT
    comm_fraud["smoothed_rate"] = (
        (comm_fraud["sum"] + prior_weight * global_rate) / (comm_fraud["count"] + prior_weight)
    )
    fraud_rate_map = comm_fraud["smoothed_rate"].to_dict()
    df = df.drop(columns=["_cust_community_tmp"])

    merchant_indeg = df.groupby("merchant")["cc_num"].nunique()
    customer_outdeg = df.groupby("cc_num")["merchant"].nunique()
    fan_in_thresh = np.percentile(merchant_indeg.values, config.FAN_IN_PERCENTILE)
    fan_out_thresh = np.percentile(customer_outdeg.values, config.FAN_OUT_PERCENTILE)
    density_values = list(community_density.values())
    dense_thresh = np.percentile(density_values, config.DENSE_COMMUNITY_PERCENTILE) if density_values else np.inf

    n_customers = df["cc_num"].nunique()
    n_merchants = df["merchant"].nunique()

    print("  assigning features to each transaction (vectorized)...")

    cust_deg_raw = pd.Series({n: G.degree(n) for n in G.nodes() if n.startswith("C_")})
    cust_degw_raw = pd.Series({n: G.degree(n, weight="weight") for n in G.nodes() if n.startswith("C_")})
    merch_deg_raw = pd.Series({n: G.degree(n) for n in G.nodes() if n.startswith("M_")})
    merch_degw_raw = pd.Series({n: G.degree(n, weight="weight") for n in G.nodes() if n.startswith("M_")})

    out = pd.DataFrame(index=df.index)
    out["cust_pagerank"] = cust_node_all.map(pagerank).fillna(0.0).values
    out["merch_pagerank"] = merch_node_all.map(pagerank).fillna(0.0).values

    # ---- NORMALIZED degree: fraction of total edges/weight, scale-invariant across splits ----
    out["cust_degree_frac"] = cust_node_all.map(cust_deg_raw).fillna(0).rank(pct=True).values
    out["merch_degree_frac"] = merch_node_all.map(merch_deg_raw).fillna(0).rank(pct=True).values
    out["cust_degree_weighted_frac"] = cust_node_all.map(cust_degw_raw).fillna(0).rank(pct=True).values
    out["merch_degree_weighted_frac"] = merch_node_all.map(merch_degw_raw).fillna(0).rank(pct=True).values

    cust_comm = cust_node_all.map(partition).fillna(-1).astype(int)
    merch_comm = merch_node_all.map(partition).fillna(-1).astype(int)
    out["cust_community"] = cust_comm.values
    out["merch_community"] = merch_comm.values
    out["cust_community_density"] = cust_comm.map(community_density).fillna(0.0).values
    out["community_fraud_rate"] = cust_comm.map(fraud_rate_map).fillna(global_rate).values

    m_indeg = df["merchant"].map(merchant_indeg).fillna(0)
    c_outdeg = df["cc_num"].map(customer_outdeg).fillna(0)
    # NORMALIZED: fraction of total distinct customers/merchants in this split
    out["merchant_indeg_frac"] = m_indeg.rank(pct=True).values
    out["customer_outdeg_frac"] = c_outdeg.rank(pct=True).values

    out["tag_fan_in"] = (m_indeg > fan_in_thresh).astype(int).values
    out["tag_fan_out"] = (c_outdeg > fan_out_thresh).astype(int).values
    out["tag_dense_community"] = (out["cust_community_density"] > dense_thresh).astype(int).values

    return out


def main():
    for split, path in [("train", config.TRAIN_CSV), ("test", config.TEST_CSV)]:
        print(f"\n=== Processing {split.upper()} ===")
        df = load_data(path)
        df = compute_baseline_features(df)
        df = tag_chains(df)
        feat_df = build_static_graph_features(df)
        out = df.join(feat_df, how="left")
        out_path = f"{config.OUT_DIR}/{split}_features.csv"
        out.to_csv(out_path, index=False)
        print(f"Saved {split}: {len(out):,} rows -> {out_path}")
        print(f"Fraud rate: {out['is_fraud'].mean():.4%}")
        for col in ["tag_fan_in", "tag_fan_out", "tag_chain", "tag_dense_community"]:
            print(f"  {col}: {out[col].mean():.4%}")


if __name__ == "__main__":
    main()