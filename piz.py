import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mne_connectivity.viz import plot_connectivity_circle  # 关键修改
from collections import defaultdict

from scipy.stats import spearmanr

import evaluate_SEED
import evaluate_SEEDIV


# matplotlib.use("TKAgg")

def find_top_correlated_pairs(adj_matrix, ch_names, top_n=10):
    mask = np.triu_indices_from(adj_matrix, k=1)
    row_indices, col_indices = mask
    values = adj_matrix[mask]

    # 构造完整条目
    items = []
    for idx, (i, j) in enumerate(zip(row_indices, col_indices)):
        items.append((
            ch_names[i],
            ch_names[j],
            values[idx],
            abs(values[idx])
        ))

    # 稳定排序：先按绝对值，再按通道名
    items_sorted = sorted(
        items,
        key=lambda x: (-x[3], x[0], x[1])
    )

    top_pairs = [
        ((ch1, ch2), val)
        for ch1, ch2, val, _ in items_sorted[:top_n]
    ]

    return top_pairs


def plot_topk_region_connections_no_norm(top_pairs, y_margin=0.02):
    """
    Plot top-k region-level connections without normalization.
    Values are sorted after region mapping, and y-axis lower bound
    is set slightly below the minimum value for better visualization.

    Parameters
    ----------
    top_pairs : list
        [ ( (ch1, ch2), value ), ... ]
    y_margin : float
        Margin subtracted from minimum value to set y-axis lower bound
    """

    # 定义脑区映射
    regions = {
        'F': ['Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'Fz'],
        'C': ['FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FCz', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'Cz'],
        'P': ['CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'CPz', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'Pz'],
        'TL': ['FT7', 'T7', 'TP7'],
        'TR': ['FT8', 'T8', 'TP8'],
        'PO': ['PO3', 'PO4', 'PO5', 'PO6', 'PO7', 'PO8', 'POz'],
        'O': ['O1', 'O2', 'Oz'],
        'CB': ['CB1', 'CB2']
    }

    def find_region(channel):
        for region, chs in regions.items():
            if channel in chs:
                return region
        return 'unknown'

    region_pairs = []
    strengths = []

    # channel-level → region-level mapping
    for (ch1, ch2), val in top_pairs:
        r1 = find_region(ch1)
        r2 = find_region(ch2)
        key = " - ".join(sorted([r1, r2]))

        region_pairs.append(key)
        strengths.append(val)

    # sort AFTER region mapping
    sorted_items = sorted(
        zip(region_pairs, strengths),
        key=lambda x: x[1],
        reverse=True
    )

    region_pairs_sorted, strengths_sorted = zip(*sorted_items)

    # y-axis lower bound
    y_min = min(strengths_sorted) - y_margin

    # # plot
    # plt.figure()
    # plt.bar(region_pairs_sorted, strengths_sorted)
    # plt.xticks(rotation=45, ha='right')
    # plt.ylabel('Adjacency Strength')
    # plt.xlabel('Brain Region Connections')
    # plt.ylim(bottom=y_min)
    # plt.tight_layout()
    # plt.show()

    return region_pairs_sorted, strengths_sorted, y_min


def aggregate_region_pairs(region_pairs, strengths, agg='mean'):
    """
    Deterministically aggregate duplicated region-pairs.

    Parameters
    ----------
    region_pairs : list[str]
        e.g. ['G3 - G5', 'G3 - G5', 'G1 - G4', ...]
    strengths : list[float]
        Corresponding strengths
    agg : str
        'mean' or 'max'

    Returns
    -------
    uniq_pairs : list[str]
        Aggregated region-pair names (deterministic order)
    uniq_strengths : list[float]
        Corresponding aggregated strengths
    """

    # --------------------------------------------------
    # Step 1: bucket strengths by region-pair
    # --------------------------------------------------
    bucket = defaultdict(list)
    for p, s in zip(region_pairs, strengths):
        bucket[p].append(float(s))

    # --------------------------------------------------
    # Step 2: aggregate (order-independent)
    # --------------------------------------------------
    aggregated = []

    for p in bucket:
        vals = bucket[p]
        if agg == 'mean':
            val = float(np.mean(vals))
        elif agg == 'max':
            val = float(np.max(vals))
        else:
            raise ValueError("agg must be 'mean' or 'max'")
        aggregated.append((p, val))

    # --------------------------------------------------
    # Step 3: deterministic sorting
    #   1) strength descending
    #   2) region-pair name ascending (tie-breaker)
    # --------------------------------------------------
    aggregated_sorted = sorted(
        aggregated,
        key=lambda x: (-x[1], x[0])
    )

    uniq_pairs = [p for p, _ in aggregated_sorted]
    uniq_strengths = [v for _, v in aggregated_sorted]

    return uniq_pairs, uniq_strengths


def plot_topk_heset_connections_no_norm(top_pairs, y_margin=0.02):
    """
    Plot top-k HESet-level connections without normalization.

    Deterministic version:
    - Channel -> HESet mapping is ordered
    - One-to-many expansion is deterministic
    - HESet-level connections are explicitly aggregated
    - Sorting uses a strict tie-breaker

    Parameters
    ----------
    top_pairs : list
        [ ( (ch1, ch2), value ), ... ]
    y_margin : float
        Margin subtracted from minimum value to set y-axis lower bound

    Returns
    -------
    heset_pairs_sorted : tuple of str
        Sorted HESet group pairs (e.g., 'G1 - G5')
    strengths_sorted : tuple of float
        Corresponding aggregated connection strengths
    y_min : float
        Lower bound for y-axis
    """

    # --------------------------------------------------
    # Step 0: HESet definitions (model-specific)
    # --------------------------------------------------
    heset_names = [
        ["FP1", "FPZ", "FP2", "AF3", "AF4"],
        ["F7", "F5", "F3", "FT7", "FC5", "FC3", "T7", "C5", "C3"],
        ["T7", "C5", "C3", "TP7", "CP5", "CP3", "P7", "P5", "P3"],
        ["F4", "F6", "F8", "FC4", "FC6", "FT8", "C4", "C6", "T8"],
        ["C4", "C6", "T8", "CP4", "CP6", "TP8", "P4", "P6", "P8"],
        ["F3", "F1", "FZ", "F2", "F4", "FC3", "FC1", "FCZ", "FC2", "FC4",
         "C3", "C1", "CZ", "C2", "C4", "CP3", "CP1", "CPZ", "CP2", "CP4",
         "P3", "P1", "PZ", "P2", "P4"],
        ["PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8",
         "CB1", "O1", "OZ", "O2", "CB2"]
    ]

    # --------------------------------------------------
    # Step 1: build deterministic channel -> HESet mapping
    # --------------------------------------------------
    # IMPORTANT:
    #   - use list instead of set
    #   - preserve group index order
    channel_to_groups = defaultdict(list)

    for gid, group in enumerate(heset_names):
        for ch in group:
            channel_to_groups[ch.upper()].append(f"G{gid}")

    # --------------------------------------------------
    # Step 2: channel-level -> HESet-level mapping
    #         with deterministic one-to-many expansion
    # --------------------------------------------------
    pair_to_values = defaultdict(list)

    for (ch1, ch2), val in top_pairs:
        g1 = channel_to_groups.get(ch1.upper(), [])
        g2 = channel_to_groups.get(ch2.upper(), [])

        # skip channels not covered by HESet
        if not g1 or not g2:
            continue

        # sort group lists to guarantee order
        g1 = sorted(g1)
        g2 = sorted(g2)

        for a in g1:
            for b in g2:
                key = " - ".join(sorted([a, b]))
                pair_to_values[key].append(val)

    if len(pair_to_values) == 0:
        raise ValueError("No valid HESet pairs found from top_pairs.")

    # --------------------------------------------------
    # Step 3: explicit HESet-level aggregation
    #         (mean aggregation is recommended)
    # --------------------------------------------------
    heset_pairs = []
    strengths = []

    for pair in sorted(pair_to_values.keys()):
        vals = pair_to_values[pair]
        heset_pairs.append(pair)
        strengths.append(float(np.mean(vals)))

    # --------------------------------------------------
    # Step 4: deterministic sorting with tie-breaker
    # --------------------------------------------------
    sorted_items = sorted(
        zip(heset_pairs, strengths),
        key=lambda x: (-x[1], x[0])  # strength ↓ , name ↑
    )

    heset_pairs_sorted, strengths_sorted = zip(*sorted_items)

    # --------------------------------------------------
    # Step 5: y-axis lower bound
    # --------------------------------------------------
    y_min = min(strengths_sorted) - y_margin

    return heset_pairs_sorted, strengths_sorted, y_min


def select_representative_subjects(
        all_region_pairs,
        all_region_strengths,
        ref_region_pairs,
        ref_region_strengths,
        jaccard_th=0.45,
        rho_th=0.45,
        min_common=5
):
    """
    Select representative subjects based on:
    1) Region-pair structure similarity (Jaccard)
    2) Strength distribution similarity (Spearman on common pairs)
    """

    selected = []

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------
    def normalize_region_pair(pair_str):
        a, b = pair_str.split(' - ')
        return ' - '.join(sorted([a, b]))

    def jaccard_similarity(set_a, set_b):
        if not set_a and not set_b:
            return 1.0
        return len(set_a & set_b) / len(set_a | set_b)

    def strength_similarity_spearman(
            pairs_subj, strengths_subj,
            pairs_ref, strengths_ref
    ):
        """
        Compute Spearman correlation on common region-pairs.
        Deterministic and order-safe.
        """

        subj_map = {
            normalize_region_pair(p): float(v)
            for p, v in zip(pairs_subj, strengths_subj)
        }
        ref_map = {
            normalize_region_pair(p): float(v)
            for p, v in zip(pairs_ref, strengths_ref)
        }

        common_pairs = sorted(set(subj_map) & set(ref_map))

        if len(common_pairs) < min_common:
            return False, None

        subj_vals = [subj_map[p] for p in common_pairs]
        ref_vals = [ref_map[p] for p in common_pairs]

        rho, _ = spearmanr(subj_vals, ref_vals)

        if np.isnan(rho):
            return False, None

        return rho >= rho_th, rho

    # --------------------------------------------------
    # Main loop
    # --------------------------------------------------
    ref_pair_set = {
        normalize_region_pair(p) for p in ref_region_pairs
    }

    for subj in all_region_pairs:

        subj_pairs = all_region_pairs[subj]
        subj_strengths = all_region_strengths[subj]

        subj_pair_set = {
            normalize_region_pair(p) for p in subj_pairs
        }

        # ---- Step 1: Jaccard on structure
        J = jaccard_similarity(subj_pair_set, ref_pair_set)
        if J < jaccard_th:
            continue

        # ---- Step 2: Strength similarity on common pairs
        ok, rho = strength_similarity_spearman(
            subj_pairs, subj_strengths,
            ref_region_pairs, ref_region_strengths
        )

        if not ok:
            continue

        selected.append(subj)

    return selected


def plot_bar_and_circle(
        region_pairs,
        strengths,
        con_matrix,
        ch_names,
        y_min,
        subject,
        n_lines=10
):
    # =============================
    # 1. 创建统一 Figure
    # =============================
    fig = plt.figure(figsize=(15, 6))

    ax_bar = fig.add_subplot(1, 2, 1)  # 普通轴
    ax_circle = fig.add_subplot(1, 2, 2, polar=True)  # 关键：polar=True

    # =============================
    # 2. 左侧：柱状图
    # =============================
    ax_bar.bar(region_pairs, strengths)
    ax_bar.set_ylabel('Adjacency Strength')
    ax_bar.set_xlabel('Brain Region Connections')
    ax_bar.set_ylim(bottom=y_min)
    ax_bar.tick_params(axis='x', rotation=45)
    ax_bar.set_title('(a) Region-level Connections')

    # =============================
    # 3. 右侧：环形连接图
    # =============================
    plot_connectivity_circle(
        con_matrix,
        ch_names,
        n_lines=10,
        colormap='viridis_r',
        show=False,
        ax=ax_circle,
        facecolor='white',
        textcolor='black'
    )

    ax_circle.set_title('(b) Channel-level Connectivity')

    # =============================
    # 4. 调整 colorbar（保留你的做法）
    # =============================
    cbar = fig.axes[-1]  # plot_connectivity_circle 自动创建的 colorbar
    cbar.set_position([0.88, 0.2, 0.02, 0.3])  # (left, bottom, width, height)

    # =============================
    # 5. 整体标题与布局
    # =============================
    fig.suptitle(f"Subject {subject}", fontsize=14, y=0.92)

    fig.subplots_adjust(
        left=0.06,
        right=0.85,
        top=0.85,
        bottom=0.15,
        wspace=0.35
    )

    return fig


def CricleDrawer(data, dataset, band, subject):
    features_min = np.min(data)
    features_max = np.max(data)
    data = (data - features_min) / (features_max - features_min)
    # EEG 通道名称
    ch_names = ['Fp1', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5',
                'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'PO7', 'PO5', 'PO3', 'O1', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2',
                'AF4', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8',
                'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'PO8', 'PO6', 'PO4', 'O2', 'CB1', 'CB2']

    con_matrix = data
    con_matrix = (con_matrix + con_matrix.T) / 2  # 保持对称
    region_pairs_sorted, strengths_sorted, y_min = plot_topk_heset_connections_no_norm(
        find_top_correlated_pairs(con_matrix, ch_names))

    # fig = plot_bar_and_circle(region_pairs_sorted, strengths_sorted, con_matrix, ch_names, y_min, subject)

    # 绘制环形连接图
    fig, _ = plot_connectivity_circle(con_matrix, ch_names, n_lines=10,
                                      colormap='viridis_r', show=False,
                                      facecolor='white', textcolor='black')
    # 调整 colorbar 位置
    cbar = fig.axes[-1]  # 获取 colorbar 轴
    cbar.set_position([0.0, 0.0, 0.03, 0.2])  # 调整 (left, bottom, width, height)

    # 调整标题，使其更靠近主体
    fig.suptitle(f"Subject {subject}", fontsize=14, y=0.92)  # y 值越小，标题越接近主体

    # 调整整体布局，使环形图、标题、图例更紧凑
    fig.subplots_adjust(left=0.1, right=0.5, top=0.85, bottom=0.1)
    # fig.savefig(f'{dataset}_{band}.pdf', format="pdf", dpi=300)
    plt.show()
    # fig.savefig(f'{dataset}_{band}.png')


def CalcData(data):
    features_min = np.min(data)
    features_max = np.max(data)
    data = (data - features_min) / (features_max - features_min)
    # EEG 通道名称
    ch_names = ['Fp1', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5',
                'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'PO7', 'PO5', 'PO3', 'O1', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2',
                'AF4', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8',
                'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'PO8', 'PO6', 'PO4', 'O2', 'CB1', 'CB2']

    con_matrix = data
    con_matrix = (con_matrix + con_matrix.T) / 2  # 保持对称

    region_pairs_sorted, strengths_sorted, _ = plot_topk_heset_connections_no_norm(
        find_top_correlated_pairs(con_matrix, ch_names))

    return region_pairs_sorted, strengths_sorted


def save_grouped_data(subjectsa=None, subjectsb=None, a_filename='a_grouped_data.pkl', b_filename='b_grouped_data.pkl'):
    """\
    有序保存每个subject的a和b数据到不同组\
    \
    参数:\
        subjects: 受试者列表\
        a_filename: a组保存文件名\
        b_filename: b组保存文件名\
    """
    # 初始化分组容器
    a_group = []  # 保存所有subject的a数据
    b_group = []  # 保存所有subject的b数据

    if subjectsa is not None:
        for i, k in enumerate(subjectsa):
            # 获取当前subject的数据
            a, ag = evaluate_SEED.main(modelpath='2025_03_09_00_15_20', subject=([k]))
            # 切片处理
            a = a[:, :-3, :-3]
            # 添加到相应分组
            a_group.append({
                'subject': k,
                'data': a,  # 形状为(5, 62, 62)的数组
                'index': i  # 保持原始顺序
            })
        # 使用pickle有序保存数据
        with open(a_filename, 'wb') as a_file:
            pickle.dump(a_group, a_file)

    if subjectsb is not None:
        for i, k in enumerate(subjectsb):
            # 获取当前subject的数据
            b, bg = evaluate_SEEDIV.main(modelpath='2025_03_08_19_09_40', subject=([k]))
            # 切片处理
            b = b[:, :-3, :-3]
            # 添加到相应分组
            b_group.append({
                'subject': i,
                'data': b,  # 形状为(5, 62, 62)的数组
                'index': i  # 保持原始顺序
            })
        with open(b_filename, 'wb') as b_file:
            pickle.dump(b_group, b_file)

    print(f"数据保存成功! A组: {a_filename}, B组: {b_filename}")


def load_grouped_data():
    # 加载A组数据
    with open('a_grouped_data.pkl', 'rb') as f:
        a_data = pickle.load(f)

        # 获取a数据
        adata = []
        for data in a_data:
            adata.append(data['data'])
        adata = np.array(adata)
        print(f"data数据形状: {adata.shape}")

    # 加载B组数据
    with open('b_grouped_data.pkl', 'rb') as f:
        b_data = pickle.load(f)

        # 获取b数据
        # 获取a数据
        bdata = []
        for data in b_data:
            bdata.append(data['data'])
        bdata = np.array(bdata)
        print(f"data数据形状: {bdata.shape}")

    return adata, bdata


def analiz_sim(data, subject):
    all_region_pairs = {}
    all_region_strengths = {}
    #
    for i, k in enumerate(subject):
        temp = data[i]

        for j in range(0, 5):
            region_pairs_sorted, strengths_sorted = CalcData(temp[j])
            region_pairs_sorted, strengths_sorted = aggregate_region_pairs(region_pairs_sorted, strengths_sorted)
            all_region_pairs[f'{i}-{j}'] = region_pairs_sorted
            all_region_strengths[f'{i}-{j}'] = strengths_sorted
            # CricleDrawer(data[i][j], f'SEED_in_{i}', j, f'{i}-{j}')
    ref_region_pairs = all_region_pairs['3-3']
    ref_region_strengths = all_region_strengths['3-3']
    sel = select_representative_subjects(all_region_pairs, all_region_strengths, ref_region_pairs, ref_region_strengths)
    for subj_band in sel:
        subj, band = subj_band.split('-')
        subj = int(subj)
        band = int(band)
        CricleDrawer(data[subj][band], f'SEED_{subj}', band, subj)
        print(all_region_pairs[subj_band])
    print(sel)


if __name__ == "__main__":
    subjectsa = range(0, 15)
    subjectsb = range(0, 15)
    # save_grouped_data(subjectsa=subjectsa)
    a, b = load_grouped_data()
    analiz_sim(a, subjectsa)
