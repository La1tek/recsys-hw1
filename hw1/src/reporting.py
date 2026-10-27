from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np


COLORS = {"ink": "#17324D", "blue": "#3769B0", "teal": "#16968A",
          "purple": "#8262B6", "amber": "#D19330", "gray": "#8795A6",
          "light": "#C4CDD8", "grid": "#E4EAF0"}

MODEL_COLORS = {
    "ItemKNN": COLORS["amber"], "iALS": COLORS["blue"],
    "EASE": COLORS["purple"], "FM": COLORS["blue"],
    "RecencyALS": COLORS["teal"], "MostPopular": COLORS["gray"],
    "FM_ID": COLORS["light"],
}


def _compact_count(value, position=None):
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:g} млн"
    if abs(value) >= 1000:
        return f"{value / 1000:g} тыс."
    return f"{value:g}"


def _clean_axis(axis, grid="y"):
    axis.set_axisbelow(True)
    axis.grid(axis=grid, color=COLORS["grid"], linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["bottom", "left"]].set_color("#C8D3DE")
    axis.tick_params(colors=COLORS["ink"], labelsize=9)


def _save(figure, output, filename, root):
    path = output / filename
    figure.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path.relative_to(root).as_posix()


def _metadata_tokens(row, field):
    value = row.get(field)
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple, set, np.ndarray)) else [value]
    tokens = []
    for entry in values:
        text = str(entry).strip()
        if text and text.lower() not in {"none", "nan", "<missing>"}:
            tokens.append(text)
    return sorted(set(tokens))


def _eda_figure(data, output, root):
    user_counts = data["train_user_counts"]
    item_counts = data["train_item_counts"]
    fig, axes = plt.subplots(1, 3, figsize=(5.1 * 3, 4.1), squeeze=False)
    axes = axes.ravel()

    ax = axes[0]
    positive = user_counts[user_counts > 0]
    bins = np.unique(np.geomspace(positive.min(), positive.max() + 1, 25))
    ax.hist(positive, bins=bins, color=COLORS["blue"], edgecolor="white", linewidth=0.6)
    ax.set(xscale="log", yscale="log", xlabel="Взаимодействия пользователя в train",
           ylabel="Число пользователей", title="Активность пользователей")
    _clean_axis(ax)

    ax = axes[1]
    popularity = np.sort(item_counts[item_counts > 0])[::-1]
    ax.plot(np.arange(1, popularity.size + 1), popularity, color=COLORS["purple"], linewidth=2)
    ax.set(xscale="log", yscale="log", xlabel="Ранг игры по популярности",
           ylabel="Взаимодействия в train", title="Популярность игр")
    _clean_axis(ax)

    ax = axes[2]
    monthly = []
    for split, label, color in [("train", "Train", COLORS["blue"]),
                                ("val", "Validation", COLORS["teal"]),
                                ("test", "Test", COLORS["amber"])]:
        months = data[f"{split}_times"].astype(np.int64).astype("datetime64[s]").astype("datetime64[M]")
        dates, counts = np.unique(months, return_counts=True)
        monthly.append((label, dates, counts, color))
    all_months = np.arange(min(row[1].min() for row in monthly),
                           max(row[1].max() for row in monthly) + np.timedelta64(1, "M"),
                           dtype="datetime64[M]")
    bottom = np.zeros(all_months.size)
    for label, months, counts, color in monthly:
        heights = np.zeros(all_months.size)
        heights[np.searchsorted(all_months, months)] = counts
        ax.bar(np.arange(all_months.size), heights, bottom=bottom, width=0.95,
               color=color, label=label)
        bottom += heights
    ticks = np.unique(np.linspace(0, all_months.size - 1, min(7, all_months.size)).astype(int))
    ax.set_xticks(ticks, [str(month) for month in all_months[ticks]], rotation=35, ha="right")
    ax.set(xlabel="Месяц отзыва", ylabel="Число взаимодействий", title="Разбиение по времени")
    ax.yaxis.set_major_formatter(FuncFormatter(_compact_count))
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _clean_axis(ax)
    fig.suptitle("Steam V2 · структура подготовленной выборки", x=0.03, ha="left",
                 fontsize=16, fontweight="bold", color=COLORS["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=2.5)
    return _save(fig, output, "fig_01_data.png", root)


def _metadata_figure(data, output, root):
    metadata = data["item_metadata"]
    fields = ["genres", "developer", "publisher", "specs"]
    field_labels = ["Жанры", "Разработчик", "Издатель", "Возможности игры"]
    coverage = [sum(bool(_metadata_tokens(row, field)) for row in metadata) / len(metadata)
                for field in fields]
    genres = Counter(token for row in metadata for token in _metadata_tokens(row, "genres"))
    fig, axes = plt.subplots(1, 2 if genres else 1, figsize=(12 if genres else 6, 4.7), squeeze=False)
    ax = axes.ravel()[0]
    positions = np.arange(len(fields))
    ax.barh(positions, coverage, height=0.63, color=COLORS["teal"])
    ax.set_yticks(positions, field_labels)
    ax.invert_yaxis()
    ax.set(xlim=(0, 1.14), xlabel="Доля игр с заполненным полем", title="Покрытие метаданных каталога")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0%}"))
    for y, value in zip(positions, coverage):
        ax.text(value + 0.02, y, f"{value:.1%}", va="center", fontsize=10, color=COLORS["ink"])
    _clean_axis(ax, "x")
    if genres:
        ax = axes.ravel()[1]
        top = genres.most_common(10)[::-1]
        ax.barh([row[0] for row in top], [row[1] for row in top], color=COLORS["purple"], height=0.65)
        ax.set(xlabel="Число игр", title="Самые частые жанры")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10], integer=True))
        ax.xaxis.set_major_formatter(FuncFormatter(_compact_count))
        _clean_axis(ax, "x")
    fig.suptitle(f"Метаданные каталога · только EDA · {len(metadata):,} игр".replace(",", " "),
                 x=0.03, ha="left", fontsize=16, fontweight="bold", color=COLORS["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=3)
    return _save(fig, output, "fig_02_metadata.png", root)


def _final_figure(rows, output, root):
    if not rows:
        return None
    rows = sorted(rows, key=lambda row: float(row["ndcg@10"]), reverse=True)
    labels = [row["model"] for row in rows]
    colors = [MODEL_COLORS[label] for label in labels]
    fig, axes = plt.subplots(1, 2, figsize=(12.7, max(4.3, 0.48 * len(rows) + 1.4)), sharey=True)
    for ax, metric, title in zip(axes, ["ndcg@10", "recall@10"], ["NDCG@10", "Recall@10"]):
        values = np.array([float(row[metric]) for row in rows])
        ax.barh(np.arange(len(rows)), values, height=0.62, color=colors)
        ax.set_yticks(np.arange(len(rows)), labels)
        ax.set(title=title, xlabel="Среднее по пользователям", xlim=(0, max(values.max() * 1.26, 0.001)))
        ax.xaxis.set_major_locator(MaxNLocator(5))
        for i, value in enumerate(values):
            ax.text(value + max(values.max() * 0.025, 0.00001), i, f"{value:.4f}",
                    va="center", fontsize=9, color=COLORS["ink"])
        _clean_axis(ax, "x")
    axes[0].invert_yaxis()
    fig.suptitle("Итоговое сравнение на test", x=0.03, ha="left", fontsize=17,
                 fontweight="bold", color=COLORS["ink"])
    fig.text(0.03, 0.015, "Единый каталог · исключены известные игры · K = 10 · параметры выбраны на validation",
             fontsize=9, color="#597084")
    fig.tight_layout(rect=(0, 0.045, 1, 0.91), w_pad=3)
    return _save(fig, output, "fig_03_test_metrics.png", root)


def _short_params(params):
    names = {"factors": "d", "regularization": "λ", "alpha": "α", "learning_rate": "lr",
             "half_life_days": "T½", "neighbors": "k", "shrinkage": "shrink"}
    ignored = {"seed", "iterations", "epochs", "batch_size", "negative_ratio"}
    lines = [f"{names.get(key, key)}={value:g}" for key, value in params.items() if key not in ignored]
    return "\n".join(lines[:4])


def _validation_figure(rows, output, root):
    if not rows:
        return None
    groups = {}
    for row in rows:
        name = row["model"]
        if name in {"MostPopular", "FM_ID"}:
            continue
        groups.setdefault(name, []).append(row)
    if not groups:
        return None
    columns = min(3, len(groups))
    nrows = int(np.ceil(len(groups) / columns))
    fig, axes = plt.subplots(nrows, columns, figsize=(5 * columns, 4.3 * nrows), squeeze=False)
    for ax, (name, candidates) in zip(axes.ravel(), groups.items()):
        values = np.array([float(row["ndcg@10"]) for row in candidates])
        best = int(np.argmax(values))
        colors = [COLORS["light"]] * len(candidates)
        colors[best] = MODEL_COLORS[name]
        ax.bar(np.arange(len(candidates)), values, color=colors, width=0.64)
        labels = [f"Вариант {i + 1}\n{_short_params(row['params'])}"
                  for i, row in enumerate(candidates)]
        ax.set_xticks(np.arange(len(candidates)), labels, fontsize=8)
        ax.set(title=name, ylabel="NDCG@10", ylim=(0, max(values.max() * 1.25, 0.001)))
        for i, value in enumerate(values):
            ax.text(i, value + max(values.max() * 0.025, 0.00001), f"{value:.4f}",
                    ha="center", fontsize=9, color=COLORS["ink"])
        _clean_axis(ax)
    for ax in axes.ravel()[len(groups):]:
        ax.set_visible(False)
    fig.suptitle("Подбор параметров на validation", x=0.025, ha="left", fontsize=17,
                 fontweight="bold", color=COLORS["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.2, w_pad=2.5)
    return _save(fig, output, "fig_04_validation.png", root)


def _training_figure(rows, output, root):
    series = []
    for row in rows:
        history = row["history"]
        if not history:
            continue
        is_als = row["model"] in ("iALS", "RecencyALS")
        step, key = ("iteration", "objective") if is_als else ("epoch", "avg_loss")
        points = np.asarray([(entry[step], entry[key]) for entry in history], dtype=float)
        series.append((row["model"], points, key))
    if not series:
        return None
    als = [entry for entry in series if entry[2] == "objective"]
    fm = [entry for entry in series if entry[2] != "objective"]
    groups = [[entry] for entry in als] + ([fm] if fm else [])
    fig, axes = plt.subplots(1, len(groups), figsize=(5.2 * len(groups), 4.0), squeeze=False)
    for ax, group in zip(axes.ravel(), groups):
        is_als = group[0][2] == "objective"
        for name, points, key in group:
            ax.plot(points[:, 0], points[:, 1], marker="o", markersize=4,
                    linewidth=2, color=MODEL_COLORS[name], label=name)
        ax.set(title=group[0][0] if is_als else "FM · обучение с негативными примерами",
               xlabel="Итерация ALS" if is_als else "Эпоха",
               ylabel="Взвешенная ошибка + регуляризация" if is_als else "Средняя binary cross-entropy")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if is_als:
            ax.yaxis.set_major_formatter(FuncFormatter(_compact_count))
        else:
            ax.legend(frameon=False, fontsize=9)
        _clean_axis(ax)
    fig.suptitle("Сходимость при финальном обучении", x=0.03, ha="left", fontsize=17,
                 fontweight="bold", color=COLORS["ink"])
    fig.text(0.03, 0.01, "Разные целевые функции показаны на отдельных осях; величины потерь между моделями несопоставимы.",
             fontsize=9, color="#597084")
    fig.tight_layout(rect=(0, 0.05, 1, 0.9), w_pad=3)
    return _save(fig, output, "fig_05_training.png", root)


def make_figures(data, validation_rows, final_rows, root):
    root = Path(root).resolve()
    output = root / "artifacts"
    output.mkdir(parents=True, exist_ok=True)
    style = {"font.family": "DejaVu Sans", "font.size": 10,
             "axes.labelcolor": COLORS["ink"], "axes.titlecolor": COLORS["ink"],
             "axes.titlesize": 11, "axes.titleweight": "semibold",
             "figure.facecolor": "white", "axes.facecolor": "white",
             "savefig.facecolor": "white"}
    with plt.rc_context(style):
        figures = {
            "eda": _eda_figure(data, output, root),
            "metadata": _metadata_figure(data, output, root),
            "final_metrics": _final_figure(final_rows, output, root),
            "validation": _validation_figure(validation_rows, output, root),
            "training": _training_figure(final_rows, output, root),
        }
    return {name: path for name, path in figures.items() if path is not None}
