import argparse
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

try:
    from joblib.externals.loky.backend import context as loky_context

    loky_context.physical_cores_cache = os.cpu_count() or 1
except Exception:
    pass


BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from core.predictor import predict


DATASET_PATH = BASE_DIR / "data" / "processed" / "final_dataset_line1_8_230101-241231.csv"
FAST_DEFAULT_MODELS = ["LightGBM", "XGBoost", "RandomForest"]

warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")
warnings.filterwarnings("ignore", message=".*Changing updater.*")
warnings.filterwarnings("ignore", message=".*No visible GPU.*")
warnings.filterwarnings("ignore", message=".*Device is changed from GPU to CPU.*")
warnings.filterwarnings("ignore", message=".*Could not find the number of physical cores.*")
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def load_sample(line_name, sample_size, seed, date_from):
    df = pd.read_csv(DATASET_PATH, parse_dates=["날짜"])
    required_cols = ["날짜", "역명", "호선", "시간", "승차인원", "하차인원", "기온", "강수량", "적설"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing columns: {', '.join(missing_cols)}")

    date_from = pd.Timestamp(date_from)
    matched = df[(df["호선"].astype(str) == line_name) & (df["날짜"] >= date_from)].copy()
    if matched.empty:
        return matched

    sample_count = min(sample_size, len(matched))
    return matched.sample(sample_count, random_state=seed).reset_index(drop=True)


def load_fast_models(model_names):
    models = {}

    if "LightGBM" in model_names:
        try:
            board_pack = joblib.load(BASE_DIR / "models" / "lightgbm" / "lgb_boadin_model_line1_8.pkl")
            alight_pack = joblib.load(BASE_DIR / "models" / "lightgbm" / "lgb_alight_model_line1_8.pkl")
            models["LightGBM"] = {
                "board": board_pack["model"],
                "alight": alight_pack["model"],
                "columns": list(board_pack.get("columns", board_pack["model"].feature_name_)),
                "loaded": True,
            }
        except Exception as exc:
            print(f"[WARN] LightGBM load failed: {exc}", flush=True)
            models["LightGBM"] = {"loaded": False}

    if "XGBoost" in model_names:
        try:
            board_model = joblib.load(BASE_DIR / "models" / "xgboost" / "xgb_board_model_line1_8.pkl")
            alight_model = joblib.load(BASE_DIR / "models" / "xgboost" / "xgb_alight_model_line1_8.pkl")
            le_station = joblib.load(BASE_DIR / "models" / "xgboost" / "label_encoder_station_line1_8.pkl")
            le_line = joblib.load(BASE_DIR / "models" / "xgboost" / "label_encoder_line_line1_8.pkl")
            models["XGBoost"] = {
                "board": board_model,
                "alight": alight_model,
                "le_station": le_station,
                "le_line": le_line,
                "columns": list(getattr(board_model, "feature_names_in_", [])),
                "loaded": True,
            }
        except Exception as exc:
            print(f"[WARN] XGBoost load failed: {exc}", flush=True)
            models["XGBoost"] = {"loaded": False}

    if "RandomForest" in model_names:
        try:
            board_model = joblib.load(BASE_DIR / "models" / "randomforest" / "randomforest_boarding_model.pkl")
            alight_model = joblib.load(BASE_DIR / "models" / "randomforest" / "randomforest_dropoff_model.pkl")
            if hasattr(board_model, "set_params"):
                board_model.set_params(n_jobs=1)
            if hasattr(alight_model, "set_params"):
                alight_model.set_params(n_jobs=1)
            columns = joblib.load(BASE_DIR / "models" / "randomforest" / "model_columns.pkl")
            le_station = joblib.load(BASE_DIR / "models" / "randomforest" / "station_encoder.pkl")
            le_line = joblib.load(BASE_DIR / "models" / "randomforest" / "line_encoder.pkl")
            models["RandomForest"] = {
                "board": board_model,
                "alight": alight_model,
                "columns": list(columns),
                "le_station": le_station,
                "le_line": le_line,
                "loaded": True,
            }
        except Exception as exc:
            print(f"[WARN] RandomForest load failed: {exc}", flush=True)
            models["RandomForest"] = {"loaded": False}

    return models


def load_lstm_models_if_needed(model_names):
    if "LSTM" not in model_names:
        return {}, None, None

    # The app loader already knows the LSTM numpy backend details, so reuse it only when requested.
    from core.model_loader import load_all_models, load_lstm_base_dataset

    all_models, le_station, loaded = load_all_models()
    if not loaded or not all_models.get("LSTM", {}).get("loaded", False):
        return {"LSTM": {"loaded": False}}, le_station, None
    return {"LSTM": all_models["LSTM"]}, le_station, load_lstm_base_dataset()


def load_models(model_names):
    fast_names = [name for name in model_names if name != "LSTM"]
    models = load_fast_models(fast_names)
    lstm_models, le_station, lstm_base_df = load_lstm_models_if_needed(model_names)
    models.update(lstm_models)
    return models, le_station, lstm_base_df


def evaluate_model(model_name, sample, all_models, le_station, lstm_base_df):
    board_true = []
    board_pred = []
    alight_true = []
    alight_pred = []
    failed = 0

    for _, row in sample.iterrows():
        station_key = f"{row['역명']}_{row['호선']}"
        try:
            dt = pd.to_datetime(row["날짜"]).to_pydatetime()
            board, alight = predict(
                station_key,
                dt,
                int(row["시간"]),
                float(row["기온"]),
                float(row["강수량"]),
                float(row["적설"]),
                model_name=model_name,
                all_models=all_models,
                le_station=le_station,
                lstm_base_df=lstm_base_df,
            )
            board_true.append(float(row["승차인원"]))
            board_pred.append(float(board))
            alight_true.append(float(row["하차인원"]))
            alight_pred.append(float(alight))
        except Exception:
            failed += 1

    if not board_true:
        return {
            "model": model_name,
            "board_mae": np.nan,
            "alight_mae": np.nan,
            "avg_mae": np.nan,
            "evaluated_rows": 0,
            "failed_rows": failed,
        }

    board_mae = mae(board_true, board_pred)
    alight_mae = mae(alight_true, alight_pred)
    return {
        "model": model_name,
        "board_mae": board_mae,
        "alight_mae": alight_mae,
        "avg_mae": (board_mae + alight_mae) / 2,
        "evaluated_rows": len(board_true),
        "failed_rows": failed,
    }


def format_result_table(results):
    df = pd.DataFrame(results).sort_values("avg_mae", na_position="last").reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def print_table(table):
    print(
        table.to_string(
            index=False,
            formatters={
                "board_mae": lambda v: "-" if pd.isna(v) else f"{v:,.2f}",
                "alight_mae": lambda v: "-" if pd.isna(v) else f"{v:,.2f}",
                "avg_mae": lambda v: "-" if pd.isna(v) else f"{v:,.2f}",
            },
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate 1-8 line model rankings by line.")
    parser.add_argument("--lines", nargs="+", default=["1호선", "2호선", "4호선"], help="Lines to evaluate.")
    parser.add_argument("--sample-size", type=int, default=100, help="Random sample size per line.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Evaluate multiple random seeds.")
    parser.add_argument("--date-from", default="2024-12-15", help="Validation start date.")
    parser.add_argument("--models", nargs="+", default=None, help="Specific models to evaluate.")
    parser.add_argument("--include-lstm", action="store_true", help="Include LSTM evaluation. This can be much slower.")
    args = parser.parse_args()

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    model_names = args.models if args.models is not None else FAST_DEFAULT_MODELS.copy()
    if args.include_lstm and "LSTM" not in model_names:
        model_names.append("LSTM")

    all_models, le_station, lstm_base_df = load_models(model_names)
    skipped = [name for name in model_names if not all_models.get(name, {}).get("loaded", False)]
    model_names = [name for name in model_names if all_models.get(name, {}).get("loaded", False)]
    if not model_names:
        raise RuntimeError("No requested models loaded.")

    seeds = args.seeds if args.seeds is not None else [args.seed]

    if skipped:
        print(f"Skipped unloaded models: {', '.join(skipped)}", flush=True)
    print(f"Models: {', '.join(model_names)}", flush=True)
    print(f"Lines: {', '.join(args.lines)}", flush=True)
    print(f"Sample: {args.sample_size} rows per line, date_from={args.date_from}, seeds={seeds}", flush=True)
    print(flush=True)

    summary_rows = []
    for seed in seeds:
        print(f"### seed={seed} ###", flush=True)
        for line_name in args.lines:
            sample = load_sample(line_name, args.sample_size, seed, args.date_from)
            print(f"=== {line_name} ===", flush=True)
            if sample.empty:
                print("No sample rows.", flush=True)
                print(flush=True)
                continue

            results = []
            for model_name in model_names:
                print(f"  - Evaluating {model_name}...", flush=True)
                result = evaluate_model(model_name, sample, all_models, le_station, lstm_base_df)
                results.append(result)
                avg_text = "-" if pd.isna(result["avg_mae"]) else f"{result['avg_mae']:,.2f}"
                print(f"    avg_mae={avg_text}, failed_rows={result['failed_rows']}", flush=True)

            table = format_result_table(results)
            print_table(table)
            print(flush=True)

            for _, row in table.iterrows():
                summary_rows.append(
                    {
                        "seed": seed,
                        "line": line_name,
                        "rank": int(row["rank"]),
                        "model": row["model"],
                        "board_mae": row["board_mae"],
                        "alight_mae": row["alight_mae"],
                        "avg_mae": row["avg_mae"],
                    }
                )

    if not summary_rows:
        return

    summary = pd.DataFrame(summary_rows)
    print("=== Summary Ranking ===", flush=True)
    print(
        summary[["seed", "line", "rank", "model", "avg_mae"]].to_string(
            index=False,
            formatters={"avg_mae": lambda v: "-" if pd.isna(v) else f"{v:,.2f}"},
        ),
        flush=True,
    )

    overall = (
        summary.groupby("model", as_index=False)
        .agg(
            mean_rank=("rank", "mean"),
            first_place_count=("rank", lambda s: int((s == 1).sum())),
            mean_avg_mae=("avg_mae", "mean"),
        )
        .sort_values(["mean_rank", "mean_avg_mae"])
        .reset_index(drop=True)
    )
    print(flush=True)
    print("=== Overall ===", flush=True)
    print(
        overall.to_string(
            index=False,
            formatters={
                "mean_rank": lambda v: f"{v:,.2f}",
                "mean_avg_mae": lambda v: "-" if pd.isna(v) else f"{v:,.2f}",
            },
        ),
        flush=True,
    )

    patterns = {}
    for (seed, line), group in summary.groupby(["seed", "line"]):
        ranking = tuple(group.sort_values("rank")["model"].tolist())
        patterns[(seed, line)] = ranking

    unique_patterns = sorted(set(patterns.values()))
    print(flush=True)
    if len(unique_patterns) == 1:
        print(f"All evaluated line/seed samples share the same ranking: {' > '.join(unique_patterns[0])}", flush=True)
    else:
        print("Ranking differs by line/seed sample:", flush=True)
        for key, ranking in patterns.items():
            print(f"  {key}: {' > '.join(ranking)}", flush=True)


if __name__ == "__main__":
    main()
