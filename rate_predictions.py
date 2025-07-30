import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from datetime import datetime

def evaluate_column(true, pred, label_name, sheet_name, wrong_predictions, df, output_dir):
    print(f"\n📊 Evaluation for: {label_name}")
    total = len(true)
    correct = sum(true == pred)
    accuracy = correct / total if total > 0 else 0
    print(f"✅ Accuracy: {accuracy:.2%} ({correct} out of {total})")

    # Detailed metrics
    report = classification_report(true, pred, output_dict=True, zero_division=0)
    print("\nDetailed Report:")
    print(classification_report(true, pred, zero_division=0))

    # F1 Score for macro average (equal class weight)
    f1 = f1_score(true, pred, average='macro', zero_division=0)
    print(f"⭐ Macro F1 Score: {f1:.2f}")

    # Confusion matrix
    labels = sorted(list(set(true) | set(pred)))
    cm = confusion_matrix(true, pred, labels=labels)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {label_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plot_name = os.path.join(output_dir, f"confusion_matrix_{sheet_name}_{label_name.lower().replace(' ', '_')}.png")
    plt.tight_layout()
    plt.savefig(plot_name)
    plt.close()
    print(f"🖼️ Confusion matrix saved as: {plot_name}")

    # Save detailed wrong predictions
    wrong_mask = true != pred
    if wrong_mask.any():
        wrong_df = df.loc[wrong_mask, ["body"]].copy()
        wrong_df["actual"] = true[wrong_mask].values
        wrong_df["predicted"] = pred[wrong_mask].values
        wrong_df["label_type"] = label_name
        wrong_df["sheet"] = sheet_name
        wrong_df["row_index"] = df.index[wrong_mask]
        wrong_predictions.append(wrong_df)

    return f1

def main():
    path = input("📁 Enter path to predictions file (e.g. predictions/predicted_blackmirror_data_1.xlsx): ").strip()

    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return

    # Create output directory
    base_output_dir = "rate_predictions_outputs"
    os.makedirs(base_output_dir, exist_ok=True)

    filename_base = os.path.splitext(os.path.basename(path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(base_output_dir, f"{filename_base}_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)

    xl = pd.read_excel(path, sheet_name=None)
    wrong_predictions = []
    label_f1_scores = {}

    for sheet_name, df in xl.items():
        print(f"\n🔍 Analyzing sheet: {sheet_name}")

        required_cols = [
            "expected_sentiment", "predicted_sentiment",
            "expected_opinion", "predicted_opinion",
            "expected_plausibility", "predicted_plausibility", "body"
        ]
        if not all(col in df.columns for col in required_cols):
            print("⚠️ Skipping sheet — missing expected/predicted/body columns.")
            continue

        f1_sent = evaluate_column(df["expected_sentiment"], df["predicted_sentiment"],
                                  "Sentiment", sheet_name, wrong_predictions, df, run_output_dir)
        f1_opinion = evaluate_column(df["expected_opinion"], df["predicted_opinion"],
                                     "Opinion Strength", sheet_name, wrong_predictions, df, run_output_dir)
        f1_plaus = evaluate_column(df["expected_plausibility"], df["predicted_plausibility"],
                                   "Plausibility", sheet_name, wrong_predictions, df, run_output_dir)

        label_f1_scores[sheet_name] = {
            "Sentiment": f1_sent,
            "Opinion Strength": f1_opinion,
            "Plausibility": f1_plaus
        }

    # Save wrong predictions to CSV
    if wrong_predictions:
        full_wrong_df = pd.concat(wrong_predictions, ignore_index=True)
        error_csv_path = os.path.join(run_output_dir, "prediction_errors.csv")
        full_wrong_df.to_csv(error_csv_path, index=False)
        print(f"\n❌ Wrong predictions saved to: {error_csv_path}")

    # Rank weakest F1s
    print("\n📉 Lowest Performing Labels (by F1 score):")
    for sheet, scores in label_f1_scores.items():
        sorted_labels = sorted(scores.items(), key=lambda x: x[1])
        for label, f1 in sorted_labels:
            print(f"{sheet} — {label}: F1 = {f1:.2f}")

    print(f"\n✅ Evaluation complete. All outputs saved in: {run_output_dir}")

if __name__ == "__main__":
    main()
