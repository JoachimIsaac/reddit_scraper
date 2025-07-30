import pandas as pd
import joblib
import os

def load_model(path):
    return joblib.load(path)

def predict(df, model, label_name):
    df[label_name] = model.predict(df['body'])
    return df

def process_sheet(sheet_df, sentiment_model, opinion_model, plausibility_model):
    sheet_df = sheet_df.dropna(subset=["body"])
    sheet_df = predict(sheet_df, sentiment_model, "predicted_sentiment")
    sheet_df = predict(sheet_df, opinion_model, "predicted_opinion")
    sheet_df = predict(sheet_df, plausibility_model, "predicted_plausibility")
    return sheet_df

def get_next_output_path(base_path):
    base_name = os.path.splitext(os.path.basename(base_path))[0]
    folder = os.path.dirname(base_path)
    counter = 1

    while True:
        numbered_path = os.path.join(folder, f"{base_name}_{counter}.xlsx")
        if not os.path.exists(numbered_path):
            return numbered_path
        counter += 1

def main():
    # Prompt user for input file path
    input_path = input("Enter the path to the Excel file you want to label (e.g. data_to_predict/curated_examples.xlsx): ").strip()

    if not os.path.exists(input_path):
        print(f"❌ File not found: {input_path}")
        return

    # Load models
    sentiment_model = load_model("models/sentiment_model.pkl")
    opinion_model = load_model("models/opinion_model.pkl")
    plausibility_model = load_model("models/plausibility_model.pkl")

    # Load sheets
    xl = pd.read_excel(input_path, sheet_name=None)
    predicted_data = {}

    for sheet_name, df in xl.items():
        if "body" in df.columns:
            print(f"📄 Predicting labels for sheet: {sheet_name}")
            df_predicted = process_sheet(df, sentiment_model, opinion_model, plausibility_model)
            predicted_data[sheet_name] = df_predicted
        else:
            print(f"⚠️ Skipping sheet {sheet_name} — no 'body' column found.")

    # Determine new output path
    base_output_path = "predictions/predicted_blackmirror_data.xlsx"
    output_path = get_next_output_path(base_output_path)

    # Save predictions
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with pd.ExcelWriter(output_path) as writer:
        for sheet_name, df in predicted_data.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"✅ Predictions complete. Saved to: {output_path}")

if __name__ == "__main__":
    main()
