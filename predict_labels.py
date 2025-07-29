import pandas as pd
import joblib

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

def main():
    input_path = "data_to_predict/blackmirror_data.xlsx"  # Or whatever name you give it
    output_path = "predictions/predicted_blackmirror_data.xlsx"  # New file with predictions

    # Load all 3 models
    sentiment_model = load_model("models/sentiment_model.pkl")
    opinion_model = load_model("models/opinion_model.pkl")
    plausibility_model = load_model("models/plausibility_model.pkl")

    # Load both sheets
    xl = pd.read_excel(input_path, sheet_name=None)
    predicted_data = {}

    for sheet_name, df in xl.items():
        if "body" in df.columns:
            print(f"📄 Predicting labels for sheet: {sheet_name}")
            df_predicted = process_sheet(df, sentiment_model, opinion_model, plausibility_model)
            predicted_data[sheet_name] = df_predicted
        else:
            print(f"⚠️ Skipping sheet {sheet_name} — no 'body' column found.")

    # Save back to a new Excel file, with the same sheet names
    with pd.ExcelWriter(output_path) as writer:
        for sheet_name, df in predicted_data.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"✅ Predictions complete. Saved to: {output_path}")

if __name__ == "__main__":
    main()
