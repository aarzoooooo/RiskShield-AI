import pandas as pd
import os

INPUT_FILE = "data/rba-dataset.csv"
OUTPUT_FILE = "data/model_data.csv"

positive_rows = []
negative_rows = []

for chunk in pd.read_csv(INPUT_FILE, chunksize=100000):

    positives = chunk[chunk["Is Account Takeover"] == True]
    negatives = chunk[chunk["Is Account Takeover"] == False]

    positive_rows.append(positives)

    # Take a small random sample of normal logins from each chunk
    negative_rows.append(
        negatives.sample(
            n=min(500, len(negatives)),
            random_state=42
        )
    )

    print(
        f"Processed chunk | Takeovers found: {sum(len(x) for x in positive_rows)}")


# Combine everything
positive_df = pd.concat(positive_rows, ignore_index=True)
negative_df = pd.concat(negative_rows, ignore_index=True)

# Keep all takeover cases + sampled normal cases
model_df = pd.concat(
    [positive_df, negative_df],
    ignore_index=True
)

# Shuffle
model_df = model_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

os.makedirs("data", exist_ok=True)

model_df.to_csv(OUTPUT_FILE, index=False)

print("\nDONE!")
print("Dataset shape:", model_df.shape)
print("\nTarget distribution:")
print(model_df["Is Account Takeover"].value_counts())
