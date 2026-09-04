import pandas as pd
import os

INPUT_FILE = "data/model_data.csv"
OUTPUT_FILE = "data/features.csv"

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

# -----------------------------
# 1. Convert timestamp
# -----------------------------

df["Login Timestamp"] = pd.to_datetime(
    df["Login Timestamp"],
    errors="coerce"
)

# Sort chronologically for each user
df = df.sort_values(
    ["User ID", "Login Timestamp"]
).reset_index(drop=True)

# -----------------------------
# 2. Time-based features
# -----------------------------

df["login_hour"] = df["Login Timestamp"].dt.hour
df["login_day"] = df["Login Timestamp"].dt.dayofweek
df["is_weekend"] = (df["login_day"] >= 5).astype(int)

# -----------------------------
# 3. New device detection
# -----------------------------

df["new_device"] = (
    ~df.duplicated(
        subset=["User ID", "Device Type"],
        keep="first"
    )
).astype(int)

# -----------------------------
# 4. New browser detection
# -----------------------------

df["new_browser"] = (
    ~df.duplicated(
        subset=["User ID", "Browser Name and Version"],
        keep="first"
    )
).astype(int)

# -----------------------------
# 5. New OS detection
# -----------------------------

df["new_os"] = (
    ~df.duplicated(
        subset=["User ID", "OS Name and Version"],
        keep="first"
    )
).astype(int)

# -----------------------------
# 6. New country detection
# -----------------------------

df["new_country"] = (
    ~df.duplicated(
        subset=["User ID", "Country"],
        keep="first"
    )
).astype(int)

# -----------------------------
# 7. New city detection
# -----------------------------

df["new_city"] = (
    ~df.duplicated(
        subset=["User ID", "City"],
        keep="first"
    )
).astype(int)

# -----------------------------
# 8. New IP detection
# -----------------------------

df["new_ip"] = (
    ~df.duplicated(
        subset=["User ID", "IP Address"],
        keep="first"
    )
).astype(int)

# -----------------------------
# 9. Device + browser combination
# -----------------------------

df["device_browser"] = (
    df["Device Type"].astype(str)
    + "_"
    + df["Browser Name and Version"].astype(str)
)

df["new_device_browser"] = (
    ~df.duplicated(
        subset=["User ID", "device_browser"],
        keep="first"
    )
).astype(int)

# -----------------------------
# 10. Network feature
# -----------------------------

df["Round-Trip Time [ms]"] = pd.to_numeric(
    df["Round-Trip Time [ms]"],
    errors="coerce"
)

# -----------------------------
# 11. Select final features
# -----------------------------

features = [
    "login_hour",
    "login_day",
    "is_weekend",

    "Round-Trip Time [ms]",

    "new_device",
    "new_browser",
    "new_os",
    "new_country",
    "new_city",
    "new_ip",
    "new_device_browser",

    "Login Successful",
    "Is Attack IP",

    "Is Account Takeover"
]

df = df[features]

# Remove missing target rows
df = df.dropna(
    subset=["Is Account Takeover"]
)

# -----------------------------
# 12. Save
# -----------------------------

os.makedirs("data", exist_ok=True)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nDONE!")
print("Dataset shape:", df.shape)

print("\nTarget distribution:")
print(
    df["Is Account Takeover"].value_counts()
)

print("\nNew-device distribution:")
print(
    df["new_device"].value_counts()
)

print("\nNew-country distribution:")
print(
    df["new_country"].value_counts()
)
