import pandas as pd

# Read the large UCI SMS dataset
data = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

# Remove empty rows
data = data.dropna()

# Save as our project's CSV format
data.to_csv("spam.csv", index=False)

print("Dataset converted successfully!")
print("Total messages:", len(data))

print("\nClass distribution:")
print(data["label"].value_counts())

print("\nFirst 5 messages:")
print(data.head())