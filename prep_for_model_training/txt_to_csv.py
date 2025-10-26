import pandas as pd

# Step 1: Load the RockYou sample
with open("rockyou_100k.txt", "r", encoding="latin-1") as file:
    passwords = [line.strip() for line in file if line.strip()]

# Step 2: Create a DataFrame
df = pd.DataFrame(passwords, columns=["password"])

# Step 3: Save to CSV
df.to_csv("rockyou_100k.csv", index=False)
print(f"rockyou_100k.csv created with {len(df)} passwords")
