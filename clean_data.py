# clean_data.py - turns the raw scraped file (cars.csv) into a clean, simple file (cars_clean.csv)
import pandas as pd

df = pd.read_csv("cars.csv")
print("Raw rows:", len(df))

# Remove duplicate rows (compare ALL raw columns, so different cars are not lost)
df = df.drop_duplicates()
print("After removing duplicates:", len(df))

# Keep only the columns we will use
df = df[["Price", "Make_Year", "CC", "Mileage", "Assembly", "Transmission"]]

# Remove impossible or extreme values (each rule is a documented assumption)
df = df[(df["Price"] >= 200_000) & (df["Price"] <= 20_000_000)]   # 2 lakh to 2 crore PKR
df = df[(df["Mileage"] >= 100) & (df["Mileage"] <= 500_000)]      # used cars, no 999999 placeholders
df = df[(df["CC"] >= 600) & (df["CC"] <= 6000)]                   # realistic engine sizes
print("After removing bad values:", len(df))

# Make a simple table of numbers the model can use
clean = pd.DataFrame({
    "year": df["Make_Year"],
    "cc": df["CC"],
    "mileage": df["Mileage"],
    "imported": (df["Assembly"] == "Imported").astype(int),        # 1 = imported, 0 = local
    "automatic": (df["Transmission"] == "Automatic").astype(int),  # 1 = automatic, 0 = manual
    "price_lakh": (df["Price"] / 100_000).round(2),                # lakh PKR, like the house lab
})

clean.to_csv("cars_clean.csv", index=False)
print("Saved cars_clean.csv with", len(clean), "rows")
print(clean.head())