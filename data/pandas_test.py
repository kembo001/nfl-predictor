import pandas as pd

print("Pandas version:", pd.__version__)

############### Series ###############

# Series = A Pandas 1-dimensional labeled array that can hold any data type
# Think of it like a single column in a spreadsheet (1-Dimensional)

data = [100, 102, 104]

series = pd.Series(data, index =['a', 'b', 'c'])

print(series)

calories = pd.Series({"Day 1": 1750, "Day 2": 2100, "Day 3": 1700})

print(calories)
print(f'You ate {calories.loc["Day 2"]} on day 2')

calories.loc["Day 3"] += 500

print(calories)
print(calories[calories >= 2000])

############### DataFrame ###############

# DataFrame = A tabular data structure with rows AND columns. (2 Dimensional) Similar to Excel spreadsheet
data2 = {"Name": ["Spongebob", "Patrick", "Squidward"],
         "Age": [30, 35, 50]
         }
df = pd.DataFrame(data2, index=["Char 1", "Char 2", "Char 3"])

print(df)
print()
print(df.loc["Char 2"])

# Add a new column
df["Job"] = ["Cook", "N/A", "Cashier"]
print()
print(df)

# Add a new Row
print()
new_row = pd.DataFrame([{"Name": "Sandy", "Age": 28, "Job": "Engineer"}],
                       index=["Char 4"])
df = pd.concat([df, new_row])
print(df)
print()
############### Importing ###############
packer_df = pd.read_csv("data/data.csv")
print(packer_df)
print()
print(packer_df.loc[5])
    
