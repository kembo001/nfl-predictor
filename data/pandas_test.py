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

packer_df = pd.read_csv("data/data.csv", index_col="week")
print(packer_df)
print()
print(packer_df.loc[5])

############### Selection ###############
# Selection by Column
print()
print(packer_df["home_team"])
print(packer_df[["home_team", "away_team", "gameday"]])
print()
# Selection by Row
print(packer_df.loc[1])
print(packer_df.loc[1:20, ["home_team", "away_team", "packers_result"]])
print()
# week_result = int(input('Which week result do you want to see?'))

# try:
#     print(packer_df.loc[week_result])
# except KeyError:
#     print(f'{week_result} not found')

print()
############### Filtering ###############

# Filtering = Keeping the rows that match a condition

wins_packers = packer_df[packer_df["packers_result"]== "W"]
print(wins_packers)

packers_score = packer_df[(packer_df["away_team"] == "GB") & 
                          (packer_df["packers_result"] == "W")]

print(packers_score)
print()
############### Aggreation ###############

# Aggregate functions = Reduces a set of values into a single summary value. Used to Summarize and analyze data. Often used with groupby() function

print(packer_df.mean(numeric_only = True))

print(packer_df["away_score"].mean())

group = packer_df.groupby("home_team")

print (group["home_score"].mean())
print()
############### Data Cleaning ###############

# Data Cleaning = The process of fixing/removing: Incomplete, incorrect, or irrelevant data. 75% of work done with Pandas is data cleaning

print(packer_df)
packer_df = packer_df.drop(columns = ["gameday"])
packer_df['home_team'] = packer_df['home_team'].replace({"GB": "Packers Home"})
packer_df['away_team'] = packer_df['away_team'].replace({"GB": "Packers Away"})
packer_df['packers_result'] = packer_df['packers_result'].replace({"W": "Win", "L": "Lost"})

print(packer_df)
