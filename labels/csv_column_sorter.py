# importing pandas package
import pandas as pandasForSortingCSV
import glob

files = glob.glob("something-something-v1-*")
print(files)
sort_head = "label"

for file in files:
    # assign dataset
    csvData = pandasForSortingCSV.read_csv(file)

    axis = 0
    for i,head in enumerate(csvData.head()):
        if head==sort_head:
            axis=i
            break

    # sort data frame
    csvData.sort_values([sort_head],
                        axis=axis,
                        ascending=[True],
                        inplace=True)

    # displaying sorted data frame
    csvData.to_csv(file, index=False)
