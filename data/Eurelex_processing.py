import numpy as np
import pandas as pd


with open('eurlex-ev-fold2-train.arff', 'r') as fp:
    file_content = fp.readlines()


def parse_row(line, len_row):
    line = line.replace('{', '').replace('}', '')

    row = np.zeros(len_row)
    for data in line.split(','):
        index, value = data.split()
        row[int(index)] = float(value)

    return row


columns = []
len_attr = len('@attribute')

# get the columns
for line in file_content:
    if line.startswith('@attribute '):
        col_name = line[len_attr:].split()[0]
        columns.append(col_name)

rows = []
len_row = len(columns)
# get the rows
for line in file_content:
    if line.startswith('{'):
        rows.append(parse_row(line, len_row))

df = pd.DataFrame(data=rows, columns=columns).T
indexes = df.index
dum_df = pd.DataFrame(indexes[:5000])
dum_df.to_csv("5000_EurlexWords.csv", index=False)