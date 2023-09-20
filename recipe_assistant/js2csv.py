
import json
import os
import pandas as pd

os.chdir('C:/Users/ermartin/OneDrive - DKV/Escritorio/')
num_rows = 1000

# Open json file:
f = open('./Asistente_Recetas/recipe_assistant/recipes.json', encoding="utf8")
# Apply load
data = json.load(f)

#Check that works:
# for i in data[0:10]:
#     print(i['id'])
#     print(i['Instructions'])
#     print('\n')

#To csv
data_csv = pd.DataFrame(data)

data_csv.drop(columns=['id'], inplace=True)
data_csv.rename(columns = {'Instructions':'Recipe'}, inplace = True)

data_csv = data_csv.sample(n = num_rows, replace = False)
print(data_csv)

#Save
data_csv.to_csv('./Asistente_Recetas/recipe_assistant/recipes.csv', index=False, encoding='utf-8', header=True)