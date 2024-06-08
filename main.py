from config import INPUT_PATH, OUTPUT_PATH
from db_utils import get_df_from_db
import spacy
import pandas as pd

nlp = spacy.load("en_core_web_lg")

def read_file_to_list(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines]
    return lines

def get_probable_score(input, vehicle_row):
    input_vector=nlp(input)
    vehicle_vector=nlp(vehicle_row)
    return input_vector.similarity(vehicle_vector)

def write_output(df):
    with open(OUTPUT_PATH, 'w') as file:

        for _, row in df.iterrows():
            file.write(f"Input: {row['input']}\n")
            file.write(f"Vehicle ID: {row['id']}\n")
            file.write(f"Confidence: {round( row['possible_score']*10,1)}\n\n")

def get_output():
    inputs=read_file_to_list(INPUT_PATH)
    vehicle_df=    get_df_from_db("SELECT id, make, model, badge, transmission_type, fuel_type, drive_type FROM autograb_schema.vehicle")
    listings_df = get_df_from_db(" select id,vehicle_id,url,price,kms from autograb_schema.listing")

    vehicle_df["all_params"]=vehicle_df['make'] + ' ' + \
        vehicle_df['model'] + ' ' + \
        vehicle_df['badge'] + ' ' + \
        vehicle_df['transmission_type'] + ' ' + \
        vehicle_df['fuel_type'] + ' ' + \
        vehicle_df['drive_type']
    vehicle_df['score']=0
    
    input_df=pd.DataFrame(inputs,columns=['input'])
    input_df['key_temp']=1
    vehicle_df['key_temp']=1

    master_score_df=pd.merge(vehicle_df,input_df, on='key_temp').drop('key_temp', axis=1)
    print(master_score_df.columns)
    
    master_score_df['possible_score']=master_score_df.apply(lambda row: get_probable_score(row["input"], row['all_params']), axis=1)
    master_score_df=master_score_df.sort_values(by='possible_score',ascending=False)
    idx = master_score_df.groupby([ 'input'])['possible_score'].idxmax()
    max_score_df = master_score_df.loc[idx]
    return max_score_df

    # max_score_df .to_csv("output.csv")

if __name__=="__main__":
    max_score_df=get_output()
    write_output(max_score_df)