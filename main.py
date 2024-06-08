from config import INPUT_PATH, OUTPUT_PATH
from db_utils import get_df_from_db
import spacy
import pandas as pd

nlp = spacy.load("en_core_web_lg")

def read_file_to_list(file_path:str)->list[str]:
    with open(file_path, 'r') as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines]
    return lines

def write_output(df:pd.DataFrame):
    with open(OUTPUT_PATH, 'w') as file:
        for _, row in df.iterrows():
            file.write(f"Input: {row['input']}\n")
            file.write(f"Vehicle ID: {row['id']}\n")
            file.write(f"Confidence: {round( row['match_score']*10,1)}\n\n")

def get_vehicle_with_tokens_df():
    vehicle_df=    get_df_from_db("SELECT id, make, model, badge, transmission_type, fuel_type, drive_type FROM autograb_schema.vehicle")
    vehicle_df["all_params"]=vehicle_df['make'] + ' ' + \
        vehicle_df['model'] + ' ' + \
        vehicle_df['badge'] + ' ' + \
        vehicle_df['transmission_type'] + ' ' + \
        vehicle_df['fuel_type'] + ' ' + \
        vehicle_df['drive_type']
    vehicle_df["all_params_token"]=vehicle_df.apply(lambda row:nlp(row['all_params']), axis=1)
    print(type(vehicle_df["all_params_token"][0]))
    vehicle_df['score']=0
    return vehicle_df

def get_matching_scores_df():
    inputs=read_file_to_list(INPUT_PATH)
    vehicle_with_token_df=get_vehicle_with_tokens_df()
    
    input_df=pd.DataFrame(inputs,columns=['input'])
    input_df['input_token']=input_df.apply(lambda row:nlp(row['input']), axis=1)
    input_df['key_temp']=1
    vehicle_with_token_df['key_temp']=1

    master_score_df=pd.merge(vehicle_with_token_df,input_df, on='key_temp').drop('key_temp', axis=1)
    print(master_score_df.columns)
    
    master_score_df['match_score']=master_score_df.apply(lambda row: row['input_token'].similarity(row['all_params_token']), axis=1)
    master_score_df=master_score_df.sort_values(by='match_score',ascending=False)
    idx = master_score_df.groupby([ 'input'])['match_score'].idxmax()
    max_score_df = master_score_df.loc[idx]
    return max_score_df

if __name__=="__main__":
    max_score_df=get_matching_scores_df()
    write_output(max_score_df)
