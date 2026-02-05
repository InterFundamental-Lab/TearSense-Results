import os 
import json 
import shutil 
import pandas as pd 
import numpy as np 
import argparse 

def load_metrics (folder_path ):
    data =[]

    if not os .path .exists (folder_path ):
        print (f"[Error] Folder not found: {folder_path}")
        return pd .DataFrame ()

    files =[f for f in os .listdir (folder_path )if f .endswith ('.json')]
    print (f"[Info] Found {len(files)} JSON files. Loading all data...")

    for filename in files :
        filepath =os .path .join (folder_path ,filename )
        try :
            with open (filepath ,'r')as f :
                content =json .load (f )



            if "tearsense"in content :
                m =content ["tearsense"]

            elif "metrics"in content :
                m =content ["metrics"]

                if "AUROC"not in m and "value"in m .get ("AUROC",{}):
                     flat ={}
                     for k ,v in m .items ():
                         if isinstance (v ,dict )and "value"in v :
                             flat [k ]=v ["value"]
                         else :
                             flat [k ]=v 
                     m =flat 

            else :
                key =list (content .keys ())[0 ]
                m =content [key ]if isinstance (content [key ],dict )else content 


            entry ={
            'filename':filename ,
            'Model Serial':m .get ('model_serial',m .get ('serial',filename .replace ('.json',''))),
            'AUROC':m .get ('AUC',m .get ('AUROC',0 )),
            'Calibration Slope':m .get ('Calibration_Slope',0 ),
            'Brier':m .get ('Brier_Score',m .get ('Brier',1.0 )),
            'Net Benefit':m .get ('Net_Benefit',-1.0 ),
            'Sensitivity':m .get ('Sensitivity',m .get ('Sensitivity_Recall',0 )),
            'Specificity':m .get ('Specificity',0 ),
            'PPV':m .get ('PPV',m .get ('PPV_Precision',0 )),
            'NPV':m .get ('NPV',0 ),
            'F1':m .get ('F1',m .get ('F1_Score',0 ))
            }
            data .append (entry )

        except Exception as e :
            print (f"[Warning] Skipped {filename}: {e}")

    return pd .DataFrame (data )

def calculate_ranks (df ):
    """
    Ranks ALL models. 
    Returns the dataframe sorted by 'Mean Rank' (Best -> Worst).
    """
    ranks =pd .DataFrame ()
    ranks ['Model Serial']=df ['Model Serial']


    high_cols =['AUROC','Net Benefit','Sensitivity','Specificity','PPV','NPV','F1']
    for col in high_cols :
        ranks [f'{col}_rank']=df [col ].rank (ascending =False )


    ranks ['Brier_rank']=df ['Brier'].rank (ascending =True )


    slope_diff =np .abs (df ['Calibration Slope']-1.0 )
    ranks ['Slope_rank']=slope_diff .rank (ascending =True )


    rank_cols =[c for c in ranks .columns if '_rank'in c ]
    ranks ['Mean Rank']=ranks [rank_cols ].mean (axis =1 )


    df_scored =df .merge (ranks [['Model Serial','Mean Rank']],on ='Model Serial')


    return df_scored .sort_values ('Mean Rank',ascending =True )

def copy_top_models (df ,source_folder ):
    """Identifies and copies just the category winners to a 'top' folder."""
    top_folder =os .path .join (source_folder ,"top")
    if not os .path .exists (top_folder ):
        os .makedirs (top_folder )

    winners =set ()


    for col in ['AUROC','Net Benefit','F1']:
        winners .add (df .loc [df [col ].idxmax ()]['filename'])
    winners .add (df .loc [df ['Brier'].idxmin ()]['filename'])


    winners .add (df .iloc [0 ]['filename'])

    print (f"\n[Info] Copying {len(winners)} 'Category Winners' to {top_folder}...")
    for fname in winners :
        try :
            shutil .copy2 (os .path .join (source_folder ,fname ),os .path .join (top_folder ,fname ))
        except :pass 

def main (folder_path ):

    df =load_metrics (folder_path )
    if df .empty :return 


    ranked_df =calculate_ranks (df )


    output_csv =os .path .join (folder_path ,"model_rankings_full.csv")


    final_cols =[
    'Model Serial',
    'AUROC',
    'Calibration Slope',
    'Brier',
    'Net Benefit',
    'Sensitivity',
    'Specificity',
    'PPV',
    'NPV',
    'F1',
    'Mean Rank'
    ]

    ranked_df [final_cols ].to_csv (output_csv ,index =False )


    print ("\n"+"="*80 )
    print (f"✅ EXPORT COMPLETE: {output_csv}")
    print ("="*80 )
    print (f"Total Models Ranked: {len(ranked_df)}")
    print ("-"*80 )
    print (ranked_df [final_cols ].head (5 ).to_string (index =False ))
    print (f"\n... (and {len(ranked_df)-5} more rows in CSV)")
    print ("="*80 )


    copy_top_models (ranked_df ,folder_path )

if __name__ =="__main__":
    target_folder ="./external_assessor/metrics/"

    main (target_folder)