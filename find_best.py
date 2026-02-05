import os 
import json 
import pandas as pd 
import numpy as np 
import argparse 

def load_metrics (folder_path ):
    data =[]

    if not os .path .exists (folder_path ):
        print (f"[Error] Folder not found: {folder_path}")
        return pd .DataFrame ()

    files =[f for f in os .listdir (folder_path )if f .endswith ('.json')]
    print (f"[Info] Found {len(files)} JSON files in {folder_path}...")

    for filename in files :
        filepath =os .path .join (folder_path ,filename )
        try :
            with open (filepath ,'r')as f :
                content =json .load (f )




            key =list (content .keys ())[0 ]
            metrics =content [key ]


            entry ={
            'filename':filename ,
            'serial':metrics .get ('model_serial',filename ),
            'AUC':metrics .get ('AUC',0 ),
            'Calibration_Slope':metrics .get ('Calibration_Slope',0 ),
            'Brier':metrics .get ('Brier_Score',1.0 ),
            'Net_Benefit':metrics .get ('Net_Benefit',-1.0 ),
            'Sensitivity':metrics .get ('Sensitivity_Recall',0 ),
            'Specificity':metrics .get ('Specificity',0 ),
            'PPV':metrics .get ('PPV_Precision',0 ),
            'NPV':metrics .get ('NPV',0 ),
            'F1':metrics .get ('F1_Score',0 )
            }
            data .append (entry )

        except Exception as e :
            print (f"[Warning] Could not parse {filename}: {e}")

    return pd .DataFrame (data )

def find_best_overall (df ):
    """
    Calculates 'Best Overall' using Mean Rank.
    1 = Best Rank. 
    Lower Mean Rank = Better overall performance across all metrics.
    """
    ranks =pd .DataFrame ()
    ranks ['serial']=df ['serial']


    high_is_better =['AUC','Net_Benefit','Sensitivity','Specificity','PPV','NPV','F1']
    for col in high_is_better :
        ranks [f'{col}_rank']=df [col ].rank (ascending =False )


    ranks ['Brier_rank']=df ['Brier'].rank (ascending =True )



    slope_diff =np .abs (df ['Calibration_Slope']-1.0 )
    ranks ['Slope_rank']=slope_diff .rank (ascending =True )


    rank_cols =[c for c in ranks .columns if '_rank'in c ]
    ranks ['mean_rank']=ranks [rank_cols ].mean (axis =1 )


    df_scored =df .merge (ranks [['serial','mean_rank']],on ='serial')
    return df_scored .sort_values ('mean_rank',ascending =True )

def main (folder_path ):
    df =load_metrics (folder_path )

    if df .empty :
        print ("No data found.")
        return 


    best_metrics ={}


    for col in ['AUC','Net_Benefit','Sensitivity','Specificity','PPV','NPV','F1']:
        best_row =df .loc [df [col ].idxmax ()]
        best_metrics [col ]=(best_row ['serial'],best_row [col ])


    best_brier =df .loc [df ['Brier'].idxmin ()]
    best_metrics ['Brier']=(best_brier ['serial'],best_brier ['Brier'])



    slope_dist =np .abs (df ['Calibration_Slope']-1.0 )
    best_slope_idx =slope_dist .idxmin ()
    best_slope =df .loc [best_slope_idx ]
    best_metrics ['Calibration_Slope']=(best_slope ['serial'],best_slope ['Calibration_Slope'])

    print ("\n"+"="*60 )
    print ("🏆 CHAMPIONS BY CATEGORY")
    print ("="*60 )
    for metric ,(serial ,score )in best_metrics .items ():

        if metric =='Calibration_Slope':
             print (f"{metric:<20} | {score:.4f} (Ideal: 1.0) | {serial}")
        elif metric =='Brier':
             print (f"{metric:<20} | {score:.4f} (Lower is better)| {serial}")
        else :
             print (f"{metric:<20} | {score:.4f} {'':<18} | {serial}")


    ranked_df =find_best_overall (df )
    winner =ranked_df .iloc [0 ]

    print ("\n"+"="*60 )
    print ("🌟 BEST OVERALL MODEL (Lowest Mean Rank)")
    print ("="*60 )
    print (f"Serial:      {winner['serial']}")
    print (f"Filename:    {winner['filename']}")
    print (f"Mean Rank:   {winner['mean_rank']:.2f} (Lower is better)")
    print ("-"*30 )
    print (f"AUC:         {winner['AUC']:.4f}")
    print (f"Brier:       {winner['Brier']:.4f}")
    print (f"Net Benefit: {winner['Net_Benefit']:.4f}")
    print (f"F1 Score:    {winner['F1']:.4f}")
    print (f"Calib Slope: {winner['Calibration_Slope']:.4f}")


    output_csv ="model_comparison_ranking.csv"
    ranked_df .to_csv (output_csv ,index =False )
    print (f"\n[Info] Full comparison table saved to: {output_csv}")

if __name__ =="__main__":

    target_folder ="./lr_external_assessment"


    parser =argparse .ArgumentParser ()
    parser .add_argument ("--folder",type =str ,default =target_folder ,help ="Path to folder containing JSONs")
    args =parser .parse_args ()

    main (args .folder )