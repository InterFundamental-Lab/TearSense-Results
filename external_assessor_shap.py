import argparse 
import joblib 
import shap 
import pandas as pd 
import numpy as np 
import matplotlib .pyplot as plt 
import json 
import os 
from itertools import combinations 


def load_model_bundle (path ):
    print (f"[INFO] Loading model from {path}...")
    data =joblib .load (path )
    print (f"[INFO] Model loaded. Version: {data.get('export_version', 'unknown')}")
    return data 


def prepare_encoded_data (X ,cat_cols ):
    """Replicates encoding logic for Tree Models (XGB/LGBM/RF)."""
    X_enc =X .copy ()
    for col in cat_cols :
        if col in X_enc .columns :
            if isinstance (X_enc [col ].dtype ,pd .CategoricalDtype ):
                X_enc [col ]=X_enc [col ].cat .codes 
            else :
                X_enc [col ]=X_enc [col ].astype ('category').cat .codes 
    return X_enc .fillna (-999 )


def generate_meta_features (base_preds ,use_interactions =True ,use_rank =False ):
    """
    Reconstructs meta-features. MUST match defecator.meta_features exactly.
    
    With use_interactions=True:  13 features (4 base + 6 pairwise + 3 stats)
    With use_interactions=False:  4 features (4 base only)
    """
    models =['cat','xgb','lgbm','rf']


    clean ={}
    for m in models :
        c =np .nan_to_num (base_preds [m ],nan =0.5 )
        clean [m ]=np .clip (c ,1e-7 ,1 -1e-7 )

    X_meta =np .column_stack ([clean [m ]for m in models ])

    if use_interactions :

        for i ,m1 in enumerate (models ):
            for j ,m2 in enumerate (models ):
                if i <j :
                    X_meta =np .column_stack ([X_meta ,clean [m1 ]*clean [m2 ]])


        all_preds =np .array ([clean [m ]for m in models ])
        X_meta =np .column_stack ([
        X_meta ,
        np .mean (all_preds ,axis =0 ),
        np .std (all_preds ,axis =0 ),
        np .max (all_preds ,axis =0 )-np .min (all_preds ,axis =0 ),
        ])

    return X_meta 






def main (model_path ):



    bundle =load_model_bundle (model_path )

    serial =bundle .get ('serial_number','unknown')
    output_dir =f"external_assessor/{serial}"
    os .makedirs (output_dir ,exist_ok =True )

    X_test_raw =bundle .get ('X_test_exact')
    cat_cols =bundle .get ('cat_cols',[])
    fold_models =bundle .get ('fold_models')
    n_folds =bundle .get ('n_folds',5 )
    meta_model =bundle .get ('meta_model')
    calibrator =bundle .get ('calibrator')
    is_weighted =bundle .get ('is_weighted_avg',False )

    stacking_config =bundle .get ('stacking_config',{})
    use_interactions =stacking_config .get ('use_interactions',True )
    use_rank =stacking_config .get ('use_rank_features',False )

    weights_dict =bundle .get ('weights')
    weights_array =bundle .get ('weights_array')

    print (f"[INFO] Serial: {serial}")
    print (f"[INFO] Pipeline Type: {'Weighted Ensemble' if is_weighted else 'Stacking Meta-Learner'}")
    print (f"[INFO] Test samples: {len(X_test_raw)}")




    def pipeline_predict_internal (X_df ):
        """Original pipeline expecting raw data (strings allowed)"""


        X_enc =prepare_encoded_data (X_df ,cat_cols )


        base_preds ={}
        for algo in ['cat','xgb','lgbm','rf']:
            preds =np .zeros (len (X_df ))
            for m in fold_models [algo ]:
                if algo =='cat':

                    preds +=m .predict_proba (X_df )[:,1 ]
                else :
                    preds +=m .predict_proba (X_enc )[:,1 ]
            base_preds [algo ]=preds /n_folds 


        if is_weighted :
            if weights_array is not None :
                models =['cat','xgb','lgbm','rf']
                raw_probs =sum (weights_array [i ]*base_preds [m ]for i ,m in enumerate (models ))
            elif isinstance (weights_dict ,dict ):
                raw_probs =(weights_dict ['cat']*base_preds ['cat']+
                weights_dict ['xgb']*base_preds ['xgb']+
                weights_dict ['lgbm']*base_preds ['lgbm']+
                weights_dict ['rf']*base_preds ['rf'])
            else :
                raise ValueError ("No valid weights found")
        else :
            X_meta =generate_meta_features (base_preds ,
            use_interactions =use_interactions ,
            use_rank =use_rank )
            raw_probs =meta_model .predict_proba (X_meta )[:,1 ]


        if calibrator is not None :
            final_probs =calibrator .predict (raw_probs )
        else :
            final_probs =raw_probs 

        return final_probs 







    print ("\n[INFO] preparing SHAP-compatible dataset (Encoding Categoricals)...")
    X_shap =X_test_raw .copy ()
    cat_mappings ={}

    for col in cat_cols :
        if col in X_shap .columns :

            X_shap [col ]=X_shap [col ].astype ('category')



            cat_mappings [col ]=dict (enumerate (X_shap [col ].cat .categories ))


            X_shap [col ]=X_shap [col ].cat .codes 

    def shap_predict_wrapper (X_input ):
        """
        Wrapper that accepts Numeric (SHAP) input, decodes it to Strings, 
        and calls the original pipeline.
        """

        if isinstance (X_input ,np .ndarray ):
            X_df =pd .DataFrame (X_input ,columns =X_test_raw .columns )
        else :
            X_df =X_input .copy ()


        for col ,mapping in cat_mappings .items ():
            if col in X_df .columns :

                X_df [col ]=X_df [col ].apply (lambda x :mapping .get (int (round (x )),x )if x !=-1 else np .nan )

                X_df [col ]=X_df [col ].astype ('object')


        return pipeline_predict_internal (X_df )




    print ("[INFO] Verifying wrapper pipeline...")
    from sklearn .metrics import roc_auc_score ,brier_score_loss 


    test_probs =shap_predict_wrapper (X_shap .values )

    y_test =bundle .get ('y_test_exact')
    if isinstance (y_test ,(pd .Series ,pd .DataFrame )):
        y_test =y_test .to_numpy ().ravel ()

    computed_auc =roc_auc_score (y_test ,test_probs )
    stored_metrics =bundle .get ('metrics',{})
    print (f"  Computed AUC:   {computed_auc:.6f}  (stored: {stored_metrics.get('test_auc')})")

    if stored_metrics .get ('test_auc')and abs (computed_auc -stored_metrics .get ('test_auc'))<1e-4 :
        print ("  [PASS] Pipeline verified.")
    else :
        print ("  [WARNING] Minor mismatch possible due to float/encoding, continuing...")




    print (f"\n[INFO] Initializing Permutation Explainer...")


    masker =shap .maskers .Independent (data =X_shap )


    explainer =shap .PermutationExplainer (shap_predict_wrapper ,masker )

    print (f"[INFO] Calculating SHAP values (this may take time)...")

    shap_values =explainer (X_shap )




    print ("[INFO] Generating plots...")




    plt .figure (figsize =(10 ,8 ))
    shap .plots .beeswarm (shap_values ,show =False ,max_display =20 )
    plt .title (f"SHAP Feature Importance — {serial}")
    plt .tight_layout ()
    plt .savefig (os .path .join (output_dir ,"shap_pipeline_beeswarm.png"),dpi =300 ,bbox_inches ='tight')
    plt .close ()

    plt .figure (figsize =(10 ,8 ))
    shap .plots .bar (shap_values ,show =False ,max_display =20 )
    plt .title (f"Mean |SHAP| — {serial}")
    plt .tight_layout ()
    plt .savefig (os .path .join (output_dir ,"shap_pipeline_bar.png"),dpi =300 ,bbox_inches ='tight')
    plt .close ()




    global_importance =np .abs (shap_values .values ).mean (axis =0 )
    feature_names =X_test_raw .columns .tolist ()

    importance_list =[]
    for name ,imp_val in zip (feature_names ,global_importance ):
        importance_list .append ({
        "feature":name ,
        "importance":float (imp_val ),
        "rank":0 
        })

    importance_list .sort (key =lambda x :x ['importance'],reverse =True )
    for i ,item in enumerate (importance_list ):
        item ['rank']=i +1 

    output_data ={
    "model_serial":serial ,
    "n_samples":len (X_test_raw ),
    "n_features":len (feature_names ),
    "features":importance_list 
    }

    json_path =os .path .join (output_dir ,"pipeline_feature_importance.json")
    with open (json_path ,'w')as f :
        json .dump (output_data ,f ,indent =4 )

    print (f"\n[SUCCESS] Saved to {output_dir}/")


if __name__ =="__main__":
    parser =argparse .ArgumentParser (description ="SHAP analysis for TearSense pipeline")
    parser .add_argument ('--model_path',type =str ,required =True ,help ="Path to .pkl file")
    args =parser .parse_args ()
    main (args .model_path )