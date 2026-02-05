import argparse 
import joblib 
import json 
import os 
import sys 
import numpy as np 
import pandas as pd 
import matplotlib .pyplot as plt 
from sklearn .linear_model import LogisticRegression 
from sklearn .preprocessing import StandardScaler 
from sklearn .metrics import (
roc_auc_score ,brier_score_loss ,confusion_matrix ,
f1_score ,roc_curve ,RocCurveDisplay 
)
from sklearn .calibration import CalibrationDisplay 
from sklearn .impute import SimpleImputer 





class NumpyEncoder (json .JSONEncoder ):
    def default (self ,obj ):
        if isinstance (obj ,(np .int_ ,np .intc ,np .intp ,np .int8 ,
        np .int16 ,np .int32 ,np .int64 ,np .uint8 ,
        np .uint16 ,np .uint32 ,np .uint64 )):
            return int (obj )
        elif isinstance (obj ,(np .float_ ,np .float16 ,np .float32 ,np .float64 )):
            return float (obj )
        elif isinstance (obj ,(np .bool_ ,)):
            return bool (obj )
        elif isinstance (obj ,np .ndarray ):
            return obj .tolist ()
        return super ().default (obj )

def bootstrap_ci (y_true ,y_pred ,metric_func ,n_boot =1000 ,alpha =0.95 ):
    """Calculates 95% CI for a metric using bootstrapping."""
    boot_stats =[]
    rng =np .random .RandomState (42 )
    for _ in range (n_boot ):
        indices =rng .randint (0 ,len (y_pred ),len (y_pred ))
        if len (np .unique (y_true [indices ]))<2 :
            continue 
        score =metric_func (y_true [indices ],y_pred [indices ])
        boot_stats .append (score )

    lower =np .percentile (boot_stats ,(1 -alpha )/2 *100 )
    upper =np .percentile (boot_stats ,(1 +alpha )/2 *100 )
    return lower ,upper 

def calculate_net_benefit (y_true ,y_prob ,thresholds ):
    """Calculates Net Benefit for DCA."""
    net_benefits =[]
    n =len (y_true )
    for t in thresholds :
        tp =np .sum ((y_prob >=t )&(y_true ==1 ))
        fp =np .sum ((y_prob >=t )&(y_true ==0 ))
        weight =t /(1 -t )
        nb =(tp /n )-(fp /n )*weight 
        net_benefits .append (nb )
    return np .array (net_benefits )





def prepare_data_for_lr (X ,cat_cols ):
    """
    Standard preprocessing for Logistic Regression:
    - Imputes NaNs
    - One-Hot Encodes Categoricals
    - Standard Scales Numerics
    """
    X =X .copy ()


    X_encoded =X .copy ()
    for col in cat_cols :
        if col in X_encoded .columns :

            X_encoded [col ]=X_encoded [col ].astype (str )

            dummies =pd .get_dummies (X_encoded [col ],prefix =col ,drop_first =True )
            X_encoded =pd .concat ([X_encoded ,dummies ],axis =1 )
            X_encoded .drop (columns =[col ],inplace =True )


    imputer =SimpleImputer (strategy ='median')
    X_imputed =pd .DataFrame (imputer .fit_transform (X_encoded ),columns =X_encoded .columns )


    scaler =StandardScaler ()
    X_scaled =pd .DataFrame (scaler .fit_transform (X_imputed ),columns =X_encoded .columns )

    return X_scaled ,X_encoded .columns 





def plot_comparisons (y_test ,lr_probs ,ts_probs ,output_dir ):
    """Generates ROC, Calibration, and DCA comparison plots."""


    plt .figure (figsize =(8 ,6 ))
    RocCurveDisplay .from_predictions (y_test ,lr_probs ,name ="Logistic Regression",ax =plt .gca ())
    if ts_probs is not None :
        RocCurveDisplay .from_predictions (y_test ,ts_probs ,name ="TearSense AI",ax =plt .gca (),plot_chance_level =True )
    plt .title ("ROC Comparison: Baseline vs AI")
    plt .savefig (os .path .join (output_dir ,"comparison_roc.png"),dpi =300 )
    plt .close ()


    plt .figure (figsize =(8 ,6 ))
    CalibrationDisplay .from_predictions (y_test ,lr_probs ,n_bins =10 ,name ="Logistic Regression",ax =plt .gca ())
    if ts_probs is not None :
        CalibrationDisplay .from_predictions (y_test ,ts_probs ,n_bins =10 ,name ="TearSense AI",ax =plt .gca ())
    plt .title ("Calibration Comparison")
    plt .savefig (os .path .join (output_dir ,"comparison_calibration.png"),dpi =300 )
    plt .close ()


    thresholds =np .linspace (0.01 ,0.99 ,100 )
    lr_nb =calculate_net_benefit (y_test ,lr_probs ,thresholds )

    plt .figure (figsize =(8 ,6 ))
    plt .plot (thresholds ,lr_nb ,label ="Logistic Regression")

    if ts_probs is not None :
        ts_nb =calculate_net_benefit (y_test ,ts_probs ,thresholds )
        plt .plot (thresholds ,ts_nb ,label ="TearSense AI",linestyle ="--")

    plt .plot (thresholds ,np .zeros_like (thresholds ),'k:',label ="Treat None")


    prevalence =np .mean (y_test )
    treat_all =prevalence -(1 -prevalence )*thresholds /(1 -thresholds )
    plt .plot (thresholds ,treat_all ,'k--',alpha =0.5 ,label ="Treat All")

    plt .ylim (-0.1 ,0.2 )
    plt .xlabel ("Threshold Probability")
    plt .ylabel ("Net Benefit")
    plt .title ("Decision Curve Analysis (DCA)")
    plt .legend ()
    plt .grid (True ,alpha =0.3 )
    plt .savefig (os .path .join (output_dir ,"comparison_dca.png"),dpi =300 )
    plt .close ()





def main (model_path ):
    print (f"\n[START] Processing model: {model_path}")


    try :
        pkg =joblib .load (model_path )
        X_train_raw =pkg ['X_train_all']
        y_train =pkg ['y_train_all']
        X_test_raw =pkg ['X_test_exact']
        y_test =pkg ['y_test_exact'].values if hasattr (pkg ['y_test_exact'],'values')else pkg ['y_test_exact']
        cat_cols =pkg ['cat_cols']


        serial =pkg .get ('serial_number','unknown_serial')
        output_dir =f"./external_assessor/{serial}/lr"
        os .makedirs (output_dir ,exist_ok =True )

    except Exception as e :
        print (f"[ERROR] Could not load data from PKL: {e}")
        return 


    print ("[1/5] Preprocessing data for Logistic Regression...")
    X_train_lr ,feature_names =prepare_data_for_lr (X_train_raw ,cat_cols )
    X_test_lr ,_ =prepare_data_for_lr (X_test_raw ,cat_cols )


    missing_cols =set (X_train_lr .columns )-set (X_test_lr .columns )
    for c in missing_cols :X_test_lr [c ]=0 
    X_test_lr =X_test_lr [X_train_lr .columns ]


    print ("[2/5] Training Logistic Regression Baseline...")
    lr =LogisticRegression (class_weight ='balanced',max_iter =2000 ,random_state =42 )
    lr .fit (X_train_lr ,y_train )
    lr_probs =lr .predict_proba (X_test_lr )[:,1 ]


    print ("[3/5] Running TearSense Inference for Comparison...")
    try :



        if hasattr (pkg ,'predict_proba'):
            ts_probs =pkg .predict_proba (X_test_raw )[:,1 ]
        elif 'meta_model'in pkg :



            print ("    (Warning: Could not run direct inference on PKL, skipping TS comparison curves)")
            ts_probs =None 
        else :
            ts_probs =None 



    except :
        ts_probs =None 


    print ("[4/5] Generating Comparison Plots...")
    plot_comparisons (y_test ,lr_probs ,ts_probs ,output_dir )


    print ("[5/5] Calculating Metrics & Extracting Formula...")


    auc =roc_auc_score (y_test ,lr_probs )
    brier =brier_score_loss (y_test ,lr_probs )


    fpr ,tpr ,thresholds =roc_curve (y_test ,lr_probs )
    j_scores =tpr -fpr 
    best_thresh =thresholds [np .argmax (j_scores )]

    preds =(lr_probs >=best_thresh ).astype (int )
    f1 =f1_score (y_test ,preds )
    tn ,fp ,fn ,tp =confusion_matrix (y_test ,preds ).ravel ()
    sens =tp /(tp +fn )
    spec =tn /(tn +fp )


    coefs =lr .coef_ [0 ]
    intercept =lr .intercept_ [0 ]
    coef_df =pd .DataFrame ({
    'Feature':feature_names ,
    'Coefficient':coefs ,
    'Abs_Coefficient':np .abs (coefs )
    }).sort_values ('Abs_Coefficient',ascending =False )


    formula_parts =[f"{intercept:.4f} (Intercept)"]
    for _ ,row in coef_df .iterrows ():
        if abs (row ['Coefficient'])>0.0001 :
            sign ="+"if row ['Coefficient']>=0 else "-"
            formula_parts .append (f"{sign} {abs(row['Coefficient']):.4f} * {row['Feature']}")

    formula_str ="Logit(P) = "+" ".join (formula_parts )






    print ("\n"+"="*80 )
    print (f"BASELINE LOGISTIC REGRESSION RESULTS (Serial: {serial})")
    print ("="*80 )
    print (f"{'METRIC':<20} | {'VALUE':<10} | {'95% CI (Bootstrap)':<20}")
    print ("-"*60 )

    metrics_to_report ={
    "AUC":(auc ,lambda y ,p :roc_auc_score (y ,p ),lr_probs ),
    "Brier Score":(brier ,lambda y ,p :brier_score_loss (y ,p ),lr_probs ),
    "F1 Score":(f1 ,lambda y ,p :f1_score (y ,(p >=best_thresh ).astype (int )),lr_probs ),
    "Sensitivity":(sens ,lambda y ,p :confusion_matrix (y ,(p >=best_thresh ).astype (int )).ravel ()[3 ]/np .sum (y ==1 ),lr_probs ),
    "Specificity":(spec ,lambda y ,p :confusion_matrix (y ,(p >=best_thresh ).astype (int )).ravel ()[0 ]/np .sum (y ==0 ),lr_probs )
    }

    for name ,(val ,func ,probs )in metrics_to_report .items ():
        low ,high =bootstrap_ci (y_test ,probs ,func )
        print (f"{name:<20} | {val:.4f}     | ({low:.4f}, {high:.4f})")

    print ("="*80 )
    print ("\nMATHEMATICAL FORMULA")
    print ("-"*20 )
    print (formula_str )
    print ("-"*20 )
    print (f"(Threshold used: {best_thresh:.4f})")
    print ("\n[Comparison]")
    if ts_probs is not None :
        ts_auc =roc_auc_score (y_test ,ts_probs )
        print (f"TearSense AUC: {ts_auc:.4f}")
        print (f"Baseline  AUC: {auc:.4f}")
        print (f"Delta        : {ts_auc - auc:+.4f}")
    else :
        print ("TearSense probabilities not available for direct comparison.")

    print ("="*80 )


    coef_df .to_csv (os .path .join (output_dir ,"lr_coefficients.csv"),index =False )

    summary ={
    "metrics":{k :v [0 ]for k ,v in metrics_to_report .items ()},
    "formula":formula_str ,
    "threshold":best_thresh 
    }
    with open (os .path .join (output_dir ,"lr_summary.json"),'w')as f :
        json .dump (summary ,f ,indent =4 ,cls =NumpyEncoder )

    print (f"\n[DONE] All outputs saved to: {output_dir}/")

if __name__ =="__main__":

    path_model ='outputs/05022026_132823_67982/model/05022026_132823_67982.pkl'

    main (path_model )
