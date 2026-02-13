import joblib 
import numpy as np 
import pandas as pd 
import json 
import os 
import argparse 
from sklearn .metrics import roc_auc_score ,brier_score_loss ,confusion_matrix ,f1_score ,roc_curve 
from sklearn .linear_model import LogisticRegression 
from sklearn .preprocessing import StandardScaler 






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






class FeatureEngineer :
    """Replicates core.py feature engineering exactly."""
    @staticmethod 
    def apply (df ):
        df =df .copy ()

        if 'tear_characteristics_Tear_AntPost'not in df .columns :
            return df 

        AP =df ['tear_characteristics_Tear_AntPost']
        ML =df ['tear_characteristics_Tear_MedLat']

        df ['Tear_AP_ML_Ratio']=np .where (ML >0 ,AP /ML ,np .nan )
        df ['Tear_Area_cm2']=(AP *ML )/100 
        df ['Log_Tear_Area']=np .log1p ((AP *ML ).clip (lower =0 ))

        if 'PREOP_Strength_IR'in df .columns and 'PREOP_Strength_ER'in df .columns :
            df ['ER_IR_Ratio']=np .where (
            df ['PREOP_Strength_IR']>0 ,
            df ['PREOP_Strength_ER']/df ['PREOP_Strength_IR'],
            np .nan 
            )

        if all (c in df .columns for c in ['Pre-Op_ROM_Pre-Op_FF','Pre-Op_ROM_Pre-Op_Abd','Pre-Op_ROM_Pre-Op_ER']):
            df ['ROM_Deficit_Score']=(
            (180 -df ['Pre-Op_ROM_Pre-Op_FF'].clip (upper =180 ))/180 +
            (180 -df ['Pre-Op_ROM_Pre-Op_Abd'].clip (upper =180 ))/180 +
            (90 -df ['Pre-Op_ROM_Pre-Op_ER'].clip (upper =90 ))/90 
            )/3 

        pain_cols =['PREOP_FOP_Activity_Pain','PREOP_FOP_Sleep_Pain','PREOP_FOP_Extreme_Pain']
        if all (c in df .columns for c in pain_cols ):
            df ['Pain_Frequency_Mean']=df [pain_cols ].sum (axis =1 )/3 

        return df 


class MetaFeatureBuilder :
    """Replicates defecator.meta_features exactly."""
    @staticmethod 
    def build (base_preds ,use_interactions =True ):
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






def find_youden_threshold (y_true ,y_probs ):
    """Find threshold that maximizes Youden's J statistic."""
    fpr ,tpr ,thresholds =roc_curve (y_true ,y_probs )
    j_scores =tpr -fpr 
    best_idx =np .argmax (j_scores )
    return thresholds [best_idx ]


def calculate_net_benefit (y_true ,y_prob ,threshold ):
    """Calculate net benefit at a given threshold."""
    tp =np .sum ((y_prob >=threshold )&(y_true ==1 ))
    fp =np .sum ((y_prob >=threshold )&(y_true ==0 ))
    n =len (y_true )
    if threshold <=0 or threshold >=1 :
        return 0.0 
    weight =threshold /(1 -threshold )
    return (tp /n )-(fp /n )*weight 


def get_calibration_slope_intercept (y_true ,y_prob ):
    """Calculate calibration slope and intercept."""
    epsilon =1e-10 
    y_prob =np .clip (y_prob ,epsilon ,1 -epsilon )
    log_odds =np .log (y_prob /(1 -y_prob ))

    clf =LogisticRegression (C =1e9 ,solver ='lbfgs',max_iter =1000 )
    clf .fit (log_odds .reshape (-1 ,1 ),y_true )

    return float (clf .coef_ [0 ][0 ]),float (clf .intercept_ [0 ])


def compute_all_metrics (y_true ,probs ,threshold ):
    """Compute all metrics for a given probability array."""
    auc =roc_auc_score (y_true ,probs )
    brier =brier_score_loss (y_true ,probs )

    preds =(probs >=threshold ).astype (int )
    tn ,fp ,fn ,tp =confusion_matrix (y_true ,preds ).ravel ()

    sens =tp /(tp +fn )if (tp +fn )>0 else 0.0 
    spec =tn /(tn +fp )if (tn +fp )>0 else 0.0 
    ppv =tp /(tp +fp )if (tp +fp )>0 else 0.0 
    npv =tn /(tn +fn )if (tn +fn )>0 else 0.0 
    f1 =f1_score (y_true ,preds ,zero_division =0 )

    net_benefit =calculate_net_benefit (y_true ,probs ,threshold )
    cal_slope ,cal_intercept =get_calibration_slope_intercept (y_true ,probs )

    return {
    "AUC":float (auc ),
    "Brier":float (brier ),
    "F1_Score":float (f1 ),
    "Sensitivity":float (sens ),
    "Specificity":float (spec ),
    "PPV":float (ppv ),
    "NPV":float (npv ),
    "Net_Benefit":float (net_benefit ),
    "Calibration_Slope":float (cal_slope ),
    "Calibration_Intercept":float (cal_intercept ),
    "Confusion_Matrix":{"TP":int (tp ),"TN":int (tn ),"FP":int (fp ),"FN":int (fn )}
    }






def prepare_data_for_lr_fit (X ,cat_cols ):
    """Prepare TRAINING data for LR. Returns encoded data + train medians."""
    X =X .copy ()

    for col in cat_cols :
        if col in X .columns :
            X [col ]=X [col ].fillna ('Missing').astype (str )

    if cat_cols :
        existing_cats =[c for c in cat_cols if c in X .columns ]
        if existing_cats :
            X =pd .get_dummies (X ,columns =existing_cats ,drop_first =True ,dummy_na =False )

    train_medians ={}
    for col in X .columns :
        if X [col ].dtype in ['float64','float32','int64','int32']:
            med =X [col ].median ()
            train_medians [col ]=med 
            X [col ]=X [col ].fillna (med )

    return X ,train_medians 


def prepare_data_for_lr_transform (X ,cat_cols ,train_medians ):
    """Prepare TEST data for LR using TRAIN medians (no leak)."""
    X =X .copy ()

    for col in cat_cols :
        if col in X .columns :
            X [col ]=X [col ].fillna ('Missing').astype (str )

    if cat_cols :
        existing_cats =[c for c in cat_cols if c in X .columns ]
        if existing_cats :
            X =pd .get_dummies (X ,columns =existing_cats ,drop_first =True ,dummy_na =False )

    for col in X .columns :
        if X [col ].dtype in ['float64','float32','int64','int32']:
            med =train_medians .get (col ,0.0 )
            X [col ]=X [col ].fillna (med )

    return X 


def align_columns (X_train ,X_test ):
    """Ensure train and test have same columns."""
    all_cols =list (set (X_train .columns )|set (X_test .columns ))

    for col in all_cols :
        if col not in X_train .columns :
            X_train [col ]=0 
        if col not in X_test .columns :
            X_test [col ]=0 

    sorted_cols =sorted (all_cols )
    return X_train [sorted_cols ],X_test [sorted_cols ]


def train_lr_baseline (X_train ,y_train ,X_test ,cat_cols ,feature_names ):
    """Train logistic regression baseline and return predictions."""

    X_train_subset =X_train [feature_names ].copy ()
    X_test_subset =X_test [feature_names ].copy ()

    X_train_enc ,train_medians =prepare_data_for_lr_fit (X_train_subset ,cat_cols )
    X_test_enc =prepare_data_for_lr_transform (X_test_subset ,cat_cols ,train_medians )

    X_train_enc ,X_test_enc =align_columns (X_train_enc ,X_test_enc )


    scaler =StandardScaler ()
    X_train_scaled =scaler .fit_transform (X_train_enc )
    X_test_scaled =scaler .transform (X_test_enc )


    lr_model =LogisticRegression (
    penalty ='l2',
    C =1.0 ,
    solver ='lbfgs',
    max_iter =1000 ,
    random_state =42 
    )
    lr_model .fit (X_train_scaled ,y_train )


    lr_probs =lr_model .predict_proba (X_test_scaled )[:,1 ]


    coefs =lr_model .coef_ [0 ]
    intercept =lr_model .intercept_ [0 ]

    coef_df =pd .DataFrame ({
    'Feature':list (X_train_enc .columns ),
    'Coefficient':coefs ,
    'Abs_Coefficient':np .abs (coefs ),
    'Odds_Ratio':np .exp (coefs )
    }).sort_values ('Abs_Coefficient',ascending =False )


    formula_parts =[f"{intercept:.4f}"]
    for _ ,row in coef_df .head (10 ).iterrows ():
        sign ="+"if row ['Coefficient']>=0 else "-"
        formula_parts .append (f"{sign} {abs(row['Coefficient']):.4f} × {row['Feature']}")

    formula_str ="z = "+" ".join (formula_parts [:3 ])+"\n    "+" ".join (formula_parts [3 :6 ])+"\n    "+" ".join (formula_parts [6 :])
    if len (coef_df )>10 :
        formula_str +=f"\n    + ... ({len(coef_df) - 10} more terms)"
    formula_str +="\n\nP(retear) = 1 / (1 + exp(-z))"

    return lr_probs ,lr_model ,coef_df ,formula_str 






def evaluate_layers (model_path ,output_dir =None ):
    print (f"Loading model: {model_path}...")
    pkg =joblib .load (model_path )

    serial_number =pkg .get ('serial_number','unknown')


    if output_dir is None :
        output_dir =os .path .join ("layer_evaluation",serial_number )
    os .makedirs (output_dir ,exist_ok =True )




    print ("Extracting Holdout Data...")
    X_test =pkg ['X_test_exact'].copy ()
    y_test =pkg ['y_test_exact']
    y_arr =y_test .values if hasattr (y_test ,'values')else np .array (y_test )

    feature_names =pkg ['feature_names']
    cat_cols =pkg ['cat_cols']
    n_folds =pkg ['n_folds']
    fold_models =pkg ['fold_models']


    X_train =pkg .get ('X_train_all')
    y_train =pkg .get ('y_train_all')
    has_train_data =X_train is not None and y_train is not None 

    if has_train_data :
        y_train_arr =y_train .values if hasattr (y_train ,'values')else np .array (y_train )
        print (f"[INFO] Training data available: {len(y_train_arr)} samples")
    else :
        print ("[WARNING] No training data in PKL. LR baseline will be skipped.")
        print ("[INFO] Update exporter.py to include X_train_all and y_train_all")


    if pkg .get ('feature_engineering',True ):
        X_test =FeatureEngineer .apply (X_test )
        if has_train_data :
            X_train =FeatureEngineer .apply (X_train )


    for col in feature_names :
        if col not in X_test .columns :
            X_test [col ]=np .nan 


    X_enc =X_test [feature_names ].copy ()
    for col in cat_cols :
        if col in X_enc .columns :
            X_enc [col ]=X_enc [col ].astype (str ).astype ('category').cat .codes 
    X_enc =X_enc .fillna (-999 )


    X_cat =X_test [feature_names ].copy ()
    for col in cat_cols :
        if col in X_cat .columns :
            X_cat [col ]=X_cat [col ].astype (str ).astype ('category')
            if "Missing"not in X_cat [col ].cat .categories :
                X_cat [col ]=X_cat [col ].cat .add_categories ("Missing")
            X_cat [col ]=X_cat [col ].fillna ("Missing")




    base_preds ={}
    for algo in ['cat','xgb','lgbm','rf']:
        preds_sum =np .zeros (len (X_test ))
        for model in fold_models [algo ]:
            if algo =='cat':
                preds_sum +=model .predict_proba (X_cat )[:,1 ]
            else :
                preds_sum +=model .predict_proba (X_enc )[:,1 ]
        base_preds [algo ]=preds_sum /n_folds 




    stack_config =pkg .get ('stacking_config',{})
    X_meta =MetaFeatureBuilder .build (
    base_preds ,
    use_interactions =stack_config .get ('use_interactions',True )
    )

    meta_model =pkg .get ('meta_model')
    is_weighted_avg =pkg .get ('is_weighted_avg',False )

    if is_weighted_avg or meta_model is None :
        weights =pkg .get ('weights_array')or pkg .get ('weights')
        if isinstance (weights ,dict ):
            meta_probs =(weights ['cat']*base_preds ['cat']+
            weights ['xgb']*base_preds ['xgb']+
            weights ['lgbm']*base_preds ['lgbm']+
            weights ['rf']*base_preds ['rf'])
        else :
            models =['cat','xgb','lgbm','rf']
            meta_probs =sum (weights [i ]*base_preds [m ]for i ,m in enumerate (models ))
    else :
        meta_probs =meta_model .predict_proba (X_meta )[:,1 ]

    calibrator =pkg .get ('calibrator')
    final_probs =calibrator .predict (meta_probs )if calibrator else meta_probs 




    youden_threshold =find_youden_threshold (y_arr ,final_probs )
    stored_threshold =pkg .get ('optimal_threshold',0.5 )
    threshold =youden_threshold 

    print (f"\n[INFO] Holdout N: {len(y_arr)}")
    print (f"[INFO] Youden's J Threshold: {youden_threshold:.4f}")
    print (f"[INFO] Stored Threshold: {stored_threshold:.4f}")
    print (f"[INFO] Using Threshold: {threshold:.4f} (Youden's J)")




    all_results ={
    "serial_number":serial_number ,
    "holdout_n":len (y_arr ),
    "threshold_used":float (threshold ),
    "youden_threshold":float (youden_threshold ),
    "stored_threshold":float (stored_threshold ),
    "layer_0_lr_baseline":{},
    "layer_1_base_models":{},
    "layer_2_meta_learner":{},
    "layer_3_full_pipeline":{}
    }




    lr_probs =None 
    lr_formula =None 
    lr_coef_df =None 

    if has_train_data :
        print ("\n"+"="*60 )
        print ("LAYER 0: LOGISTIC REGRESSION BASELINE")
        print ("="*60 )

        lr_probs ,lr_model ,lr_coef_df ,lr_formula =train_lr_baseline (
        X_train ,y_train_arr ,X_test ,cat_cols ,feature_names 
        )


        lr_threshold =find_youden_threshold (y_arr ,lr_probs )
        lr_metrics =compute_all_metrics (y_arr ,lr_probs ,lr_threshold )
        lr_metrics ['threshold_used']=float (lr_threshold )
        lr_metrics ['n_features']=len (lr_coef_df )
        all_results ["layer_0_lr_baseline"]=lr_metrics 

        print (f"\n>> Logistic Regression (L2, default class_weight)")
        print (f"   Threshold (LR's Youden): {lr_threshold:.4f}")
        print (f"   AUC              {lr_metrics['AUC']:.4f}")
        print (f"   Brier            {lr_metrics['Brier']:.4f}")
        print (f"   F1 Score         {lr_metrics['F1_Score']:.4f}")
        print (f"   Sensitivity      {lr_metrics['Sensitivity']:.4f}")
        print (f"   Specificity      {lr_metrics['Specificity']:.4f}")
        print (f"   PPV (Precision)  {lr_metrics['PPV']:.4f}")
        print (f"   NPV              {lr_metrics['NPV']:.4f}")
        print (f"   Net Benefit      {lr_metrics['Net_Benefit']:.4f}")
        print (f"   Calib Slope      {lr_metrics['Calibration_Slope']:.4f}")
        print (f"   Calib Intercept  {lr_metrics['Calibration_Intercept']:.4f}")


        coef_path =os .path .join (output_dir ,f"lr_coefficients_{serial_number}.csv")
        lr_coef_df .to_csv (coef_path ,index =False )
        print (f"\n[EXPORT] LR coefficients saved to: {coef_path}")




    print ("\n"+"="*60 )
    print ("LAYER 1: INDIVIDUAL BASE MODELS (Fold Ensemble)")
    print ("="*60 )

    algo_names ={
    'cat':'CatBoost',
    'xgb':'XGBoost',
    'lgbm':'LightGBM',
    'rf':'RandomForest'
    }

    for algo in ['cat','xgb','lgbm','rf']:
        metrics =compute_all_metrics (y_arr ,base_preds [algo ],threshold )
        all_results ["layer_1_base_models"][algo_names [algo ]]=metrics 

        print (f"\n>> {algo_names[algo]} (Ensemble of {n_folds} folds)")
        print (f"   AUC              {metrics['AUC']:.4f}")
        print (f"   Brier            {metrics['Brier']:.4f}")
        print (f"   F1 Score         {metrics['F1_Score']:.4f}")
        print (f"   Sensitivity      {metrics['Sensitivity']:.4f}")
        print (f"   Specificity      {metrics['Specificity']:.4f}")
        print (f"   PPV (Precision)  {metrics['PPV']:.4f}")
        print (f"   NPV              {metrics['NPV']:.4f}")
        print (f"   Net Benefit      {metrics['Net_Benefit']:.4f}")
        print (f"   Calib Slope      {metrics['Calibration_Slope']:.4f}")
        print (f"   Calib Intercept  {metrics['Calibration_Intercept']:.4f}")




    print ("\n"+"="*60 )
    print ("LAYER 2: META-LEARNER (Uncalibrated)")
    print ("="*60 )

    layer2_metrics =compute_all_metrics (y_arr ,meta_probs ,threshold )
    all_results ["layer_2_meta_learner"]=layer2_metrics 

    method_type ="Weighted Average"if (is_weighted_avg or meta_model is None )else type (meta_model ).__name__ 
    print (f"\n>> Meta-Learner Type: {method_type}")
    print (f"   AUC              {layer2_metrics['AUC']:.4f}")
    print (f"   Brier            {layer2_metrics['Brier']:.4f}")
    print (f"   F1 Score         {layer2_metrics['F1_Score']:.4f}")
    print (f"   Sensitivity      {layer2_metrics['Sensitivity']:.4f}")
    print (f"   Specificity      {layer2_metrics['Specificity']:.4f}")
    print (f"   PPV (Precision)  {layer2_metrics['PPV']:.4f}")
    print (f"   NPV              {layer2_metrics['NPV']:.4f}")
    print (f"   Net Benefit      {layer2_metrics['Net_Benefit']:.4f}")
    print (f"   Calib Slope      {layer2_metrics['Calibration_Slope']:.4f}")
    print (f"   Calib Intercept  {layer2_metrics['Calibration_Intercept']:.4f}")




    print ("\n"+"="*60 )
    print ("LAYER 3: COMPLETE PIPELINE (Calibrated)")
    print ("="*60 )

    if calibrator :
        print (f"\n>> Calibrator: {type(calibrator).__name__}")
    else :
        print ("\n>> Calibrator: None (Layer 3 = Layer 2)")

    layer3_metrics =compute_all_metrics (y_arr ,final_probs ,threshold )
    all_results ["layer_3_full_pipeline"]=layer3_metrics 

    print (f"   AUC              {layer3_metrics['AUC']:.4f}")
    print (f"   Brier            {layer3_metrics['Brier']:.4f}")
    print (f"   F1 Score         {layer3_metrics['F1_Score']:.4f}")
    print (f"   Sensitivity      {layer3_metrics['Sensitivity']:.4f}")
    print (f"   Specificity      {layer3_metrics['Specificity']:.4f}")
    print (f"   PPV (Precision)  {layer3_metrics['PPV']:.4f}")
    print (f"   NPV              {layer3_metrics['NPV']:.4f}")
    print (f"   Net Benefit      {layer3_metrics['Net_Benefit']:.4f}")
    print (f"   Calib Slope      {layer3_metrics['Calibration_Slope']:.4f}")
    print (f"   Calib Intercept  {layer3_metrics['Calibration_Intercept']:.4f}")




    print ("\n"+"="*88 )
    print ("SUMMARY COMPARISON (All Layers)")
    print ("="*88 )


    best_base =max (all_results ["layer_1_base_models"].items (),
    key =lambda x :x [1 ]['AUC'])

    print (f"\n{'Model':<20} {'AUC':>8} {'Brier':>8} {'F1':>8} {'Sens':>8} {'Spec':>8} {'PPV':>8} {'NB':>8}")
    print ("-"*88 )


    if has_train_data :
        m =all_results ["layer_0_lr_baseline"]
        print (f"{'LR Baseline':<20} {m['AUC']:>8.4f} {m['Brier']:>8.4f} {m['F1_Score']:>8.4f} "
        f"{m['Sensitivity']:>8.4f} {m['Specificity']:>8.4f} {m['PPV']:>8.4f} {m['Net_Benefit']:>8.4f}")

    print ("-"*88 )


    for name ,m in all_results ["layer_1_base_models"].items ():
        marker =" *"if name ==best_base [0 ]else ""
        print (f"{name:<20} {m['AUC']:>8.4f} {m['Brier']:>8.4f} {m['F1_Score']:>8.4f} "
        f"{m['Sensitivity']:>8.4f} {m['Specificity']:>8.4f} {m['PPV']:>8.4f} {m['Net_Benefit']:>8.4f}{marker}")

    print ("-"*88 )

    m =layer2_metrics 
    print (f"{'Meta-Learner':<20} {m['AUC']:>8.4f} {m['Brier']:>8.4f} {m['F1_Score']:>8.4f} "
    f"{m['Sensitivity']:>8.4f} {m['Specificity']:>8.4f} {m['PPV']:>8.4f} {m['Net_Benefit']:>8.4f}")

    m =layer3_metrics 
    print (f"{'Full Pipeline':<20} {m['AUC']:>8.4f} {m['Brier']:>8.4f} {m['F1_Score']:>8.4f} "
    f"{m['Sensitivity']:>8.4f} {m['Specificity']:>8.4f} {m['PPV']:>8.4f} {m['Net_Benefit']:>8.4f}")

    print ("-"*88 )
    print (f"* Best base model")


    print (f"\n>> Improvement Analysis:")
    best_base_auc =best_base [1 ]['AUC']
    final_auc =layer3_metrics ['AUC']

    if has_train_data :
        lr_auc =all_results ["layer_0_lr_baseline"]['AUC']
        print (f"   LR Baseline → Full Pipeline:     AUC Δ{final_auc - lr_auc:+.4f} ({(final_auc - lr_auc)/lr_auc*100:+.1f}%)")

    print (f"   Best Base ({best_base[0]}) → Full Pipeline: AUC Δ{final_auc - best_base_auc:+.4f}")
    print (f"   Meta-Learner → Full Pipeline:     AUC Δ{layer3_metrics['AUC'] - layer2_metrics['AUC']:+.4f}")


    if lr_formula :
        print (f"\n{'='*88}")
        print ("LOGISTIC REGRESSION FORMULA")
        print (f"{'='*88}")
        print (lr_formula )






    if lr_formula :
        all_results ["lr_formula"]=lr_formula 

    json_path =os .path .join (output_dir ,f"layer_evaluation_{serial_number}.json")
    with open (json_path ,'w')as f :
        json .dump (all_results ,f ,indent =4 ,cls =NumpyEncoder )

    print (f"\n[EXPORT] Results saved to: {json_path}")
    print ("[DONE]")

    return all_results ,lr_probs ,final_probs 






if __name__ =="__main__":







    serials_to_run =[
    '05022026_132823_67982',
    ]

    for serial in serials_to_run :
        model_path =f'outputs/{serial}/model/{serial}.pkl'
        evaluate_layers (model_path )