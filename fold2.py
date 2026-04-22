from transformers import RobertaForSequenceClassification, TrainingArguments, DataCollatorWithPadding, Trainer 
from models import (
    DistilBertForSequenceClassificationSig,
    GazeConcatForSequenceRegression,
    XLMRobertaForSequenceClassificationSig,
)
from data_loader import MyDataset
from custom_trainer import CustomTrainerMSE, CustomTrainerCCC, CustomTrainerRobust, CustomTrainerMSE_CCC, CustomTrainerRobustCCC
from metrics import compute_metrics
import torch
import pandas as pd
import numpy as np
import json
from utils import create_prediction_tables


    
    
def training_fold2(model, loss, timestamp, params, dataset, preds_dir, checkpoint, gaze_config=None):
    
    output_dir2 = "Output Directory/" + timestamp + "/fold2"
    model_dir = "model/" + timestamp + "/fold2"
    log_dir = "runs/" + timestamp + "/fold2"
    gaze_config = gaze_config or {}
    use_gaze_concat = bool(gaze_config.get("use_gaze_concat", False))

    train_data = dataset[1][0]
    val_data = dataset[1][1]
    
    # Chooses the model and batch size
    if(model == 'distilbert'):
        batch_size = params['batch_size_distil']
    elif(model == 'xlmroberta-base'):
        batch_size = params['batch_size_xlmrB']
    elif(model == 'xlmroberta-large'):
        batch_size = params['batch_size_xlmrL']

    if use_gaze_concat:
        model = GazeConcatForSequenceRegression(
            checkpoint=checkpoint,
            tokenizer=train_data.tokenizer,
            et2_checkpoint_path=gaze_config.get("et2_checkpoint_path"),
            features_used=gaze_config.get("features_used", [1, 1, 1, 1, 1]),
            fp_dropout=tuple(gaze_config.get("fp_dropout", [0.0, 0.3])),
        )
    elif(model == 'distilbert'):
        model = DistilBertForSequenceClassificationSig.from_pretrained(checkpoint, num_labels=2)
    elif(model == 'xlmroberta-base'):
        model = XLMRobertaForSequenceClassificationSig.from_pretrained(checkpoint, num_labels=2)
    elif(model == 'xlmroberta-large'):
        model = XLMRobertaForSequenceClassificationSig.from_pretrained(checkpoint, num_labels=2)
    
    training_args = TrainingArguments(
        output_dir=output_dir2,
        logging_dir='logs/logs2',
        logging_steps=200,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size, 
        num_train_epochs=params['train_epochs'],
        learning_rate=params['lr'],
        weight_decay=params['weight_decay'],
        optim=params.get('optim', 'adamw_torch'),
        gradient_accumulation_steps=params.get('gradient_accumulation_steps', 1),
        seed=params.get('seed', 42),

        group_by_length=True,
        evaluation_strategy="epoch", 
        save_strategy="epoch",
        load_best_model_at_end=True,
        warmup_ratio=params['warmup_ratio'],
        # report_to="wandb"
        ) 
    
    
    
    print("Starting fold 2")

    data_collator = DataCollatorWithPadding(train_data.tokenizer)
    
    if(loss == 'mse'):
        trainer2 = CustomTrainerMSE(
        model,
        training_args,
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=val_data,    
        tokenizer=train_data.tokenizer,
        compute_metrics=compute_metrics,
        )
    elif(loss == 'ccc'):
        trainer2 = CustomTrainerCCC(
        model,
        training_args,
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=val_data,    
        tokenizer=train_data.tokenizer,
        compute_metrics=compute_metrics,
        )
    elif(loss == 'robust'):
        trainer2 = CustomTrainerRobust(
        model,
        training_args,
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=val_data,    
        tokenizer=train_data.tokenizer,
        compute_metrics=compute_metrics,
        )
    elif(loss == 'mse+ccc'): 
        trainer2 = CustomTrainerMSE_CCC(
        model,
        training_args,
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=val_data,    
        tokenizer=train_data.tokenizer,
        compute_metrics=compute_metrics,
        )
    elif(loss == 'robust+ccc'): 
        trainer2 = CustomTrainerRobustCCC(
        model,
        training_args,
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=val_data,    
        tokenizer=train_data.tokenizer,
        compute_metrics=compute_metrics,
        )
    
    
    trainer2.train()
        
    # eval
    preds1 = trainer2.predict(val_data)
  
    run_metrics = preds1.metrics
    preds_df1 = pd.DataFrame(preds1.predictions)
    
    preds_df1.to_csv(preds_dir + "/predictions_fold1.csv")      # Write file with predictions on fold2 data
    with open(preds_dir + '/fold1_metrics.csv', 'w') as fb:     # Write run metrics
        for key in run_metrics.keys():
            fb.write("%s,%s\n"%(key,run_metrics[key]))
    fb.close()
    
    trainer2.save_model(model_dir)  
    
