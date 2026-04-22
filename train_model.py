import argparse
import os
import json
import socket
from datetime import datetime
from signal import signal
from utils import create_prediction_tables, handle_signal, set_preds_dir
from data_loader import MyDataset
from fold1 import training_fold1
from fold2 import training_fold2



# CUDA_LAUNCH_BLOCKING make cuda report the error where it actually occurs
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

preds_dir = 'temp'
run_parameters={}

# TODO Completely separate 2 fold training

def _parse_features_used(raw_value):
    try:
        parsed = [int(x.strip()) for x in str(raw_value).split(",")]
    except ValueError as exc:
        raise ValueError("features_used must be a comma-separated list of integers.") from exc

    if len(parsed) != 5:
        raise ValueError("features_used must contain exactly 5 values (nFix,FFD,GPT,TRT,fixProp).")
    if any(value not in (0, 1) for value in parsed):
        raise ValueError("features_used values must be 0 or 1.")
    if sum(parsed) == 0:
        raise ValueError("At least one gaze feature must be enabled in features_used.")
    return parsed


def _parse_fp_dropout(raw_value):
    try:
        parsed = [float(x.strip()) for x in str(raw_value).split(",")]
    except ValueError as exc:
        raise ValueError("fp_dropout must be a comma-separated list of floats.") from exc
    if len(parsed) != 2:
        raise ValueError("fp_dropout must contain exactly 2 values.")
    return parsed


def _validate_positive_int(name, value):
    if value <= 0:
        raise ValueError(f"{name} must be > 0.")
    return value


if __name__ == '__main__':
    signal(2, handle_signal)

    models = ['distilbert', 'xlmroberta-base', 'xlmroberta-large']
    losses = ['mse', 'ccc', 'robust', 'mse+ccc', 'robust+ccc']

    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=models)
    parser.add_argument("loss", choices=losses)
    parser.add_argument(
        "--use-gaze-concat",
        action="store_true",
        help="Enable Seeing Eye to AI-style GazeConcat fusion using ET model 2 features.",
    )
    parser.add_argument(
        "--et2-checkpoint",
        default=None,
        help="Path to ET model 2 checkpoint (.pt/.safetensors). If omitted, ET2_CHECKPOINT_PATH is used.",
    )
    parser.add_argument(
        "--features-used",
        default="1,1,1,1,1",
        help="Gaze feature flags in nFix,FFD,GPT,TRT,fixProp order. Example: 0,1,0,1,0",
    )
    parser.add_argument(
        "--fp-dropout",
        default="0.0,0.3",
        help="Dropout for ET feature projector as p1,p2. Example: 0.1,0.3",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Global per-device batch size override for all models.",
    )
    parser.add_argument(
        "--batch-size-distil",
        type=int,
        default=None,
        help="Per-device batch size override for DistilBERT.",
    )
    parser.add_argument(
        "--batch-size-xlmrb",
        dest="batch_size_xlmrB",
        type=int,
        default=None,
        help="Per-device batch size override for XLM-RoBERTa-base.",
    )
    parser.add_argument(
        "--batch-size-xlmrl",
        dest="batch_size_xlmrL",
        type=int,
        default=None,
        help="Per-device batch size override for XLM-RoBERTa-large.",
    )
    parser.add_argument("--learning-rate", type=float, default=6e-6)
    parser.add_argument("--train-epochs", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument(
        "--optim",
        type=str,
        default="adamw_torch",
        help="TrainingArguments optimizer name (e.g. adamw_torch, adamw_hf, adafactor).",
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--maxlen", type=int, default=200)
    args = parser.parse_args()

    model = args.model
    loss = args.loss

    if(model == 'distilbert'):
        checkpoint = "distilbert-base-multilingual-cased"
    elif(model == 'xlmroberta-base'):
        checkpoint = "xlm-roberta-base"
    elif(model == 'xlmroberta-large'):
        checkpoint = "xlm-roberta-large"

    try:
        features_used = _parse_features_used(args.features_used)
        fp_dropout = _parse_fp_dropout(args.fp_dropout)
        _validate_positive_int("train_epochs", args.train_epochs)
        _validate_positive_int("gradient_accumulation_steps", args.gradient_accumulation_steps)
        _validate_positive_int("maxlen", args.maxlen)

        if args.batch_size is not None:
            _validate_positive_int("batch_size", args.batch_size)
        if args.batch_size_distil is not None:
            _validate_positive_int("batch_size_distil", args.batch_size_distil)
        if args.batch_size_xlmrB is not None:
            _validate_positive_int("batch_size_xlmrB", args.batch_size_xlmrB)
        if args.batch_size_xlmrL is not None:
            _validate_positive_int("batch_size_xlmrL", args.batch_size_xlmrL)
    except ValueError as exc:
        parser.error(str(exc))

    base_batch_size = args.batch_size if args.batch_size is not None else 16
    batch_size_distil = args.batch_size_distil if args.batch_size_distil is not None else base_batch_size
    batch_size_xlmrB = args.batch_size_xlmrB if args.batch_size_xlmrB is not None else base_batch_size
    batch_size_xlmrL = args.batch_size_xlmrL if args.batch_size_xlmrL is not None else base_batch_size

    if args.use_gaze_concat:
        max_text_len_for_concat = 255
        if args.maxlen > max_text_len_for_concat:
            parser.error(
                "When --use-gaze-concat is enabled, maxlen must be <= 255 "
                "to avoid positional limit overflow (concat length is about 2*maxlen+2)."
            )

    gaze_config = {
        "use_gaze_concat": args.use_gaze_concat,
        "et2_checkpoint_path": args.et2_checkpoint,
        "features_used": features_used,
        "fp_dropout": fp_dropout,
    }

    run_parameters['model'] = model
    run_parameters['loss_function'] = loss
    run_parameters['use_gaze_concat'] = gaze_config["use_gaze_concat"]
    run_parameters['et2_checkpoint_path'] = gaze_config["et2_checkpoint_path"]
    run_parameters['features_used'] = gaze_config["features_used"]
    run_parameters['fp_dropout'] = gaze_config["fp_dropout"]
    
    # Creates directories
    now = datetime.now()
    timestamp = now.strftime("%b-%d_%H-%M-%S")

    host_name = os.environ.get('COMPUTERNAME') or os.environ.get('HOST') or socket.gethostname()
    preds_dir = 'Preds/' + timestamp + "_" + host_name

    os.makedirs(preds_dir)
    set_preds_dir(preds_dir)
    run_parameters['path'] = preds_dir
    
    # Parameters
    params = {
        'batch_size_distil' : batch_size_distil,
        'batch_size_xlmrB' : batch_size_xlmrB,
        'batch_size_xlmrL' : batch_size_xlmrL,
        'lr' : args.learning_rate,
        'train_epochs' : args.train_epochs,
        'weight_decay' : args.weight_decay,
        'warmup_ratio': args.warmup_ratio,
        'optim': args.optim,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'seed': args.seed,
        'maxlen': args.maxlen,
    }
    
    # Write parameters to json file
    for k,v in params.items():
        run_parameters[k] = v
    with open(preds_dir + '/training_parameters.json', "w") as f:
        json.dump(run_parameters,f)
        
    # Dataset
    data_dir = "data" 
    filename_1 = os.path.join(data_dir, "full_dataset_fold1.csv")
    filename_2 = os.path.join(data_dir, "full_dataset_fold2.csv")

    
    split_1 = MyDataset(filename=filename_1, checkpoint=checkpoint, maxlen=args.maxlen)
    split_2 = MyDataset(filename=filename_2, checkpoint=checkpoint, maxlen=args.maxlen)
    dataset = [[split_1, split_2], [split_2, split_1]]

    # Train and Prediction
    training_fold1(model, loss, timestamp, params, dataset, preds_dir, checkpoint, gaze_config=gaze_config)
    print('\n\n\n------------ NOW ON FOLD 2 -------------- \n\n\n')
    training_fold2(model, loss, timestamp, params, dataset, preds_dir, checkpoint, gaze_config=gaze_config)
    
    create_prediction_tables(preds_dir)

    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
