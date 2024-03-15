import logging, os, sys, torch
import random
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.callbacks as cb

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger  # Import TensorBoardLogger
from argparse import ArgumentParser

from dataset import MedMCQA_Datamodule
from model import MedMCQAModel






def seed_everything(seed):
    """
    Set the random seed for reproducibility.
    """
    pl.seed_everything(seed,  workers=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)
    random.seed(seed)
    # torch.use_deterministic_algorithms(True)
    
def init_logger(out_path, path1, path2):
    logger = logging.getLogger("lightning")
    logger.setLevel(logging.DEBUG)
    file_path = os.path.join(out_path, path1, path2, "logs.txt")
    os.makedirs(os.path.join(out_path, path1, path2), exist_ok=True)
    with open(file_path, 'w') as fp: # touch
        pass
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(file_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler(sys.stdout))
            
    return logger

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    seed_everything(seed=2024)    
    
    parser = ArgumentParser(add_help=True)
    parser.add_argument("--model_style", type=str, default="classification", help="classification or clip")
    parser.add_argument("--pretrained_model", type=str, default="bert-base-uncased", help="pretrained model name: bert-base-uncased, pubmed-bert. ")
    parser.add_argument("--has_additional_prompt", action="store_true", help="Whether to use additional prompt for the model")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--foldername", type=str, default=f"lightning_logs")
    parser.add_argument("--expname", type=str, default=f"experiment")
    
    # training specifics
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--hidden_dropout", type=float, default=0.5)
    parser.add_argument("--attention_dropout", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=3)
    
    # parser.add_argument("--datasrc", type=str, default=f"/workspace/SMILES_dataset")
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=16)
    
    # for early stopping/model saving
    parser.add_argument("--metric", type=str, default="val/mean_acc")
    parser.add_argument("--metricmode", type=str, default="max")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the checkpoint file to resume training")
    parser.add_argument("--saved_model_path", type=str, default=None, help="Path to the checkpoint file to resume training, not strict")
    
    # constrastive
    parser.add_argument("--contrastive", action="store_true", help="Whether to use contrastive learning")
    # contrastive_coeff defualt None
    parser.add_argument("--contrastive_coeff", type=float, default=None, help="Contrastive loss coefficient")
    
    args = vars(parser.parse_known_args()[0])
    
    # txt logger 
    out_path = f"/root/pubmedQA_291/exps"
    path1 = args['foldername'] 
    path2 = args['pretrained_model'] + "_"+ args['model_style']+"/" + args['expname']
    my_logger = init_logger(out_path, path1, path2)
    my_logger.info(f'[Main] Output Path: {out_path}/{path1}/{path2}')
    my_logger.info(f'[Main] gpu: {torch.cuda.get_device_name()}')
    # my_logger.info(f'[Main] Hyperparameters: {hparam_string}')
    
    
    # Instantiate the model and datamodule
    data_module = MedMCQA_Datamodule(batch_size=args["bs"], parser_args=args)
    if args['saved_model_path']:
        model = MedMCQAModel.load_from_checkpoint(args['saved_model_path'],strict=False, parser_args=args)
    else:
        model = MedMCQAModel(parser_args=args, **args)
        
    # Trainer, callbacks
    metric, metricmode, patience = args["metric"], args["metricmode"], args["patience"]

    tbl = TensorBoardLogger(save_dir=out_path, name=path1, version=path2)
    
    checkpoint_callback = cb.ModelCheckpoint(monitor=metric, mode=metricmode, save_last=True, save_top_k = 1)
    early_stopping = EarlyStopping(monitor=metric, mode=metricmode, patience=patience)
    lr_monitor = cb.LearningRateMonitor(logging_interval="step")
    trainer = pl.Trainer(
                         max_epochs=args["epochs"],
                         accelerator="gpu",
                         logger=tbl, 
                         callbacks=[checkpoint_callback, early_stopping, lr_monitor],
                        #  limit_train_batches = 10
                        )
    
    
    my_logger.info("[Main] Begin Training!")
    trainer.fit(model, data_module,ckpt_path=args["checkpoint_path"])
    my_logger.info("[Main] Training Complete! Start to do a final validation run")
    val_result = trainer.validate(model, data_module,ckpt_path=checkpoint_callback.best_model_path)
    my_logger.info(f"[Main] Final Validation Result: {val_result}")
    # no test because test dataset is not publicly available 