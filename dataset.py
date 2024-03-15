import logging
import torch, os, pytorch_lightning as pl, glob
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset



class MedMCQA_Datamodule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32, parser_args=None):
        super().__init__()
        self.logger = logging.getLogger("lightning")

        self.batch_size = batch_size
        # self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.parser_args = parser_args


    def setup(self, stage=None):
        folder_path = f"/root/pubmedQA_291/dataset_pickles/medmcqa/{self.parser_args['pretrained_model']}/{self.parser_args['model_style']}_style/"   # either classification or clip
        if stage == 'fit' or stage == "validate" or stage is None:
            
            self.logger.info('Setting up train and val dataset')
            if self.parser_args["contrastive"] == False:
                self.logger.info('Loading train dataset without explanations')
                self.train_dataset = torch.load(folder_path + "train_dataset.pt")
            else:
                self.logger.info('Loading train dataset with explanations')
                self.train_dataset = torch.load(folder_path + "train_dataset_with_explanations.pt")

            if self.parser_args['has_additional_prompt'] == False:
                self.val_dataset = torch.load(folder_path + "val_dataset.pt")
            else:
                model_name = "general-bert" if self.parser_args['pretrained_model'] == "bert-base-uncased" else "pubmed-bert"
                self.logger.info(f'Loading val dataset with hints added tokenized by {model_name}')
                self.val_dataset = torch.load(f'/root/pubmedQA_291/dataset_pickles/medmcqa/hints-added-tokenized-by-{model_name}/classification_style/val.pt')
            
        if stage == "test"   : 
            self.logger.info('Setting up test dataset')
            self.test_dataset = torch.load(folder_path + "test_dataset.pt")
            
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, shuffle=True, 
                          num_workers=self.parser_args["num_workers"], pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size,
                            num_workers=self.parser_args["num_workers"], pin_memory=True, persistent_workers=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.batch_size, 
                          num_workers=self.parser_args["num_workers"], pin_memory=True, persistent_workers=True)