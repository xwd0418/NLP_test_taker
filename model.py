import logging
import torch , numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from pytorch_lightning import LightningModule



class MedMCQAModel(LightningModule):
    def __init__(self, num_labels: int = 4, parser_args=None, **kwargs):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained(
                            parser_args['pretrained_model'], 
                            num_labels=num_labels,
                            attention_probs_dropout_prob=parser_args['attention_dropout'],
                            hidden_dropout_prob=parser_args['hidden_dropout'],
                            )
        
        self.learning_rate = parser_args["lr"]
        self.parser_args = parser_args
        
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.logger_should_sync_dist = torch.cuda.device_count() > 1
        self.save_hyperparameters(*kwargs.keys())
        self.my_logger = logging.getLogger("lightning")

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        return output

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids, attention_mask, labels=labels)
        loss = outputs.loss
        self.log("train/ce_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)  # Log training loss
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids, attention_mask, labels=labels)
        val_loss = outputs.loss
        self.log("val/ce_loss", val_loss, prog_bar=True, logger=True)  # Log validation loss
        
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        acc = torch.tensor(torch.sum(preds == labels).item() / len(preds))
        # self.log("val/acc", acc, prog_bar=True, logger=True)
        self.validation_step_outputs.append(acc)
        return val_loss

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids, attention_mask, labels=labels)
        test_loss = outputs.loss
        self.log("test/ce_loss", test_loss, prog_bar=True, logger=True)  # Log test loss
        
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        acc = torch.tensor(torch.sum(preds == labels).item() / len(preds))
        # self.log("test/acc", acc, prog_bar=True, logger=True)
        self.test_step_outputs.append(acc)
        return test_loss
        
    def on_validation_epoch_end(self):
        # return
        mean_acc = np.mean(self.validation_step_outputs)
        self.log("val/mean_acc", mean_acc, on_epoch=True, prog_bar=True)
        self.validation_step_outputs.clear()
        self.my_logger.info(f"Validation Accuracy: {mean_acc:.4f}")
        
    def on_test_epoch_end(self):
        # return
        mean_acc = np.mean(self.test_step_outputs)
        self.log("test/mean_acc", mean_acc, on_epoch=True, prog_bar=True)
        self.test_step_outputs.clear()
        
        
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), 
                                 lr=self.learning_rate,
                                 weight_decay = self.parser_args['weight_decay'],
                                )
    
    
    def log(self, name, value, *args, **kwargs):
        # Set 'sync_dist' to True by default
        if kwargs.get('sync_dist') is None:
            kwargs['sync_dist'] = kwargs.get(
                'sync_dist', self.logger_should_sync_dist)
        super().log(name, value, *args, **kwargs)
