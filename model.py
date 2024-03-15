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
                            output_attentions=False if parser_args.get('output_attentions') is None else parser_args['output_attentions'],
                            )
        
        self.learning_rate = parser_args["lr"]
        self.parser_args = parser_args
        
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.logger_should_sync_dist = torch.cuda.device_count() > 1
        self.save_hyperparameters(*kwargs.keys())
        self.my_logger = logging.getLogger("lightning")
        
        if self.parser_args["contrastive"] == True:
            self.exp_projection = torch.nn.Linear(768, 784)
            self.mcq_projection = torch.nn.Linear(768, 784)
            self.temp = torch.nn.Parameter(torch.tensor(1.0),requires_grad=True)

    def forward(self, input_ids, attention_mask, labels=None, output_hidden_states=False):
        output = self.model(input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=output_hidden_states)
        return output

    def training_step(self, batch, batch_idx):
        if self.parser_args["contrastive"] == False:
            input_ids, attention_mask, labels = batch
        else:
            input_ids, attention_mask, exp_ids, exp_attention_mask, exp_exist, labels = batch
        
        # CE loss
        outputs = self(input_ids, attention_mask, labels=labels, output_hidden_states=self.parser_args["contrastive"])
        loss = outputs.loss
        self.log("train/ce_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)  # Log training loss
        
        
        # train accu
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        acc = torch.tensor(torch.sum(preds == labels).item() / len(preds))
        self.log("train_acc", acc.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # contrastive loss
        if self.parser_args["contrastive"] == True:
            mc_cls_embedding = outputs.hidden_states[-1][:,0,:].squeeze(1) # shape of batch*784
        
            exp_outputs = self(exp_ids, exp_attention_mask, output_hidden_states=True)
            exp_cls_embedding = exp_outputs.hidden_states[-1][:,0,:].squeeze(1) # shape of batch*784
            exp_cls_embedding = self.exp_projection(exp_cls_embedding)
            exp_cls_embedding = torch.nn.functional.normalize(exp_cls_embedding, dim=-1)
            mc_cls_embedding = self.mcq_projection(mc_cls_embedding)
            mc_cls_embedding = torch.nn.functional.normalize(mc_cls_embedding, dim=-1)
            contrastive_logits = exp_cls_embedding @ mc_cls_embedding.T
            contrastive_logits = contrastive_logits * torch.exp(self.temp)
        
            contrastive_ground_truth = torch.diag(exp_exist).to(self.device)
            contrastive_ground_truth = torch.where(exp_exist==0, torch.ones(len(exp_exist)).to(self.device)/len(exp_exist), contrastive_ground_truth).float()
            contrastive_ground_truth = contrastive_ground_truth.T

            contrastive_loss = torch.nn.functional.cross_entropy(contrastive_logits, contrastive_ground_truth)
            self.log("train_contrastive_loss", contrastive_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            loss += contrastive_loss * self.parser_args['contrastive_coeff']
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
