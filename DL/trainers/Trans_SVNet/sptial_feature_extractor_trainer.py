from trainers.mastoid.mastoid_trainer_base import MastoidTrainerBase
from pytorch_lightning import Trainer


class SptialFeatureExtractorTrainer(MastoidTrainerBase):
    def _predict(self) -> None:
        trainer = Trainer(
            gpus=self.hprms.gpus, logger=self.loggers,
            resume_from_checkpoint=self.hprms.resume_from_checkpoint)
        predictions = trainer.predict(
            self.module, datamodule=self.datamodule)
        
        for batch_pred in predictions:
            for i in range(batch_pred.size(dim = 0)):
                pred_tensor = batch_pred[i, :]
                print(pred_tensor.shape)
        
