from trainers.mastoid.mastoid_trainer_base import MastoidTrainerBase

if __name__ == "__main__":
    trainer = MastoidTrainerBase()
    trainer()
    # trainer.datamodule.prepare_data()
    # trainer.datamodule.setup()
    # dataloader = trainer.datamodule.train_dataloader()
    # for idx , batch in enumerate(dataloader):
    #     x, targets = batch
    #     print(x.shape)
    #     x = x.to('cuda:1')
    #     print(x.device)
