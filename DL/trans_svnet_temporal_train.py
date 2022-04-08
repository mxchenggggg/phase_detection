from trainers.mastoid.mastoid_trainer_base import MastoidTrainerBase

if __name__ == "__main__":
    trainer = MastoidTrainerBase(
        default_config_file="/home/ubuntu/phase_detection/DL/modules/Trans_SVNet/config/temporal_extractor.yml")
    trainer()
