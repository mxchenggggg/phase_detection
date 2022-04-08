from trainers.Trans_SVNet.temporal_feature_extractor_trainer import TemporalFeatureExtractorTrainer

if __name__ == "__main__":
    trainer = TemporalFeatureExtractorTrainer   (
        default_config_file="/home/ubuntu/phase_detection/DL/modules/Trans_SVNet/config/temporal_extractor.yml")
    trainer()
