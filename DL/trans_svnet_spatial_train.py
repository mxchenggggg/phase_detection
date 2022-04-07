from trainers.Trans_SVNet.sptial_feature_extractor_trainer import SptialFeatureExtractorTrainer

if __name__ == "__main__":
    trainer = SptialFeatureExtractorTrainer(
        default_config_file="/home/ubuntu/phase_detection/DL/modules/Trans_SVNet/config/SpatialExtractor.yml")
    trainer()
