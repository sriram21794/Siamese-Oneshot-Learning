import argparse

from config import Config, config
global config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Copyright ML Flows')
    parser.add_argument('run_type')
    parser.add_argument('--config', '-c', type=str, required=False)
    parser.add_argument('--background_directory', '-b', type=str, required=True)
    parser.add_argument('--evaluation_directory', '-e', type=str, required=True)
    parser.add_argument('--model_path', '-m', type=str, required=True)
    

    args = parser.parse_args()
    config.load_json(args.config)

    from model import SiameseModel
    siamese_model = SiameseModel()

    siamese_model.train(args.background_directory, args.evaluation_directory, args.model_path)
