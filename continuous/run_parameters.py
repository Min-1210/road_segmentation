
import yaml
import copy
from train_continuous import main

def run_experiments():
    with open('config.yaml', 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f)

    model_name_to_test = "DeepLabV3Plus"
    encoders_to_test = [
        "resnet18",
        "resnet50",
        "efficientnet-b1",
    ]

    for encoder in encoders_to_test:
        exp_config = copy.deepcopy(base_config)

        exp_config['model']['name'] = model_name_to_test
        exp_config['model']['encoder_name'] = encoder

        try:
            main(exp_config)
        except Exception as e:
            print(f"LỖI khi huấn luyện với encoder {encoder}: {e}")
            print("Tiếp tục với thí nghiệm tiếp theo...")
            continue

if __name__ == '__main__':
    run_experiments()
