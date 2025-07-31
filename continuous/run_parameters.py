import yaml
import copy
from train_continuous import main

def run_experiments():
    with open('config.yaml', 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f)

    model_name_to_test = "SegFormer"
    encoders_to_test = [
        "resnet50",
        "resnet18",
        "vgg11",
        "vgg13",
        "densenet121",
        "densenet169",
        "efficientnet-b1",
        "efficientnet-b2",
        "efficientnet-b3",
        "mobileone_s0",
        "mobileone_s1",
        "mobileone_s2",
        "mobileone_s3",
        "mobileone_s4",
    ]

    for encoder in encoders_to_test:
        print(f"\n CHUẨN BỊ THÍ NGHIỆM: Model={model_name_to_test}, Encoder={encoder} ".center(80, "-"))

        exp_config = copy.deepcopy(base_config)

        exp_config['model']['name'] = model_name_to_test
        exp_config['model']['encoder_name'] = encoder

        try:
            main(exp_config)
        except Exception as e:
            print(f"LỖI khi huấn luyện với encoder {encoder}: {e}")
            print("Tiếp tục với thí nghiệm tiếp theo...")
            continue

    print(" Done ".center(80, "="))


if __name__ == '__main__':
    run_experiments()
