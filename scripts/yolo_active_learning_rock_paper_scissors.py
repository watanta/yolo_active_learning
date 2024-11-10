from ultralytics import YOLO
import yaml
import shutil
import numpy as np
from pathlib import Path
import random
import os
from tqdm.auto import tqdm
import time

random.seed(42)  # 任意の固定値（ここでは42）

class ActiveLearning:
    def __init__(self, data_root='data/coco', initial_samples=100):
        # 絶対パスに変換
        self.data_root = Path(data_root).absolute()
        self.initial_samples = initial_samples
        
        # 作業ディレクトリも絶対パスで設定
        self.active_data_dir = self.data_root / 'active_learning_data'
        self.pool_dir = self.active_data_dir / 'pool'
        self.train_dir = self.active_data_dir / 'train'
        self.val_dir = self.active_data_dir / 'val'
        
        # 既存のディレクトリを削除
        if self.active_data_dir.exists():
            print(f"Removing existing directory: {self.active_data_dir}")
            try:
                shutil.rmtree(self.active_data_dir)
            except Exception as e:
                print(f"Error removing directory: {e}")
                raise
        
        print("Creating directories...")
        # 各ディレクトリの画像とラベルのサブディレクトリを作成
        for d in [self.pool_dir, self.train_dir, self.val_dir]:
            (d / 'images').mkdir(parents=True, exist_ok=True)
            (d / 'labels').mkdir(parents=True, exist_ok=True)

    def prepare_initial_dataset(self):
        """初期データセットの準備"""
        print("Preparing initial dataset...")
        
        # 訓練データの処理
        train_images_dir = self.data_root / 'train' / 'images'
        all_images = list(train_images_dir.glob('*.jpg'))
        
        total_images = len(all_images)
        if total_images == 0:
            raise ValueError(f"No images found in {train_images_dir}")
        
        # 検証用データを2000枚固定で選択
        val_size = 600
        print(f"\nSelecting {val_size} images for validation...")
        val_images = set(random.sample(all_images, val_size))
        remaining_images = [img for img in all_images if img not in val_images]
        
        # 検証用データをコピー
        print("Copying validation data...")
        for img_path in tqdm(val_images, desc="Copying validation data", position=1):
            label_path = self.data_root / 'train' / 'labels' / (img_path.stem + '.txt')
            shutil.copy2(img_path, self.val_dir / 'images' / img_path.name)
            if label_path.exists():
                shutil.copy2(label_path, self.val_dir / 'labels' / label_path.name)
        
        # プール用のデータを5000枚選択
        pool_size = 5000
        print(f"\nSelecting {pool_size} images for pool...")
        pool_images = set(random.sample(remaining_images, pool_size))
        
        # 初期訓練データの選択
        remaining_for_initial = [img for img in pool_images]  # pool_imagesからコピーを作成
        actual_initial_samples = min(self.initial_samples, len(remaining_for_initial))
        print(f"Using {actual_initial_samples} images for initial training")
        
        # ランダムに初期サンプルを選択
        initial_images = set(random.sample(remaining_for_initial, actual_initial_samples))
        
        # データの振り分けとコピー
        print("Copying images to respective directories...")
        with tqdm(pool_images, desc="Processing images", position=1) as pbar:
            for img_path in pbar:
                label_path = self.data_root / 'train' / 'labels' / (img_path.stem + '.txt')
                
                if img_path in initial_images:
                    dest_dir = self.train_dir
                    pbar.set_postfix({'Destination': 'train'})
                else:
                    dest_dir = self.pool_dir
                    pbar.set_postfix({'Destination': 'pool'})
                    
                shutil.copy2(img_path, dest_dir / 'images' / img_path.name)
                if label_path.exists():
                    shutil.copy2(label_path, dest_dir / 'labels' / label_path.name)
        
        # 最終的なデータセットサイズを表示
        train_size = len(list((self.train_dir / 'images').glob('*.jpg')))
        pool_size = len(list((self.pool_dir / 'images').glob('*.jpg')))
        val_size = len(list((self.val_dir / 'images').glob('*.jpg')))
        
        print("\nDataset preparation completed:")
        print(f"Training set: {train_size} images")
        print(f"Pool set: {pool_size} images")
        print(f"Validation set: {val_size} images")

    def create_yaml(self):
        """データセット設定ファイルの作成"""
        print("Creating dataset configuration file...")
        
        # 元のYAMLファイルからクラス名を取得
        original_yaml_path = self.data_root / 'data.yaml'
        if not original_yaml_path.exists():
            raise FileNotFoundError(f"Original data.yaml not found at {original_yaml_path}")
            
        with open(original_yaml_path, 'r') as f:
            original_yaml = yaml.safe_load(f)
            
        # 新しいYAML設定を作成（絶対パスを使用）
        yaml_content = {
            'path': str(self.active_data_dir),
            'train': str(self.train_dir / 'images'),
            'val': str(self.val_dir / 'images'),
            'test': '',
            'names': original_yaml['names'],
            'nc': len(original_yaml['names'])
        }
        
        # パスの存在確認
        for key in ['train', 'val']:
            if key in yaml_content and yaml_content[key]:
                path = Path(yaml_content[key])
                if not path.exists():
                    print(f"Warning: {key} path does not exist: {path}")
                else:
                    files = list(path.glob('*.*'))
                    print(f"Found {len(files)} files in {key} directory")
        
        # 新しいYAMLファイルを保存
        yaml_path = self.active_data_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f)
            
        print(f"Created dataset configuration at {yaml_path}")
        print("YAML content:")
        for k, v in yaml_content.items():
            print(f"{k}: {v}")
            
        return yaml_path

    def select_next_sample(self, model, num_samples=100):
        """確信度が最も低い上位num_samples個のサンプルを選択"""
        pool_images = list((self.pool_dir / 'images').glob('*.jpg'))
        if not pool_images:
            return []

        # 全画像の確信度を評価
        image_confidences = []
        with tqdm(pool_images, desc="Evaluating images", position=2) as pbar:
            for img_path in pbar:
                results = model.predict(
                    str(img_path), 
                    conf=0.1, 
                    verbose=False,
                    stream=True
                )
                result = next(results)
                
                if len(result.boxes) > 0:
                    confidences = [box.conf.item() for box in result.boxes]
                    min_conf = min(confidences)  # 平均値から最小値に変更
                else:
                    min_conf = 0.0  # 検出なしの場合は最高確信度とする
                
                image_confidences.append((img_path, min_conf))
                pbar.set_postfix({
                    'Images Processed': len(image_confidences),
                    'Current Min Conf': f'{min_conf:.4f}'
                })

        # 確信度でソートし、最も低い上位num_samples個を選択
        selected_images = sorted(image_confidences, key=lambda x: x[1])[:num_samples]
        
        print(f"\nSelected {len(selected_images)} images")
        print(f"Confidence range: {selected_images[0][1]:.4f} - {selected_images[-1][1]:.4f}")
        
        return [img_path for img_path, _ in selected_images]

    def add_to_training(self, selected_images, model, round_num):
        """選択したサンプルを訓練セットに追加し、予測結果付きの画像を保存"""
        if not selected_images:
            return 0

        # ラウンドごとの保存ディレクトリを作成
        round_dir = self.active_data_dir / f'round_{round_num}_selected'
        round_dir.mkdir(parents=True, exist_ok=True)

        samples_added = 0
        with tqdm(selected_images, desc="Adding to training set", position=2) as pbar:
            for image_path in pbar:
                label_path = self.pool_dir / 'labels' / (image_path.stem + '.txt')
                
                # ラベルファイルが存在する場合のみ処理
                if label_path.exists():
                    try:
                        # 予測を実行して結果を保存
                        results = model.predict(
                            str(image_path),
                            conf=0.1,
                            verbose=False,
                            save=True,  # 予測結果を画像として保存
                            project=str(round_dir),
                            name='',
                            exist_ok=True
                        )
                        
                        # 予測の確信度情報を保存
                        result = results[0]
                        conf_info = []
                        if len(result.boxes) > 0:
                            for box in result.boxes:
                                cls = result.names[int(box.cls)]
                                conf = box.conf.item()
                                conf_info.append(f"{cls}: {conf:.4f}")
                        
                        info_path = round_dir / f"{image_path.stem}_info.txt"
                        with open(info_path, 'w') as f:
                            f.write(f"Image: {image_path.name}\n")
                            f.write(f"Confidence scores:\n")
                            f.write("\n".join(conf_info) if conf_info else "No detections")
                        
                        # 画像とラベルを訓練セットに移動
                        shutil.move(image_path, self.train_dir / 'images' / image_path.name)
                        shutil.move(label_path, self.train_dir / 'labels' / label_path.name)
                        samples_added += 1
                        
                        pbar.set_postfix({'Added': samples_added})
                    except Exception as e:
                        print(f"Error processing {image_path.name}: {e}")
                        continue
                else:
                    print(f"Skipping {image_path.name} - no label file found")

        print(f"\nSuccessfully added {samples_added} samples to training set")
        return samples_added
    
def print_metrics(metrics, prefix=""):
    """メトリクスを表示する補助関数"""
    # validationのメトリクスを表示
    val_metrics = {
        'Validation mAP50': metrics.get('metrics/mAP50(B)', 'N/A'),
        'Validation mAP50-95': metrics.get('metrics/mAP50-95(B)', 'N/A'),
        'Validation Precision': metrics.get('metrics/precision(B)', 'N/A'),
        'Validation Recall': metrics.get('metrics/recall(B)', 'N/A')
    }
    
    print(f"\n{prefix} Validation Metrics:")
    for name, value in val_metrics.items():
        if value != 'N/A':
            print(f"{name}: {value:.4f}")
        else:
            print(f"{name}: {value}")

def print_final_results(active_learner, rounds_completed, metrics):
    """最終結果を表示する補助関数"""
    print("\n=== Final Results ===")
    print(f"Rounds completed: {rounds_completed}")
    print(f"Final training set size: {len(list((active_learner.train_dir / 'images').glob('*.jpg')))} images")
    print(f"Final pool size: {len(list((active_learner.pool_dir / 'images').glob('*.jpg')))} images")
    print("\nFinal model performance:")
    for key in ['metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'train/box_loss', 'train/cls_loss']:
        if key in metrics:
            print(f"{key}: {metrics[key]:.4f}")

def train_with_active_learning():
    # active_learning_dataディレクトリの存在確認
    active_data_dir = Path('active_learning_data')
    if active_data_dir.exists():
        print(f"Found existing active learning data directory.")
        print(f"This will be removed to ensure a clean start.")
    
    # 全体の進捗バーを設定
    print("\n=== Active Learning Process ===")
    max_rounds = 25
    samples_per_round = 100  # 1ラウンドあたり100枚追加
    
    with tqdm(total=max_rounds, desc="Overall Progress", position=0) as pbar_overall:
        print("\n1. Initializing Active Learning...")
        active_learner = ActiveLearning(initial_samples=100, data_root='data/rock_paper_scissors')
        active_learner.prepare_initial_dataset()
        yaml_path = active_learner.create_yaml()
        
        # 学習設定
        config = {
            'epochs': 5,
            'batch': 2,
            'imgsz': 640,
            'device': 0,
            'workers': 8,
            'patience': 20,
            'save': True,
            'project': 'runs',
            'name': 'rock_paper_scissors',
        }
        
        print("\n2. Initializing YOLO model...")
        model = YOLO('yolov8n.pt')
        
        # 初期状態の表示
        initial_train_size = len(list((active_learner.train_dir / 'images').glob('*.jpg')))
        initial_pool_size = len(list((active_learner.pool_dir / 'images').glob('*.jpg')))
        print(f"\nInitial Status:")
        print(f"Training set size: {initial_train_size} images")
        print(f"Pool size: {initial_pool_size} images")
    
        # 現在のデータセットで学習
        results = model.train(data=str(yaml_path), **config)
        
        # メトリクスの表示
        metrics = results.results_dict
        print_metrics(metrics, prefix="Initial Training")
        
        # メインループ
        for round_num in range(max_rounds):
            print(f"\n=== Round {round_num + 1}/{max_rounds} ===")

            current_train_size = len(list((active_learner.train_dir / 'images').glob('*.jpg')))
            current_pool_size = len(list((active_learner.pool_dir / 'images').glob('*.jpg')))
            
            pbar_overall.set_postfix({
                'Train Size': current_train_size,
                'Pool Size': current_pool_size
            })
            

            
            # サンプル選択と追加
            print("\nSelecting new samples...")
            selected_samples = active_learner.select_next_sample(model, num_samples=samples_per_round)
            
            if selected_samples:
            # modelとround_numを追加
                samples_added = active_learner.add_to_training(
                    selected_images=selected_samples,
                    model=model,  # modelを追加
                    round_num=round_num + 1
                )
                print(f"\nAdded {samples_added} new samples to training set")
                
                # 新しいサンプルを追加した後で再学習
                print(f"\n=== Round {round_num + 1}/{max_rounds} (Training with New Samples) ===")
                results = model.train(data=str(yaml_path), **config)
                
                # 再学習後のメトリクスを表示
                metrics = results.results_dict
                print_metrics(metrics, prefix="After Adding Samples")
            else:
                print("\nNo more samples in pool. Ending active learning.")
                print_final_results(active_learner, round_num + 1, metrics)
                return
            
            # ラウンドの進捗を更新
            pbar_overall.update(1)
            
            # ラウンド終了時の状態を表示
            print(f"\nRound {round_num + 1} Summary:")
            print(f"Added {samples_added} new samples")
            print(f"Current training set size: {len(list((active_learner.train_dir / 'images').glob('*.jpg')))} images")
            print(f"Remaining pool size: {len(list((active_learner.pool_dir / 'images').glob('*.jpg')))} images")

        # 全ラウンド完了
        print_final_results(active_learner, max_rounds, metrics)

if __name__ == '__main__':
    train_with_active_learning()