import argparse
import logging
import os
from datetime import datetime

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader import load_data, retrieval
from model import RGP

pkl_path = '../dataset/smpd/dataset_q.pkl'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_model(model, gpu_ids):
    """Move the model to the primary GPU and enable multi-GPU training."""
    global device

    if torch.cuda.is_available() and gpu_ids:
        device = torch.device(f'cuda:{gpu_ids[0]}')
        model = model.to(device)
        if len(gpu_ids) > 1:
            model = nn.DataParallel(model, device_ids=gpu_ids)
    else:
        model = model.to(device)
    return model


def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model

def log_args(args):
    log_message = "Arguments:\n" + "\n".join(
        f"  {name}: {value}" for name, value in sorted(vars(args).items())
    )
    print(log_message)
    logging.info(log_message)

def train_validate_retrieval(model, train_data_loader,valid_data_loader, loss_fn, optimizer, device):
    train_loss=0
    model.train()
    for data in tqdm(train_data_loader):
        # Unpack data
        base = data['base_data']
        top = data['top_data']
        
        # Base features (aligned with train_retrieval naming)
        text_features = base['text_features'].to(device)      # [batch_size, text_dim]
        img_features = base['img_features'].to(device)        # [batch_size, img_dim]
        bool_data = base['bool_data'].to(device)              # [batch_size]
        user_id = base['user_id'].to(device)                  # [batch_size]
        pp_label = data['pp_label'].to(device)                # [batch_size, seq_len]

        # Positive sample features (aligned with train_retrieval naming)
        top_text_features = top['top_text_features'].to(device)    # [batch_size, top_k, text_dim]
        top_img_features = top['top_img_features'].to(device)      # [batch_size, top_k, img_dim]
        top_score = top['top_score'].to(device)                    # [batch_size, top_k]
        top_label = top['top_label'].to(device)  # [batch_size, top_k]
        top_bool_data = top['top_bool_data'].to(device)               # [batch_size]
        top_user_id = top['top_user_id'].to(device)                   # [batch_size]

        optimizer.zero_grad()
        
        # Model forward pass (train mode)
        out = model(
            text_features=text_features,
            img_features=img_features,
            bool_data=bool_data,
            user_id=user_id,
            top_text_features=top_text_features,
            top_img_features=top_img_features,
            top_score=top_score,
            top_label=top_label,
            top_bool_data=top_bool_data,
            top_user_id=top_user_id,
        )
        
        # Loss calculation
        r_label = pp_label.unsqueeze(1)
        loss = loss_fn(out, r_label)

        train_loss += loss.item()
        
        # Backward pass
        loss.backward()

        optimizer.step()
    train_loss /= len(train_data_loader) 
    logging.info(f"Train Loss: {train_loss:.4f}\n")
    print(f"Train Loss: {train_loss:.4f}\n")


    all_outputs = []
    all_labels = []
    
    model.eval()
    for data in tqdm(valid_data_loader):
        base = data['base_data']
        top = data['top_data']
        
        # Base features (aligned with train_retrieval naming)
        text_features = base['text_features'].to(device)      # [batch_size, text_dim]
        img_features = base['img_features'].to(device)        # [batch_size, img_dim]
        bool_data = base['bool_data'].to(device)              # [batch_size]
        user_id = base['user_id'].to(device)                  # [batch_size]
        pp_label = data['pp_label'].to(device)                # [batch_size, seq_len]

        # Positive sample features (aligned with train_retrieval naming)
        top_text_features = top['top_text_features'].to(device)    # [batch_size, top_k, text_dim]
        top_img_features = top['top_img_features'].to(device)      # [batch_size, top_k, img_dim]
        top_score = top['top_score'].to(device)                    # [batch_size, top_k]
        top_label = top['top_label'].to(device)  # [batch_size, top_k]
        top_bool_data = top['top_bool_data'].to(device)               # [batch_size]
        top_user_id = top['top_user_id'].to(device)                   # [batch_size]


        with torch.no_grad():
            out = model(
                text_features=text_features,
                img_features=img_features,
                bool_data=bool_data,
                user_id=user_id,
                top_text_features=top_text_features,
                top_img_features=top_img_features,
                top_score=top_score,
                top_label=top_label,
                top_bool_data=top_bool_data,
                top_user_id=top_user_id,
            )
            

            out_label = out.cpu().detach().numpy()
            r_label = pp_label.unsqueeze(1).cpu().detach().numpy()
            
            all_outputs.append(out_label)
            all_labels.append(r_label)


    all_outputs = np.concatenate(all_outputs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    

    MAE = mean_absolute_error(all_labels, all_outputs)
    SRC, _ = spearmanr(all_outputs, all_labels)
    MSE = mean_squared_error(y_pred=all_outputs, y_true=all_labels)
    nMSE = np.mean(np.square(all_outputs - all_labels)) / (all_labels.std() ** 2)

    logging.info(f"[ Valid Result ]:  \n MSE {MSE}"
                f"\nSRC {SRC}\n MAE {MAE}\n  nMSE {nMSE}\n")
    print(f"[ Valid Result ]:  \n MSE {MSE}"
                f"\nSRC {SRC}\n MAE {MAE}\n  nMSE {nMSE}\n")

    return MSE

def delete_model(folder_path):

    model_files = [f for f in os.listdir(folder_path) if f.endswith('.pth')]
    
    if not model_files:
        logging.warning(f"No .pth files found in {folder_path}.")
        return


    model_files.sort(key=lambda x: float(x.split('-')[1].replace('.pth', '')))
    

    for model_file in model_files[:-1]:
        file_path = os.path.join(folder_path, model_file)
        try:
            os.remove(file_path)
            logging.info(f"Deleted old model: {file_path}")
        except Exception as e:
            logging.error(f"Failed to delete {file_path}: {e}")

def test(model, test_data_loader):

    all_outputs = []
    all_labels = []
    
    model.eval()
    for data in tqdm(test_data_loader):
        base = data['base_data']
        top = data['top_data']
        
        # Base features (aligned with train_retrieval naming)
        text_features = base['text_features'].to(device)      # [batch_size, text_dim]
        img_features = base['img_features'].to(device)        # [batch_size, img_dim]
        bool_data = base['bool_data'].to(device)              # [batch_size]
        user_id = base['user_id'].to(device)                  # [batch_size]
        pp_label = data['pp_label'].to(device)                # [batch_size, seq_len]
        # Positive sample features (aligned with train_retrieval naming)
        top_text_features = top['top_text_features'].to(device)    # [batch_size, top_k, text_dim]
        top_img_features = top['top_img_features'].to(device)      # [batch_size, top_k, img_dim]
        top_score = top['top_score'].to(device)                    # [batch_size, top_k]
        top_label = top['top_label'].to(device)  # [batch_size, top_k]
        top_bool_data = top['top_bool_data'].to(device)               # [batch_size]
        top_user_id = top['top_user_id'].to(device)                   # [batch_size]

        with torch.no_grad():
            out = model(
                text_features=text_features,
                img_features=img_features,
                bool_data=bool_data,
                user_id=user_id,
                top_text_features=top_text_features,
                top_img_features=top_img_features,
                top_score=top_score,
                top_label=top_label,
                top_bool_data=top_bool_data,
                top_user_id=top_user_id,
            )
            

            out_label = out.cpu().detach().numpy()
            r_label = pp_label.unsqueeze(1).cpu().detach().numpy()
            
            all_outputs.append(out_label)
            all_labels.append(r_label)


    all_outputs = np.concatenate(all_outputs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    

    MAE = mean_absolute_error(all_labels, all_outputs)
    SRC, _ = spearmanr(all_outputs, all_labels)
    MSE = mean_squared_error(y_pred=all_outputs, y_true=all_labels)
    nMSE = np.mean(np.square(all_outputs - all_labels)) / (all_labels.std() ** 2)
    
    logging.info(f"[ TEST Result ]:  \n MSE {MSE}"
                f"\nSRC {SRC}\n MAE {MAE}\n nMSE {nMSE}\n")
    print(f"[ TEST Result ]:  \n MSE {MSE}"
                f"\nSRC {SRC}\n MAE {MAE}\n nMSE {nMSE}\n")

def train_epoch(args, model, train_loader, val_loader, test_loader):

    model_name = model.__class__.__name__
    
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S') + model_name
    log_dir = os.path.join(args.log + '/log', current_time)
    os.makedirs(log_dir, exist_ok=True)
    
    
    logging.basicConfig(
        filename=os.path.join(log_dir, f'_train.log'),
        level=logging.INFO
    )
    log_args(args)
    print(f'Created directory: {log_dir}')
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.6, patience=1, min_lr=args.min_lr)


    loss_fn_name = loss_fn.__class__.__name__
    logging.info(model_name)
    logging.info(loss_fn_name)

    min_mae = float('inf')
    early_stop_patience = args.early_stop
    count = 0

    
    for epoch in range(args.epochs):
        avg_val_loss = train_validate_retrieval(model,train_loader, val_loader, loss_fn, optimizer, device)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} Learning Rate: {current_lr}")
        logging.info(f"Epoch {epoch+1} Learning Rate: {current_lr}")

        # Update scheduler and check early stopping
        scheduler.step(avg_val_loss)

        update_lr = optimizer.param_groups[0]['lr']
        if update_lr < current_lr:
            logging.info(f"Learning rate updated from {current_lr} to {update_lr}")
            print(f"Learning rate updated from {current_lr} to {update_lr}")
        torch.cuda.empty_cache()
        
        if avg_val_loss < min_mae:
            min_mae = avg_val_loss
            count = 0
            logging.info(f"all_mae:{avg_val_loss},count:{count}")
            logging.info(f"best mae:{min_mae}")
            delete_model(log_dir)
            torch.save(unwrap_model(model).state_dict(), os.path.join(log_dir, f'{epoch + 1}-{avg_val_loss:.4f}.pth'))
            print(f'Saved model with MAE: {avg_val_loss:.4f}')
            print(f"all_mae:{avg_val_loss},count:{count}")

            
        else:
            count += 1
            logging.info(f"all_mae:{avg_val_loss},count:{count}")
            if count >= early_stop_patience:
                print("Early stopping triggered.")
                break

    model_files = [f for f in os.listdir(log_dir) if f.endswith('.pth')]
    model_files.sort(key=lambda x: float(x.split('-')[1].replace('.pth', '')))
    best_model_files=os.path.join(log_dir,model_files[0])
    print(f'\nvaild_file:{best_model_files}\n')
    best_model_dict = torch.load(best_model_files, map_location=device)
    unwrap_model(model).load_state_dict(best_model_dict)
    test(model=model, test_data_loader=test_loader)


parser = argparse.ArgumentParser(description='Model trainer')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--epochs', type=int, default=160)

parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--min_lr', default=0.00001, type=float) # 
parser.add_argument('--weight_decay', default=0.0001, type=float)
parser.add_argument('--data_files', type=str, default=pkl_path)

parser.add_argument('--log', type=str, default='out')

parser.add_argument('--way', type=str, default='train', choices=['train', 'test'])

parser.add_argument('--train_ratio', default=0.8, type=float) #8:1:1

parser.add_argument('--early_stop',  default=5, type=int)
parser.add_argument('--top_k',  default=40, type=int)

parser.add_argument('--checkpoint', type=str, default=None)
parser.add_argument(
    '--gpu_ids',
    type=str,
    default="0,1,2",
    help='Comma-separated CUDA device IDs, for example 0,1; default: all visible GPUs',
)
if __name__ == '__main__':
    args = parser.parse_args()

    if args.gpu_ids is None:
        gpu_ids = list(range(torch.cuda.device_count()))
    else:
        gpu_ids = [int(gpu_id.strip()) for gpu_id in args.gpu_ids.split(',')]
    if not gpu_ids and torch.cuda.is_available():
        parser.error('--gpu_ids must contain at least one GPU ID')

    if args.way == 'train':
        model = RGP(args.batch_size, args.top_k)
        train_loader, val_loader, test_loader = load_data(args)

        model = setup_model(model, gpu_ids)
        train_epoch(args, model, train_loader, val_loader, test_loader)
    elif args.way == 'test':
        if not args.checkpoint:
            parser.error('--checkpoint is required when --way test')

        full_dataset = retrieval(
            args.data_files,
            top_k=args.top_k,
        )
        loader_args = {
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'pin_memory': True,
            'persistent_workers': args.num_workers > 0,
            'drop_last': False
        }
        test_loader = DataLoader(full_dataset, shuffle=False, **loader_args)
        model = RGP(args.batch_size, args.top_k)
        model = setup_model(model, gpu_ids)
        model_dict = torch.load(args.checkpoint, map_location=device)
        unwrap_model(model).load_state_dict(model_dict)
        test(model=model, test_data_loader=test_loader)
        torch.cuda.empty_cache()

    else:
        print(r"please choose 'train' or 'test'")
