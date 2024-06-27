from functools import partial
import os
import torch
import json
import torch.optim as optim
from src.model import *
from src.datasets import *
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, recall_score
import argparse
import importlib
from zeus.monitor import ZeusMonitor # requires nvidia GPU from december 2017 or newer
import optuna

torch.manual_seed(42)

def benchmark_trial(trial,encoding,epoch,path,input_size):
    
    # initialize best model
    monitor = ZeusMonitor(gpu_indices=[torch.cuda.current_device()])
    module_name = importlib.import_module("src.encodings")
    function_encoding = getattr(module_name,encoding)
    batch_size = 64
    device = torch.device('cuda')
    net = ANNDense(trial=trial,input_size=input_size).to(device)
    epochs = epoch
    lr = trial.suggest_float("lr", 1e-5, 0.5, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    optimizer = getattr(optim, optimizer_name)(net.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    train_set = SMILESDataset(mode="train", transform=function_encoding)
    test_set = SMILESDataset(mode="test", transform=function_encoding)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set,batch_size=batch_size, shuffle=True)
    
    # train model
    monitor.begin_window("total_train")
    for epoch in range(epochs):
            for i, (input, label) in enumerate(train_loader): # training loop
                optimizer.zero_grad()
                output = net(input.to(device)).to(device)
                loss = loss_fn(output.to(device),label.to(device))
                loss.backward()
                optimizer.step()
    mes = monitor.end_window("total_train")
    total_time_train = mes.time
    total_energy_train = mes.total_energy # in J
    
    # benchmark accuracy and AUC on test dataset
    with torch.no_grad():   
        test_labels = []
        test_pred = []
        test_conf = []
        for i, (input, label) in enumerate(test_loader):
            outputs = net.forward(input.to(device))
            pred_class = torch.argmax(outputs, dim=1)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            test_labels.append(label)
            test_pred.append(pred_class)
            test_conf.append(probs)
            
    conf_list = [x.tolist()[1] for xs in test_conf for x in xs]
    pred_list = [x.item() for xs in test_pred for x in xs]
    label_list = [x.item() for xs in test_labels for x in xs]
        
    bal_acc=balanced_accuracy_score(label_list,pred_list)
    
    auc = roc_auc_score(label_list, conf_list)
    
    recall = recall_score(label_list, pred_list) 
    
    specificity = recall_score(label_list, pred_list, pos_label=0)
    
    # benchmark time and energy consumed by model
    monitor = ZeusMonitor(gpu_indices=[torch.cuda.current_device()])
    monitor.begin_window("total_test")
    with torch.no_grad():   
        num_samples = 10
        num_predictions = 6250
        for i in range(num_samples):
            input,label = test_set[i]
            input = input.unsqueeze(0) # batch size = 1
            for i in range(num_predictions):
                output = net.forward(input.to(device))
    mes = monitor.end_window("total_test")
    total_time_test = mes.time
    total_energy_test = mes.total_energy
    
    return bal_acc, auc, label_list, conf_list, total_energy_train, total_time_train, total_energy_test, total_time_test, recall, specificity
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description ='Set up right HPO serch')
    parser.add_argument("--dataset", help='Which dataset to choose for training', choices={"OHE", "morgan1024","macc167"})
    parser.add_argument("--input_size", help="Number of input neurons",type=int)
    args = parser.parse_args()

    study_name = f"ANN_dense_{args.dataset}"
    path=f'./experiments/{study_name}/'
    os.makedirs(path, exist_ok=True)
    storage_name = "sqlite:///{}{}.db".format(path,study_name)
    study = optuna.create_study(direction="minimize", 
                                 study_name=study_name, 
                                 storage=storage_name,
                                 load_if_exists=True)

    print('Study statistics: ')
    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    best_trial = study.best_trial
    print("Value: ", best_trial.value)
    
    with open(f'{path}{study_name}.json', "r") as f:
            properties = json.load(f)

    list_train_energy = []
    list_train_time = []
    num_runs = 1
    for i in range(num_runs):
        balanced_accuracy, auc, label_list, conf_list, total_energy_train, total_time_train, total_energy_test, total_time_test, recall, specificity = benchmark_trial(best_trial,encoding=args.dataset,epoch=properties["best_epoch"],path=path,input_size=args.input_size)
        list_train_energy.append(total_energy_train)
        list_train_time.append(total_time_train)
    
    results = {"best_balanced_accuracy": balanced_accuracy, 
               "AUC": auc,
               "recall" : recall,
               "specificity" : specificity,
               "dataset": args.dataset,
               "hyperparameters": best_trial.params, 
               "label_list": label_list, 
               "conf_list": conf_list,
               "avg_train_energy": sum(list_train_energy) / len(list_train_energy),
               "avg_train_time": sum(list_train_time) / len(list_train_time),
               "avg_test_energy": total_energy_test,
               "avg_test_time": total_time_test,
               "predictions_per_second": 6250 / (total_time_test / 10),
               "avg_energy_per_inference": total_energy_test / 10 / 6250,
               "num_runs": num_runs
              }
    with open(f'{path}{study_name}_benchmark_efficiency.json', "w") as f:
        print("saving results")
        json.dump(results, f, indent=4)