import os
import torch
import json
import torch.optim as optim
from src.model import *
from src.datasets import *
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, recall_score
import argparse
import importlib
from zeus.monitor import ZeusMonitor
import optuna
import lava_dl.src.lava.lib.dl.slayer as slayer

torch.manual_seed(42)

def benchmark_trial(trial,encoding,epoch,path,input_size,time_steps):
    
    # initialize best model
    monitor = ZeusMonitor(gpu_indices=[torch.cuda.current_device()])
    time_steps = trial.suggest_int("time_steps", 1,25) # time_steps
    module_name = importlib.import_module("src.encodings")
    function_encoding = getattr(module_name,encoding)
    batch_size = 64
    device = torch.device('cuda')
    net = SLAYERDense(trial=trial,input_size=input_size).to(device)
    stats = slayer.utils.LearningStats()
    epochs = 100
    lr = trial.suggest_float("lr", 1e-5, 0.5, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    optimizer = getattr(optim, optimizer_name)(net.parameters(), lr=lr)
    true_rate = trial.suggest_float("true_rate",0.5,1.0)
    false_rate = trial.suggest_float("false_rate",0.0,0.5)
    error = slayer.loss.SpikeRate(true_rate=true_rate, false_rate=false_rate, reduction="mean").to(device)
    assistant = slayer.utils.Assistant(net, error, optimizer, stats, classifier=slayer.classifier.Rate())
    
    train_set = SMILESDataset(mode="train", transform=function_encoding,time_steps=time_steps)
    test_set = SMILESDataset(mode="test", transform=function_encoding,time_steps=time_steps)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set,batch_size=batch_size, shuffle=True)
    
    # train model
    monitor.begin_window("total_train")
    for epoch in range(epochs):
            for i, (sample, label) in enumerate(train_loader): # training loop
                output = assistant.train(sample, label)
            stats.update()
    
    mes = monitor.end_window("total_train")
    total_time_train = mes.time
    total_energy_train = mes.total_energy # in J
    # benchmark it on test dataset
    for i, (sample, label) in enumerate(test_loader): 
            output = assistant.test(sample, label)
            
    stats.plot(figsize=(15, 5),path=path,figures=(2,2))
    
    conf_list = [x.tolist()[1] for xs in stats.testing.test_conf for x in xs] # probability of the class with the greater label

    pred_list = [x.item() for xs in stats.testing.test_pred for x in xs]
    label_list = [x.item() for xs in stats.testing.test_labels for x in xs]
        
    bal_acc=balanced_accuracy_score(label_list,pred_list)
    
    auc = roc_auc_score(label_list, conf_list)
    
    recall = recall_score(label_list, pred_list) 
    
    specificity = recall_score(label_list, pred_list, pos_label=0)
    
    return bal_acc, auc, label_list, conf_list, total_energy_train, total_time_train, recall, specificity
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description ='Benchmark best model')
    parser.add_argument("--dataset", help='Which dataset to choose for training', choices={"OHE", "morgan1024x1","macc167x1","macc167x16", "macc167xT", "morgan1024xT"})
    parser.add_argument("--input_size", help="Number of input neurons",type=int)
    parser.add_argument("-t", help="Number of time steps",type=int)
    
    args = parser.parse_args()

    study_name = f"SLAYER_dense_{args.dataset}" # _{args.t}
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
        balanced_accuracy, auc, label_list, conf_list, total_energy_train, total_time_train, recall, specificity = benchmark_trial(best_trial,encoding=args.dataset,
                                                 epoch=properties["best_epoch"],
                                                 path=path,
                                                 input_size=args.input_size,
                                                 time_steps=args.t)
        list_train_energy.append(total_energy_train)
        list_train_time.append(total_time_train)
    
    results = {"best_balanced_accuracy": balanced_accuracy, 
               "AUC": auc,
               "recall" : recall,
               "specificity": specificity,
               "dataset": args.dataset,
               "hyperparameters": best_trial.params, 
               "label_list": label_list, 
               "conf_list": conf_list,
               "avg_train_energy": sum(list_train_energy) / len(list_train_energy),
               "avg_train_time": sum(list_train_time) / len(list_train_time),
               "num_runs": num_runs
              }
    with open(f'{path}{study_name}_benchmark_efficiency.json', "w") as f:
        print("saving results")
        json.dump(results, f, indent=4)
