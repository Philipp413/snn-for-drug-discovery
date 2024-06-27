from functools import partial
import os
import torch
import json
import torch.optim as optim
from src.model import *
from src.datasets import *
from sklearn.model_selection import StratifiedKFold
from optuna.visualization.matplotlib import plot_param_importances
from matplotlib import pyplot as plt
import argparse
import importlib
from src.torch_callbacks import EarlyStopping
import optuna
import lava_dl.src.lava.lib.dl.slayer as slayer

best_epoch = 0
best_model = None
best_loss = 100.0
best_val_history = []
best_train_history = []

def objective(trial, encoding,input_size,time_steps):
    
    time_steps = time_steps
    module_name = importlib.import_module("src.encodings")
    function_encoding = getattr(module_name,encoding)
    data_set = SMILESDataset(mode="train", transform=function_encoding,time_steps=time_steps)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True)

    val_losses = []
    
    samples = list(data_set.df["smiles"])
    targets = list(data_set.df["exp_mean [nM]"].values)
    for fold, (train_ids, val_ids) in enumerate(skf.split(samples,targets)):
        batch_size = 64
        train_set = torch.utils.data.dataset.Subset(data_set,train_ids)
        val_set = torch.utils.data.dataset.Subset(data_set,val_ids)
                
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_set,batch_size=batch_size, shuffle=True)
    
        device = torch.device('cuda')
        net = SLAYERDense(trial=trial,input_size=input_size).to(device)
        stats = slayer.utils.LearningStats()
        epochs = 100 # very high number that will never be reached because we have early stopping

        lr = trial.suggest_float("lr", 1e-5, 0.5, log=True)
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
        optimizer = getattr(optim, optimizer_name)(net.parameters(), lr=lr)
        true_rate = trial.suggest_float("true_rate",0.8,1.0)
        false_rate = trial.suggest_float("false_rate",0.0,0.2)

        error = slayer.loss.SpikeRate(true_rate=true_rate, false_rate=false_rate, reduction="mean").to(device)

        assistant = slayer.utils.Assistant(net, error, optimizer, stats, classifier=slayer.classifier.Rate.predict)
        
        callbacks = [EarlyStopping(patience=5, delta=1e-5, criterion="val_loss", mode="min")]
        history = {"train_loss": list(), "val_loss": list()}
        
        for epoch in range(epochs):
            for i, (sample, label) in enumerate(train_loader): # training loop
                output = assistant.train(sample, label)
            history["train_loss"].append(stats.training.loss)

            # prune on validation loss
            for i, (sample, label) in enumerate(val_loader): # training loop
                output = assistant.valid(sample, label)
            history["val_loss"].append(stats.validation.loss)
            stop_training = False
            if callbacks is not None:
                for callback in callbacks:
                    callback.on_epoch_end(epoch_ix=epoch, history=history)
                stop_training_flags = [callback.stop_training for callback in callbacks]
                stop_training = stop_training | (sum(stop_training_flags) > 0)
            if stop_training:
                epochs = epoch
                trial.set_user_attr("num_epochs", epoch)
                break
            stats.update()
    
        # At end of training, report validation loss
        for i, (sample, label) in enumerate(val_loader): # training loop
                    output = assistant.valid(sample, label)
        final_val_loss = stats.validation.loss
        val_losses.append(final_val_loss)
        global best_epoch, best_model, best_loss, best_val_history, best_train_history
        if final_val_loss < best_loss:
            best_epoch = epochs
            best_model = net
            best_loss = final_val_loss
            best_val_history = history["val_loss"]
            best_train_history = history["train_loss"]
    
    return sum(val_losses) / len(val_losses)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description ='Set up right HPO serch')
    parser.add_argument("--dataset", help='Which dataset to choose for training', choices={"OHE", "morgan1024x1","macc167x1","macc167x16", "macc167xT", "morgan1024xT","morgan1024x16"})
    parser.add_argument("--input_size", help="Number of input neurons",type=int)
    parser.add_argument("-t", help="Number of time steps",type=int)
    args = parser.parse_args()

    study_name = f"SLAYER_dense_{args.dataset}_{args.t}"
    path=f'./experiments/{study_name}/'
    os.makedirs(path, exist_ok=True)
    storage_name = "sqlite:///{}{}.db".format(path,study_name)
    study = optuna.create_study(direction="minimize", 
                                 study_name=study_name, 
                                 storage=storage_name,
                                 load_if_exists=True)
    study.optimize(lambda trial: objective(trial,args.dataset,args.input_size,args.t), n_trials=100)

    all_trials = [t for t in study.trials]
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print('Study statistics: ')
    print("Number of finished trials: ", len(study.trials))
    print("Number of complete trials: ", len(completed_trials))

    print("Best trial:")
    trial = study.best_trial
    print("Value: ", trial.value)

    print("Params: ")
    for key, value in trial.params.items():
        print("{}: {}".format(key, value))
    all_attr = [t.user_attrs for t in all_trials]
    results = {"best_val_loss": trial.value, "all trials": all_attr,"dataset": args.dataset,"best_hyperparameters": trial.params, "best_epoch": best_epoch,"train_loss": best_train_history,"val_loss": best_val_history}
    
    with open(f'{path}{study_name}.json', "w") as f:
        print("saving results")
        json.dump(results, f, indent=4)
    
    best_model.export_hdf5(f'{path}{study_name}.net')
    torch.save(best_model.state_dict(), f'{path}{study_name}.pt')

    plot_param_importances(study)
    plt.savefig(f'{path}/result.png')
