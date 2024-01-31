"""
Starting point for training a new model!
"""

import yaml # for loading config files
import argparse # config file via command line
import torch # autograd, optimization
import numpy as np
import os 
import matplotlib.pyplot as plt # plot results

import model # transformer model
import data # vocabulary, dataset, dataloader

def main(cfg):
    """
    Options for:
    - Pretraining a new model,
    - Finetuneing a pretrained model,
    - Evaluating a model.
    """
    # save some basic path stems
    cfg['results_path'] = f"results/{cfg['name']}"
    cfg['saved_path'] = f"saved/{cfg['name']}"
    # and specific paths for important files
    cfg['results_f'] = f"results/{cfg['name']}_results.yaml"
    cfg['model_f'] = f"saved/{cfg['name']}.pt"

    if cfg["mode"] == "pretrain":
        pretrain(cfg)
    elif cfg["mode"] == "finetune":
        finetune(cfg)
    elif cfg["mode"] == "evaluate":
        evaluate(cfg)
    else:
        raise ValueError("mode must be one of: pretrain, finetune, evaluate, not {}".format(cfg["mode"]))

def pretrain(cfg):
    """
    Pretrain a new model.
    """

    vocab = data.Vocabulary(cfg["data_f"], cfg["smiles_vocab_f"])
    dataloader = data.make(cfg)
    tf = model.make(cfg) # will load existing if possible
    optimizer = torch.optim.AdamW(tf.parameters(), lr=cfg["lr"])
    loss_fn = torch.nn.CrossEntropyLoss()

    data_iter = iter(dataloader)
    # open existing results if possible
    if os.path.exists(cfg["results_f"]):
        try:
            results = yaml.unsafe_load(open(cfg["results_f"], "rb"))
            n_batches = results[-1]["n_batches"] # most recent batch number
            print(f"Loaded existing results from {cfg['results_path']}")
        except FileNotFoundError:
            raise Exception(\
                f"Error opening results path {cfg['results_path']}")
    else:
        n_batches = 0
        results = []
        print("No existing results found, starting from scratch")

    while True:
        try:
            batch = next(data_iter)
        except StopIteration: # end of dataset, start again!
            data_iter = iter(dataloader)
            batch = next(data_iter)
    
        x, y = batch # (batch_size, block_size)
        y_hat = tf(x) # (batch_size, block_size, vocab_size)
        """
        Cross Entropy Loss accepts (N, C) and (N),
        where N is the batch size and C is the number of classes.
        Flatten y_hat and y to (N * C) and (N) respectively.
        """
        y_hat = y_hat.view(-1, y_hat.size(-1)) # (batch_size * block_size, vocab_size)
        y = y.view(-1) # (batch_size * block_size)
        loss = loss_fn(y_hat, y) # cross entropy against indices
        loss.backward() # compute gradients
        optimizer.step() # update parameters
        optimizer.zero_grad() # reset gradients

        # logging, saving
        n_batches += 1
        results.append({"loss": loss.item(), "n_batches": n_batches})
        for k, v in results[-1].items():
            print("{}: {}".format(k, round(v,3)))
        if n_batches % cfg["eval_every"] == 0:
            mols = tf.sample(cfg['n_samples'], cfg['sample_len'])
            # change from indices to strings
            mols = [vocab.decode(mol) for mol in mols]
            print("Random samples:")
            for mol in mols:
                print(mol)
            save_mol_text = "\n".join(mols)
            with open(f"{cfg['results_path']}_samples.txt", "a") as f:
                f.write(f"***Batch {n_batches}***\n")
                f.write(save_mol_text)
            print(f"Saved samples to {cfg['results_path']}_samples.txt")
        if n_batches % cfg["save_every"] == 0:
            torch.save(tf.state_dict(), cfg["model_f"])
            with open(cfg['results_f'], "w") as f:
                yaml.dump(results, f)
            plot_results(results, cfg["results_path"])

def plot_results(results, results_path):
    """
    Simple matplotlib graph for loss.
    """
    save_as = f"{results_path}_loss.png"
    x = [r["n_batches"] for r in results]
    y = [r["loss"] for r in results]
    plt.plot(x, y)
    plt.xlabel("Batch number")
    plt.ylabel("Loss")
    plt.title("Loss over time")
    plt.savefig(save_as)
    print(f"Saved loss graph to {save_as}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="config file path")
    args = parser.parse_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    main (cfg)
