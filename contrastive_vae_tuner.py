import os
import sys
import argparse
import numpy as np
import h5py
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts as CAWR
h5=h5py

from loss import *
from model import *
from utilities import *

import wandb
import ray
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray import tune

import gc

from squark.config import QuantizerConfigScope
from squark.constraints import MinMax


#this function is used to find the reconstruction-based distance metric to measure how well the given data is reconstructed by the model.

#higher loss means the sample is less likely under the trained distribution
def distance(model, data):

    #encode the input data
    mean, _ = model.Encoder(data)

    #decoder descontructs the input
    reco = model.Decoder(mean)

    #computes the per-sample reconstruction error
    score = nn.functional.mse_loss(data,reco,reduction = 'none').mean(-1)

    return score.to("cpu").detach().numpy()



def run(config):

    #login for wandb
    wandb.login(key="24d1d60ce26563c74d290d7b487cb104fc251271")

    #Name the project in wandb
    wandb.init(project = "C_VAE_test",
               settings=wandb.Settings(_disable_stats=True),
               config = config)
    
    
    #extract hyperparameters from the Ray Tune search space
    blur_p = config['blur_p']
    blur_m = config['blur_m']
    blur_s = config['blur_s']

    mask_p = config['mask_p']

    beta = config['beta']
    VIC_lr = config['vic_lr']
    VAE_lr = config['vae_lr']
    alpha = config['alpha']

    reco_scale = alpha * (1 - beta)
    kl_scale = beta

    #device = 'cuda:0'  # This will be set later
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")

    # Epochs_contrastive = 50  # This will be set later
    Epochs_contrastive = 1  # This will be set later
    
    #Epochs_VAE = 480
    Epochs_VAE = 5

    #the number of training samples processed before the model updates its weights in one iteration of training
    Batch_size = 4096

    #Extracts the number of neurons in each layer of the VICReg encoder.
    vic_encoder_nodes = config['encoder_nodes']

    #This determines the size of the projection head in VICReg, which is used for self-supervised learning.
    projector_features = vic_encoder_nodes[-1] * 4

    #Defines the hidden layers in the VAE encoder.
    vae_encoder_nodes = config['vae_nodes']

    #Defines the size of the latent space (bottleneck representation) in the VAE.
    vae_latent_dim = config['vae_latent']

    # Making it symmetric
    vae_decoder_nodes = [vic_encoder_nodes[-1]] + vae_encoder_nodes.copy()
    vae_decoder_nodes.reverse()

    # --------------------------------------------------------------------------

    #upload the data
    f = h5.File('/home/ellison5/diptarko_code/Data.h5', 'r')

    #define training data and testing data
    x_train = f['Background_data']['Train']['DATA'][:]
    x_test = f['Background_data']['Test']['DATA'][:]

    #define the background train and test data and reshape it
    x_train_background = np.reshape(x_train, (x_train.shape[0], -1))
    x_test_background = np.reshape(x_test, (x_test.shape[0], -1))

    #normalize
    scale = f['Normalisation']['norm_scale'][:]
    bias = f['Normalisation']['norm_bias'][:]

    
    l1_bits_bkg_test = f['Background_data']['Test']['L1bits'][:]

    # --------------------------------------------------------------------------

    #Applies random feature blurring to encourage robustness in the model.
    feature_blur = FastFeatureBlur(p=blur_p, strength=blur_s, magnitude=blur_m, device=device)
    feature_blur_prime = FastFeatureBlur(p=blur_p, strength=blur_s, magnitude=blur_m, device=device)

    #Simulates oscillations or missing parts of the input, making the model learn to recognize objects despite missing data.
    object_mask = FastObjectMask(p=mask_p, device=device)
    object_mask_prime = FastObjectMask(p=mask_p, device=device)

    #Applies Lorentz transformations (which could be spatial distortions, rotations, or time-space alterations), 
    #possibly    making the model robust to geometric variations.
    lorentz_rot = FastLorentzRotation(p=0.5, norm_scale=scale, norm_bias=bias, device=device)
    lorentz_rot_prime = FastLorentzRotation(p=0.5, norm_scale=scale, norm_bias=bias, device=device)

    # --------------------------------------------------------------------------
    #converts x_train_background and x_test_background into PyTorch tensors
    dataset = torch.tensor(x_train_background, dtype=torch.float32, device=device)
    dataset_test = torch.tensor(x_test_background, dtype=torch.float32, device=device)
    del x_train_background
    gc.collect()

    # --------------------------------------------------------------------------

    #training loop for VICReg model using quantization-aware training and data augmentations

    #Quantization-aware training (QAT) is applied to both weights and activations.- keeps weights activations and bias quantized
    weight_config = QuantizerConfigScope('kbi', ('weight', 'bias'), bc=MinMax(0, 6), b0=4)
    # bc = bit-width constraint, b0 = initial bit-width
    # More options in docstring of squark.config.QuantizerConfig
    activation_config = QuantizerConfigScope('kif', 'datalane', f0=4)

    #loads the quantization-aware VICReg model
    with weight_config, activation_config:
        Backbone = ModelBackbone(nodes=vic_encoder_nodes)
        Projection = ModelProjector(projector_features)

        #uploads the VICReg model and names it model
        model = VICReg(backbone=Backbone, projector=Projection, num_features=projector_features, batch_size=Batch_size)

    #moves the model to GPU or CPU for training
    model = model.to(device)

    #optimizer for training
    optimizer = torch.optim.Adam(model.parameters(), lr=VIC_lr)

    #gradually reduces the learning rate over time
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Epochs_contrastive, eta_min=0.0)

    #puts the model in training mode
    model.train()


    #training the VICReg model
    for present_epoch in tqdm(range(0, Epochs_contrastive, 1)):
        train_loss = 0
        train_steps = 0
 
        #shuffles the data by generating a random permutation of indices
        index = torch.randperm(dataset.shape[0])

        #processes the dataset in batches of Batch_size
        for i in range(dataset.shape[0] // Batch_size):
            batch = dataset[index[i * Batch_size : (i + 1) * Batch_size]]

            #creates two separate view of the same batch (like in the paper)
            batch_x = batch.clone()
            batch_y = batch.clone()

            #applies, blur, mask, and lorentz transformation
            batch_x = feature_blur(batch_x)
            batch_x = object_mask(batch_x)
            batch_x = lorentz_rot(batch_x)

            batch_y = feature_blur_prime(batch_y)
            batch_y = object_mask_prime(batch_y)
            batch_y = lorentz_rot_prime(batch_y)

            # optimizer.zero_grad()
            for param in model.parameters():
                param.grad = None

            #computes VICReg loss 
            loss, repr_loss, std_loss, cov_loss = model(batch_x, batch_y)

            #backpropagates gradients
            loss.backward()

            #updates model weights
            optimizer.step()
            train_loss += loss.item()
            train_steps += 1

        #accumulates training loss for averaging later
        train_loss = train_loss / train_steps

        #updates learning rate based on Cosine annealing scheduler
        scheduler.step()

        #logs everything to wandb to track model performance over time
        metric_embed = {}
        metric_embed['TrainLossC'] = train_loss
        metric_embed['EpochC'] = present_epoch
        metric_embed['LrC'] = scheduler.get_last_lr()[0]
        wandb.log(metric_embed)

    #########################################################################################################################
    #########################################################################################################################
    #########################################################################################################################

    #extracts the trained VICReg backbone (feature extractor)
    vic_encoder = model.backbone

    #puts the encoder in evaluation mode
    vic_encoder.eval()

    #defines and creates a VAE
    with weight_config, activation_config:
        encoder = VAE_Encoder(nodes=vae_encoder_nodes, feature_size=vae_latent_dim)
        decoder = VAE_Decoder(nodes=vae_decoder_nodes)
        model = VarationalAutoEncoder(Encoder=encoder, Decoder=decoder, device=device).to(device)

    #updates VAE weights
    optimizer = torch.optim.Adam(model.parameters(), lr=VAE_lr)

    #uses Cosine annealing with Warm Restarts as the learning rate scheduler
    scheduler = CAWR(optimizer, first_cycle_steps=32, cycle_mult=2, max_lr=VAE_lr, min_lr=0, warmup_steps=10, gamma=0.65)

    #converts the raw dataset into VICReg-learned embeddings
    #The extracted latent representations will be used as input to the VAE
    with torch.no_grad():
        dataset_latent = vic_encoder(dataset.float())
        dataset_latent_test = vic_encoder(dataset_test.float())

    #############################################
    # Signal Data
    #############################################

    #loads signal names from an HDF5 file
    SIGNAL_NAMES = list(f['Signal_data'].keys())

    #stores VICReg-transformed signal data
    signal_data_dict = {}

    #Stores L1 trigger bits for each signal
    signal_l1_dict = {}


    #converts raw signal data into VICReg-encoded latent representations making them suitable for VAE training
    for signal_name in SIGNAL_NAMES:
        
        #converts signal data into a PyTorch tensor
        x_signal = torch.tensor(f['Signal_data'][signal_name]['DATA'][:], dtype=torch.float32, device=device)

        #flattens each  signal sample into a vector for VAE processing
        x_signal = torch.reshape(x_signal, (x_signal.shape[0], -1))

        #passes the signal through VICReg to extract latent features
        x_signal = vic_encoder(x_signal)

        #loads L1 trigger bits
        l1_bits = f['Signal_data'][signal_name]['L1bits'][:]

        signal_data_dict[signal_name] = x_signal
        signal_l1_dict[signal_name] = l1_bits
    f.close()

    #training the VAE
    for present_epoch in tqdm(range(0, Epochs_VAE, 1)):
        model.train()

        train_loss = 0
        train_steps = 0

        #shuffles dataset indices and splits dataset into batches
        index = torch.randperm(dataset_latent.shape[0])
        for i in range(dataset_latent.shape[0] // Batch_size):
            batch = dataset_latent[index[i * Batch_size : (i + 1) * Batch_size]]

            # optimizer.zero_grad()
            for param in model.parameters():
                param.grad = None

            #forward pass through VAE - (feeding an input through the encoder to obtain latent representation, 
            #then passing it through the decoder to reconstruct the original input. Outputs are reconstructed input, 
            #mean of latent distribution and log variance of latent distribution
            reconstruction, mean, log_var = model(batch)

            
            model.Encoder.model

            #use mean squared error to measure reconstruction quality
            reconstruction_loss = reco_scale * nn.functional.mse_loss(reconstruction, batch, reduction='sum')  # one value

            #use KL divergence loss to ensure a structured latent space
            kl_loss = kl_scale * kl_div(mean, log_var)

            #Quantization-aware loss for both encoder and decoder
            hgq_loss = sum(model.Encoder.model.losses) + sum(model.Decoder.model.losses)

            #calculated final loss
            loss = torch.mean(reconstruction_loss + kl_loss) + hgq_loss

            #computes gradients
            loss.backward()

            #updates model parameters
            optimizer.step()

            #accumulates loss for averaging later
            train_loss += loss.item()
            train_steps += 1

        #computes average training loss
        train_loss = train_loss / train_steps

        #evaluates how well the VAE distinguishes background v. signal
        metric = fast_score(
            model=model,
            data_bkg=dataset_latent_test,
            bkg_l1_bits=l1_bits_bkg_test,
            distance_func=distance,
            data_signal=signal_data_dict,
            signal_l1_bits=signal_l1_dict,
            evaluation_threshold=1,
        )

        #logs everything to wandb
        metric['TrainLossVae'] = train_loss
        metric['EpochVae'] = present_epoch
        metric['LrVae'] = scheduler.get_lr()[-1]

        wandb.log(metric)

        #reports evaluation results using ray.train
        ray.train.report(metrics=metric)

        #adjusts learning rate after each epoch
        scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--address', type=str, default=None)

    args = parser.parse_args()

    if args.address:
        ray.init(address=args.address)
    else:
        ray.init(address='auto')

    search_space = {
        'vic_lr': tune.loguniform(1e-5, 1e-3),
        'vae_lr': tune.loguniform(1e-5, 1e-3),
        'blur_p': tune.uniform(0, 1),
        'blur_m': tune.uniform(0, 1),
        'blur_s': tune.uniform(0, 1),
        'mask_p': tune.uniform(0, 1),
        'beta': tune.uniform(0, 1),
        'alpha': tune.uniform(0, 1),
        'encoder_nodes': tune.sample_from(lambda spec: [tune.randint(24, 32).sample(), tune.randint(8, 18).sample()]),
        'vae_latent': tune.sample_from(lambda spec: tune.randint(4, 12).sample()),
        'vae_nodes': tune.sample_from(lambda spec: [tune.randint(6, 12).sample(), tune.randint(3, 12).sample()]),
    }

    optuna_search = OptunaSearch(
        metric='pure-pure/haa4b-ma15',
        mode='max',
    )

    scheduler = ASHAScheduler(
        metric='pure-pure/haa4b-ma15',
        mode='max',
        max_t=480,
        grace_period=120,
        reduction_factor=2,
    )

    analysis = tune.run(
        run,
        config=search_space,
        # storage_path='/pscratch/sd/d/diptarko/TECH-L1AD/FinalModelDevelopment/ray_tune_experiments',
        search_alg=optuna_search,
        scheduler=scheduler,
        # num_samples=1000,
        num_samples=1,
        # resources_per_trial={'cpu': 8, 'gpu': 1 / 4},
        resources_per_trial={'cpu': 8},
    
    )
