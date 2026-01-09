import numpy as np
import random
import matplotlib.pyplot as plt
import keras
import os

import sys
sys.path.append("..")
import trial_history


directory="../../training/"
name = "hyperparameter"

# List of directories to create if don't already exist
directories = [directory+"figures", directory+"models", directory+"results"]
for dir in directories:
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(f"Directory created: {dir}")


params = {"axes.labelsize": 14,
          "axes.titlesize": 16,}
plt.rcParams["axes.linewidth"] = 1
plt.rcParams['mathtext.bf'] = 'STIXGeneral:italic:bold'
plt.rcParams['figure.dpi'] = 100
plt.rcParams.update(params)

def place(ax):
  ax.tick_params(direction="in", which="minor", length=3)
  ax.tick_params(direction="in", which="major", length=5, labelsize=13)
  ax.grid(which="major", ls="dashed", dashes=(1, 3), lw=0.8, zorder=0)
  #ax.legend(frameon=True, loc="best", fontsize=12,edgecolor="black")
  fig.tight_layout()


def combine_data(xbins, rho, muloc, c1, T, mu):
    data = {}

    data = np.zeros(xbins.shape, dtype=[('xbins', 'f8'),
                                        ('rho', 'f8'), ('muloc', 'f8'), ('c1', 'f8'),
                                        ('T', 'f8'), ('mu', 'f8')])


    data['xbins'] = xbins
    data['rho'] = rho
    data['muloc'] = muloc
    data['c1'] = c1
    data['mu'] = mu
    data['T'] = T
    return data

    # Load the simData dictionary from the file
simData_T1 = np.load(directory+"data/WCA_T1.npy", allow_pickle=True).item()
simData_T1_5 = np.load(directory+"data/WCA_T1.5.npy", allow_pickle=True).item()
simData_T2 = np.load(directory+"data/WCA_T2.npy", allow_pickle=True).item()

simData = {}

for key in ['training', 'validation', 'test']:
    # Combine the inner dictionaries
    simData[key] = { **simData_T1.get(key, {}), **simData_T1_5.get(key, {}), **simData_T2.get(key, {})}

# Combine all simulations into one list
all_simulations = []
for category in ['training', 'validation', 'test']:
    all_simulations.extend(list(simData[category].keys()))


for sim in all_simulations:
    category = next(cat for cat in simData if sim in simData[cat])
    data = simData[category][sim]
    xbins = data['xbins']
    rho = data['rho']
    muloc = data['muloc']
    c1 = data['c1']
    T = data['T']
    mu = data['mu']

    sim_name = sim + "_mirror"
    combined_data_mirror = combine_data(xbins, rho[::-1],
                                    muloc[::-1], c1[::-1], T, mu)
    
    simData[category][sim_name] = combined_data_mirror

    
# Combine all simulations into one list
all_simulations = []
for category in ['training', 'validation', 'test']:
    all_simulations.extend(list(simData[category].keys()))

print("Number of simulations = {}".format(len(all_simulations)))


# Inspect the Data

# Select a random simulation
random_sim = random.choice(all_simulations)

# Determine which category the random simulation belongs to
category = next(cat for cat in simData if random_sim in simData[cat])


# Get the data for the random simulation
data = simData[category][random_sim]

# Extract z, rho, muloc, c1, T
xbins = data['xbins']

rho = data['rho']
muloc = data['muloc']
c1 = data['c1']
T = data['T']

# Plot muloc(z), rho(z), and c1(z)
fig, ax = plt.subplots(3, 1, figsize=(5,6), sharex='all')


ax[0].plot(xbins, muloc, color='deepskyblue')

ax[0].set_ylabel(r'$\beta\mu - \beta V_{\mathrm{ext}}(x)$')
ax[0].set_title(f'{random_sim}')

ax[1].plot(xbins, rho, color='deepskyblue')
ax[1].set_ylabel(r'$\rho(x)$')


ax[2].plot(xbins, c1, color='deepskyblue')
ax[2].set_ylabel(r'$c^{(1)}(x)$')
ax[2].set_xlabel(r'$x$ [$\mathrm{\sigma}$]')
ax[0].legend(frameon=True, loc="best", fontsize=12,edgecolor="black")
#ax[2].set_xlim(0, 20)

place(ax[1])
place(ax[0])
place(ax[2])

plt.savefig(directory+"figures/inspect_data_{}_{}.png".format(name, random_sim))
plt.show()

# Curate data for training, sliding window approach

import sys
sys.path.append("..")
from data_generators import DataGenerator_inhom

# Generator options
generatorOptions = {
    "batch_size": 256,
    "windowSigma": 3.00,
    "inputKeys": ["rho"],
    "paramsKeys": ["T"],
    "outputKeys": ["c1"],
    "binKey": "xbins",
}

# Create data generators
trainingGenerator = DataGenerator_inhom(simData["training"], **generatorOptions)
validationGenerator = DataGenerator_inhom(simData["validation"], **generatorOptions)

# Create neural network for model

def build_model(hp):
    units = hp.Choice("units", [32, 64, 128, 256, 512])
    
    # Define the model inputs
    rho_input = keras.Input(shape=trainingGenerator.inputShape, name="rho")
    params_input = {paramKey: keras.Input(shape=(1,), name=paramKey) for paramKey in ["T"]}

    # Concatenate the two inputs
    x = keras.layers.Concatenate()([rho_input, *params_input.values()])
    x = keras.layers.Dense(units, activation="softplus")(x)
    x = keras.layers.Dense(units, activation="softplus")(x)

    # learn the c1
    output = keras.layers.Dense(trainingGenerator.outputShape[0], name="c1")(x)

    model = keras.Model(inputs=[rho_input, *params_input.values()], outputs=output)

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanAbsoluteError()]
    )

    model.summary()

    keras.utils.plot_model(model, show_shapes=True, show_layer_names=True ,show_layer_activations=True, dpi=80, to_file=directory+'models/{}.png'.format(name))
        
    return model

tuner = trial_history.GridSearch(
        build_model,
        objective="val_mean_absolute_error",
        max_trials=10,
        executions_per_trial=2,
        directory=directory+"results",
        overwrite=True,
        project_name="{}".format(name),
        )

tuner.search_space_summary()



# Train neural network

import callbacks as cb


# Define the callbacks
callbacks = [
    keras.callbacks.LearningRateScheduler(cb.lrschedule),
    keras.callbacks.ModelCheckpoint(
        filepath=directory+"models/WCA_{}.keras".format(name), # Name of checkpoint file
        monitor="val_mean_absolute_error",
        save_best_only=True),
    keras.callbacks.EarlyStopping(
        monitor="val_mean_absolute_error",
        patience=100),
    cb.LossHistory()]

tuner.search(
        trainingGenerator,
        validation_data=validationGenerator,
        epochs=100,
        callbacks=callbacks,
        )


models = tuner.get_best_models(1)
best_model = models[0]
best_model.summary()
tuner.results_summary()

best_hps = tuner.get_best_hyperparameters(1)

model = build_model(best_hps[0]) # build model with best hp
model.save(directory+"models/WCA_{}.keras".format(name))

testGenerator = DataGenerator_inhom(simData["test"], **generatorOptions)
test_metrics = model.evaluate(testGenerator)



# See predicted correlation function of test set


def generate_windows(array, bins, mode="wrap"):

    padded_array = np.pad(array, bins, mode=mode)
    windows = np.empty((len(array), 2 * bins + 1))
    for i in range(len(array)):
        windows[i] = padded_array[i:i + 2 * bins + 1]
    return windows


def c1_pred(model, rho, T, input_bins=1201):


    window_bins = (input_bins - 1) // 2
    rho_windows = generate_windows(rho, window_bins).reshape(rho.shape[0], input_bins, 1)

    c1_result = model.predict_on_batch([rho_windows, T]).flatten()
    return c1_result


# Combine all tests simulations into one list
all_test_simulations = []
for category in ['test']:
    all_test_simulations.extend(list(simData[category].keys()))


# Select a random simulation
random_sim = random.choice(all_test_simulations)

# Determine which category the random simulation belongs to
category = next(cat for cat in simData if random_sim in simData[cat])


# Get the data for the random simulation
data = simData[category][random_sim]

# Extract z, rho, muloc, and c1
xbins = data['xbins']
rho = data['rho']
muloc = data['muloc']
c1 = data['c1']
T = data['T']

# Plot muloc(z), rho(z), and c1(z)
fig, ax = plt.subplots(3, 1, figsize=(5,6), sharex='all')


ax[0].plot(xbins, muloc, color='pink')

ax[0].set_ylabel(r'$\beta\mu - \beta V_{\mathrm{ext}}(x)$')
ax[0].set_title(f'{random_sim}')

ax[1].plot(xbins, rho, label='H', color='pink')

ax[2].plot(xbins, c1, color='pink', label="sim", lw=2)

c1_prediction = c1_pred(model, rho, T)

ax[2].plot(xbins, c1_prediction, label='predicted', color='hotpink', ls='--')


ax[1].set_ylabel(r'$\rho(x)$')
ax[2].set_ylabel(r'$c^\mathrm{(1)}(x)$')

ax[2].legend()
#ax[2].set_xlim(0, 24)

place(ax[1])
place(ax[0])
place(ax[2])

plt.savefig(directory+"figures/predicted_data_{}_{}.png".format(name, random_sim))
plt.show()