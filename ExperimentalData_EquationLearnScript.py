import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from scipy.integrate import odeint
from scipy.optimize import nnls
import numpy as np
import matplotlib.pyplot as plt
from ProcessData import load_and_process

# Code to run the equation learning framework in "Biologically-informed neural networks and equation
# learning reveal nano-engineered particle-cell interactions" by Johnston and Faria for experimental data.
# Experimental data is assumed to be in ('./Flow Data')

dataSource = "leo"                                                              # Data library "leo" or "fuchs"
saveFigure = "no"                                                               # Choice of whether to save figure upon completion

if dataSource == "leo":
    particleTypes = ["1032nm_capsule_thp1",                                     # Types of particles present in the data library
                     "150nm_coreshell_thp1",
                     "214nm_capsule_thp1",
                     "282nm_coreshell_thp1",
                     "480nm_capsule_thp1",
                     "633nm_coreshell_thp1",
                     "1032nm_capsule_thp1"]
    particleFuncs = ["NA"]                                                      # No functionalisation on particles
    concentration = [0]                                                         # Particle concentration number
    timeVals = np.array([0.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1440.0])        # Time (minutes) of measurements
    timeValsHours = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 24.0])             # Time (hours) of measurements
elif dataSource == "fuchs":
    particleTypes = ["PIP", "FLU", "PIP"]                                       # Types of particles present in the data library
    particleFuncs = ["UG", "control", "J591", "peptide", "UG"]                  # Particle functionalisations in the data library
    concentration = [1, 2, 3]                                                   # Particle concentration number
    timeVals = np.array([0.0, 30.0, 120.0, 240.0])                              # Time (minutes) of measurements
    timeValsHours = np.array([0.0, 0.5, 2.0, 4.0])                              # Time (hours) of measurements

# Script parameters

nDatasets = np.size(particleTypes)*np.size(particleFuncs)*np.size(concentration)    # Number of datasets in the library
nTerms = 10                                                                         # Maximum order of associated nanoparticles in equation learning
nTimesUsed = np.zeros(nTerms+1)                                                     # Array of counts of the times each model component is used
modelCounter = np.zeros(2**(nTerms+1))                                              # Array of counts of the number each potential model is learned
nRepeats = 100                                                                      # Number of realisations of the equation learning process
repeats = range(0, nRepeats)                                                        # Replicate numbers
storeModel = np.zeros((nDatasets, nRepeats))                                        # Store the model number for each realisation
counter = 0
nEpochs = 30                                                                    # Number of epochs in the neural network
nNodes = 128                                                                    # Number of nodes in each hidden layer of the neural network
trainDataFraction = 0.6                                                         # Fraction of data used for training
validateDataFraction = 0.3                                                      # Fraction of data used for validation
testDataFraction = 1.0 - trainDataFraction - validateDataFraction               # Fraction of data used for testing

maxTime = max(timeValsHours)                                                    # Final time in data (hours)
nTimePoints = timeVals.size                                                     # Number of time steps in data


# Define function to prune unnecessary model terms
def pruneModel(rhsTermsIn, dydtIn, firstError):
    """
    :param rhsTermsIn: Equation learning model components
    :param dydtIn: Data to be constructed from model components
    :param firstError: Error of unpruned model
    :return: Coefficients of pruned model
    """
    outputCoef = np.ones(rhsTermsIn.shape[1])                       # Preallocate output coefficients
    indicesStore = range(0, rhsTermsIn.shape[1])                    # Potential indices
    for ii in range(rhsTermsIn.shape[1] - 1, -1, -1):               # Loop over indices (backwards)
        indices = np.delete(indicesStore, ii)                       # Remove iith index
        if indices.size > 0:
            proposedError = nnls(rhsTermsIn[:, indices], dydtIn)[1]
            if proposedError < 1.25 * firstError:                        # If error for reduced model is <25% worse
                outputCoef[ii] = 0                                      # than full model, remove iith term
                indicesStore = np.delete(indicesStore, ii)

    return np.nonzero(outputCoef)[0]                                # Return output coefficients


# Proposed dynamical systems model
def odesystem(y_ode, t, coefs_ode, intercept_ode):
    """
    :param y_ode: Prediction of associated nanoparticles
    :param t: Time in ODE
    :param coefs_ode: Coefficients of each model component
    :param intercept_ode: Coefficient of 0th order term
    :return: ODE Solution
    """
    dydt_ode = intercept_ode
    for iode in range(0, coefs_ode.size):
        dydt_ode += coefs_ode[iode] * (y_ode ** (iode + 1))
    return dydt_ode

# Loop over each particle type, functionalisation and concentration
for iParticleType in particleTypes:
    for iParticleFuncs in particleFuncs:
        for iConcentration in concentration:
            counter += 1
            for iRepeat in repeats:
                Parameters = {}
                Data = {}

                # Define parameters of relevant dataset
                Parameters["dataSource"] = dataSource
                Parameters["particleType"] = iParticleType
                Parameters["particleFunc"] = iParticleFuncs
                Parameters["concentration"] = iConcentration

                # Load data corresponding to each observation time
                for i in timeVals:
                    Data["%d" % i] = load_and_process(Parameters, i)                            # Load data for time i
                    Data["%d Mean" % i] = np.mean(Data["%d" % i])                               # Mean data for time i
                    Data["%d Time" % i] = i/60 * np.ones(len(Data["%d" % i])).reshape(-1, 1)    # Time i (hours)

                originalMeanData = np.zeros(nTimePoints)                                        # Store original mean of data
                originalTime = timeValsHours                                                    # Store original time points
                dataIn = np.array([])                                                           # Initialise neural network input
                dataOut = np.array([])                                                          # Initialise neural network output

                # Store mean, input and output for each observation time
                for i in range(0, nTimePoints):
                    originalMeanData[i] = Data["%d Mean" % timeVals[i]]
                    dataIn = np.append(dataIn, Data["%d Time" % timeVals[i]])
                    dataOut = np.append(dataOut, Data["%d" % timeVals[i]])

                dataIn = dataIn.reshape(-1, 1)
                dataOut = dataOut.reshape(-1, 1)
                dataIn = maxTime - dataIn                               # Rescale time data for constraints
                scalerMethod = StandardScaler().fit(dataIn)             # Standardise input data to zero mean
                dataIn = scalerMethod.fit_transform(dataIn)             # Standardise input data to zero mean
                maxData = np.max(dataOut)                               # Maximum output value

                uniqueTime = np.unique(dataIn)                          # Unique values in transformed time

                # Rescale data to ensure positive 1st and 2nd derivative
                maxOutput = np.max(originalMeanData)                    # Maximum mean value
                dataOut = 1.0 - dataOut / maxOutput                     # Scale neural network output for constraints

                nData = dataIn.size                                     # Number of observations

                # Split into training, validation and test datasets
                dataIndices = np.linspace(0, nData - 1, nData, dtype=int)
                trainIndices = np.random.choice(dataIndices, int(np.ceil(nData * trainDataFraction)), replace=False)
                unusedIndices = np.setdiff1d(dataIndices, trainIndices)
                validateIndices = np.random.choice(unusedIndices, int(np.ceil(nData * validateDataFraction)), replace=False)
                testIndices = np.setdiff1d(unusedIndices, validateIndices)

                # Define training dataset
                trainIn = dataIn[trainIndices]
                trainOut = dataOut[trainIndices]
                meanTrain = np.zeros(uniqueTime.size)                   # Calculate mean of training dataset
                for i in range(0, uniqueTime.size):
                    meanTrain[i] = np.mean(trainOut[trainIn == uniqueTime[-(i+1)]])

                # Define validation dataset
                validateIn = dataIn[validateIndices]
                validateOut = dataOut[validateIndices]
                meanValidate = np.zeros(uniqueTime.size)                # Calculate mean of validation dataset
                for i in range(0, uniqueTime.size):
                    meanValidate[i] = np.mean(validateOut[validateIn == uniqueTime[-(i+1)]])

                # Define testing dataset
                testIn = dataIn[testIndices]
                testOut = dataOut[testIndices]
                meanTest = np.zeros(uniqueTime.size)                    # Calculate mean of testing dataset
                for i in range(0, uniqueTime.size):
                    meanTest[i] = np.mean(testOut[testIn == uniqueTime[-(i+1)]])

                timeCheck = np.linspace(np.min(dataIn), np.max(dataIn), 1000).reshape(-1, 1)    # Time values for solution
                timeData = max(dataIn) - (np.unique(dataIn) - min(dataIn))                      # Reverse time values for constraints
                meanData = 1.0 - originalMeanData / maxOutput                                   # Rescale mean data

                # Define checkpoints for neural network
                # Allows for minimum validation value to be selected
                checkpoint_filepath = './tmp/checkpoint'
                model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                    filepath=checkpoint_filepath,
                    save_weights_only=False,
                    monitor='val_loss',
                    mode='min',
                    save_best_only=True)

                # Define neural network
                model = tf.keras.models.Sequential([
                        tf.keras.layers.Dense(nNodes, activation='softplus', kernel_constraint="NonNeg"),
                        tf.keras.layers.Dense(1, activation='linear', kernel_constraint="NonNeg"),
                ])

                loss_fn = tf.keras.losses.MeanSquaredError()                                    # Loss function
                optimDef = tf.keras.optimizers.Adam(learning_rate=0.001)                        # Default optimizer
                model.compile(optimizer=optimDef, loss=loss_fn)                                 # Model details

                # Fit neural network model to training data and select best via validation
                outputModel = model.fit(trainIn,
                                        trainOut,
                                        epochs=nEpochs,
                                        validation_data=(validateIn, validateOut),
                                        callbacks=[model_checkpoint_callback])

                bestModel = tf.keras.models.load_model(checkpoint_filepath)                     # Load best model
                predictions = np.flip(1 - bestModel(timeCheck))                                 # Model details

                t = list(timeCheck)                                                             # Transform to tf data
                t = tf.constant(t)

                # Calculate derivative of neural network model with respect to time
                with tf.GradientTape() as tape:
                    tape.watch(t)
                    y = bestModel(t)

                dydt = tape.gradient(y, t).numpy()                                              # Derivatives at ODE time
                y = np.flip(1 - np.ravel(bestModel(t).numpy()))                                 # Model predictions at ODE time

                rhsTerms = np.zeros([len(y), nTerms])

                # Define RHS terms coresponding to each model component
                for iy in range(0, nTerms):
                    rhsTerms[:, iy] = y ** (iy + 1)

                shifted_dydt = -(np.flip(np.ravel(dydt))-np.max(np.ravel(dydt)))                # Reorient derivative
                sparse_Lasso = LassoCV(fit_intercept=False, max_iter=10000, positive=True)      # Cross validated sparse lasso fit (with enforced non-negativity)
                sparse_Lasso.fit(rhsTerms, shifted_dydt)                                        # Fit model components to derivative via LASSO

                nonzeroIndices = np.nonzero(sparse_Lasso.coef_)[0]                              # Non-zero model component coefficients

                # Perform model pruning for each non-zero model component (provided more than one model component)
                if nonzeroIndices.size > 1:
                    coefs, startError = nnls(rhsTerms[:, nonzeroIndices], shifted_dydt)         # Initial error in model
                    relevantIndices = pruneModel(rhsTerms[:, nonzeroIndices], shifted_dydt, startError)     # Prune model terms
                    coefs, _ = nnls(rhsTerms[:, nonzeroIndices[relevantIndices]], shifted_dydt)             # Calculate new model coefficients
                    coefStorage = np.zeros(nTerms)
                    coefStorage[nonzeroIndices[relevantIndices]] = coefs
                    coefs = coefStorage                                                                     # New model coefficients
                else:
                    coefs, startError = nnls(rhsTerms[:, nonzeroIndices], shifted_dydt)                     # Else just calculate coefficients
                    coefStorage = np.zeros(nTerms)
                    coefStorage[nonzeroIndices] = coefs
                    coefs = coefStorage                                                                     # Calculate new model coefficients

                # Re-transform parameters
                coefs = -1.0*coefs
                intercept = np.max(np.ravel(dydt))                                                          # Intercept in model

                # Store which model components are used
                modelNum = 0
                for i in range(0, nTerms):
                    if coefs[i] != 0:
                        nTimesUsed[i+1] += 1

                # Store if 0th order term is used
                if intercept != 0:
                    nTimesUsed[0] += 1

                # Calculate which model number is used (number = binary representation, 1 if model component used, 0 otherwise)
                for i in range(0, nTerms):
                    if coefs[i] != 0:
                        modelNum += 2**(i+1)

                if intercept != 0:
                    modelNum += 1

                modelCounter[modelNum] += 1
                storeModel[counter-1, iRepeat] = modelNum

                y0 = min(y)                                                                     # Initial condition for dynamical systems model
                time_ratio = maxTime / (np.ravel(timeCheck)[-1] - np.ravel(timeCheck)[0])       # Ratio of time in transformed and untransformed models
                ode_time = scalerMethod.inverse_transform(np.ravel(timeCheck))                  # Define time range for dynamical systems model

                # Scale model components by maximum output (i.e. undo initial scaling)
                for iy in range(0, nTerms):
                    rhsTerms[:, iy] *= maxOutput ** (iy + 1)

                # Scale coefficients and initial condition to be consistent with input data
                for im in range(0, nTerms):
                    coefs[im] /= maxOutput ** im

                intercept *= maxOutput / time_ratio
                coefs /= time_ratio
                y0 *= maxOutput

                # Solve differential equation model
                sol = odeint(odesystem, y0, ode_time, args=(coefs, intercept))


                # Uncomment to plot comparison between LASSO prediction
                # and neural network prediction of the association rate

                # fig, ax1 = plt.subplots()
                # ax1.plot(y * maxOutput, np.flip(dydt) / time_ratio * maxOutput, linewidth=3,
                #          label="Neural network derivative")
                # ax1.plot(y * maxOutput, np.inner(coefs, rhsTerms) + intercept, linestyle='--', linewidth=3,
                #          label="LASSO prediction")
                # ax1.set(xlabel="Particles per cell", ylabel="Particles per cell per time")
                # ax1.legend(loc="upper right")
                # ax1.set_xlim((0, maxOutput))
                # plt.savefig('./Results/DerivativeFit.eps')
                # plt.show()

                plotTime = np.flip(scalerMethod.inverse_transform(timeData))    # Recover data plotting time
                plotTimeCheck = scalerMethod.inverse_transform(timeCheck)       # Recover model plotting time
                plotTrain = (1.0 - meanTrain) * maxOutput                       # Rescale training data
                plotValidate = (1.0 - meanValidate) * maxOutput                 # Rescale validation data
                plotTest = (1.0 - meanTest) * maxOutput                         # Rescale testing data
                plotPredictions = maxOutput * predictions                       # Rescale predictions

                # Set up array to generate box plots
                if Parameters["dataSource"] == "fuchs":
                    boxData = np.array([Data["0"].reshape(-1, 1), Data["30"].reshape(-1, 1),
                                        Data["120"].reshape(-1, 1), Data["240"].reshape(-1, 1)], dtype=object)
                elif Parameters["dataSource"] == "leo":
                    boxData = np.array([Data["0"].reshape(-1, 1), Data["60"].reshape(-1, 1),
                                        Data["120"].reshape(-1, 1), Data["240"].reshape(-1, 1),
                                        Data["480"].reshape(-1, 1), Data["960"].reshape(-1, 1),
                                        Data["1440"].reshape(-1, 1)], dtype=object)

                # Plot ode solution against testing dataset
                if saveFigure == "yes":
                    fig, ax2 = plt.subplots()
                    ax2.plot(plotTimeCheck, sol, linewidth=3, color="grey", label="Equation learning prediction")
                    ax2.plot(plotTime, plotTest, marker='o', color="red", linestyle="", label="Test data")
                    ax2.boxplot(boxData, positions=timeValsHours)
                    ax2.set(xlabel="Time (h)", ylabel="Particles per cell")
                    ax2.set_xlim((0 - 0.3, maxTime + 0.3))
                    if dataSource == "fuchs":
                        ax2.set_ylim((-10, np.ceil(np.max(Data["240"]) + 10)))
                    elif dataSource == "leo":
                        ax2.set_ylim((-1, np.ceil(np.max(Data["1440"]) + 1)))
                    plt.savefig('./Results/%s_%s_%s_%d_ModelFit.eps' % (Parameters["dataSource"],
                                                                        Parameters["particleType"],
                                                                        Parameters["particleFunc"],
                                                                        Parameters["concentration"]))
                    plt.show()

                # Uncomment to plot comparison between neural network
                # prediction against test dataset

                # fig, ax3 = plt.subplots()
                # ax3.plot(plotTimeCheck, plotPredictions, linewidth=3, color="blue", label="Neural network prediction")
                # ax3.errorbar(plotTime, plotTest, yerr=errorbars, fmt='o', mec="red", mfc="red", color="red",
                #              label="Test data")
                # ax3.set(xlabel="Time (h)", ylabel="Particles per cell")
                # ax3.set_xlim((0 - 0.2, maxTime + 0.2))
                # ax3.set_ylim((0, np.ceil(maxOutput + 1)))
                # ax3.legend(loc="lower right")
                # plt.savefig('./Results/NeuralNetworkFit.eps')
                # plt.show()

                # Uncomment to plot evolution of training and validation loss functions

                # fig, ax4 = plt.subplots()
                # ax4.plot(outputModel.history["loss"][5:])
                # ax4.plot(outputModel.history["val_loss"][5:])
                # ax4.set(xlabel="Epoch", ylabel="Error")
                # plt.yscale("log")
                # plt.savefig('./Results/ValidationFit.eps')
                # plt.show()

