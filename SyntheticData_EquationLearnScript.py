import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from scipy.integrate import odeint
from scipy.optimize import lsq_linear
import numpy as np
import matplotlib.pyplot as plt

# Code to run the equation learning framework in "Biologically-informed neural networks and equation
# learning reveal nano-engineered particle-cell interactions" by Johnston and Faria for synthetic data.
# Example data is provided for an association rate that is a cubic function in "ExampleSyntheticData.csv"

# Script parameters

nEpochs = 2000                                                       # Number of epochs in the neural network
nNodes = 128                                                        # Number of nodes in each hidden layer of the neural network
trainDataFraction = 0.6                                             # Fraction of data used for training
validateDataFraction = 0.3                                          # Fraction of data used for validation
testDataFraction = 1.0 - trainDataFraction - validateDataFraction   # Fraction of data used for testing
maxTime = 24.0                                                      # Final time in data (hours)
repeats = 2500                                                      # Number of data points per time step
nTimePoints = 7                                                     # Number of time steps in data
y_terms_max = 10                                                    # Maximum order of associated nanoparticles in equation learning
nReplicates = 1                                                     # Number of realisations of the equation learning process
replicates = range(0, nReplicates)                                  # Replicate numbers
nTimesUsed = np.zeros(y_terms_max+1)                                # Array of counts of the times each model component is used
modelCounter = np.zeros(2**(y_terms_max+1))                         # Array of counts of the number each potential model is learned
storeModel = np.zeros(nReplicates)                                  # Store the model number for each realisation

# Define function to prune unnecessary model terms
def pruneModel(rhsTermsIn, dydtIn, firstError, boundsIn):
    """
    :param rhsTermsIn: Equation learning model components
    :param dydtIn: Data to be constructed from model components
    :param firstError: Error of unpruned model
    :param boundsIn: Bounds on parameters
    :return: Coefficients of pruned model
    """

    outputCoef = np.ones(rhsTermsIn.shape[1])                       # Preallocate output coefficients
    indicesStore = range(0, rhsTermsIn.shape[1])                    # Potential indices
    for ii in range(rhsTermsIn.shape[1] - 1, -1, -1):               # Loop over indices (backwards)
        indices = np.delete(indicesStore, ii)                       # Remove iith index
        proposedError = lsq_linear(rhsTermsIn[:, indices], dydtIn,  # Calculate error for reduced model
                                   bounds=(boundsIn[indices, 0],
                                           boundsIn[indices, 1])).cost
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

# Loop over each replicate
for iReplicate in replicates:
    print(iReplicate)

    # Load and process data
    dataString = "ExampleSyntheticData.csv"                             # String name for csv file with data
    readData = np.genfromtxt(dataString, delimiter=",", usemask=True).data    # Load synthetic data in (N_i, t_i) format
    timeData = readData[:, 1].reshape(-1, 1)                            # Time data (neural network input)
    npData = readData[:, 0].reshape(-1, 1)                              # Number of associated nanoparticles (neural network output)
    timeValues = np.unique(timeData)                                    # Time values in data
    meanNPValues = np.mean(npData.reshape(nTimePoints,
                                          repeats, order='F'), axis=1)  # Mean number of nanoparticles at each point

    originalMeanData = meanNPValues                                     # Store mean nanoparticle data
    originalTime = timeValues                                           # Store original time data

    # Rescale time to ensure non-negative 1st and 2nd derivative
    dataOut = npData                                                    # Set neural network output data
    dataOut[timeData == 0.0] = 0.0                                      # Zero data for control condition
    dataIn = maxTime - timeData                                         # Rescale time data for constraints
    scalerMethod = StandardScaler().fit(dataIn)                         # Standardise input data to zero mean
    dataIn = scalerMethod.transform(dataIn)                             # Standardise input data to zero mean

    maxData = np.max(dataOut)                                           # Maximum output value

    uniqueTime = np.unique(dataIn)                                      # Unique values in transformed time

    # Rescale data to ensure positive 1st and 2nd derivative
    maxOutput = np.max(originalMeanData)                                # Maximum mean value
    dataOut = 1.0 - dataOut / maxOutput                                 # Scale neural network output for constraints

    nData = dataIn.size                                                 # Number of observations

    # Split into training, validation and test datasets
    dataIndices = np.linspace(0, nData - 1, nData, dtype=int)
    trainIndices = np.random.choice(dataIndices, int(np.ceil(nData * trainDataFraction)), replace=False)
    unusedIndices = np.setdiff1d(dataIndices, trainIndices)
    validateIndices = np.random.choice(unusedIndices, int(np.ceil(nData * validateDataFraction)), replace=False)
    testIndices = np.setdiff1d(unusedIndices, validateIndices)

    # Define training dataset
    trainIn = dataIn[trainIndices]
    trainOut = dataOut[trainIndices]
    meanTrain = np.zeros(uniqueTime.size)
    for i in range(0, uniqueTime.size):                                 # Calculate mean of training dataset
        meanTrain[i] = np.mean(trainOut[trainIn == uniqueTime[-(i + 1)]])

    # Define validation dataset
    validateIn = dataIn[validateIndices]
    validateOut = dataOut[validateIndices]
    meanValidate = np.zeros(uniqueTime.size)
    for i in range(0, uniqueTime.size):                                 # Calculate mean of validation dataset
        meanValidate[i] = np.mean(validateOut[validateIn == uniqueTime[-(i + 1)]])

    # Define testing dataset
    testIn = dataIn[testIndices]
    testOut = dataOut[testIndices]
    meanTest = np.zeros(uniqueTime.size)
    for i in range(0, uniqueTime.size):                                 # Calculate mean of testing dataset
        meanTest[i] = np.mean(testOut[testIn == uniqueTime[-(i + 1)]])

    timeCheck = np.linspace(np.min(dataIn), np.max(dataIn), 100).reshape(-1, 1)     # Time values for solution
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

    loss_fn = tf.keras.losses.MeanSquaredError()                                        # Loss function
    model.compile(optimizer='adam', loss=loss_fn)                                       # Model details

    # Fit neural network model to training data and select best via validation
    outputModel = model.fit(trainIn,
                            trainOut,
                            epochs=nEpochs,
                            validation_data=(validateIn, validateOut),
                            callbacks=[model_checkpoint_callback])

    bestModel = tf.keras.models.load_model(checkpoint_filepath)                         # Load best model
    predictions = np.flip(1 - bestModel(timeCheck))                                     # Model predictions

    t = list(timeCheck)                                                                  # Transform to tf data
    t = tf.constant(t)

    # Calculate derivative of neural network model with respect to time
    with tf.GradientTape() as tape:
        tape.watch(t)
        y = bestModel(t)

    dydt = tape.gradient(y, t).numpy()                                                  # Derivatives at ODE time
    y = np.flip(1 - np.ravel(bestModel(t).numpy()))                                     # Model predictions at ODE time

    total_model_components = y_terms_max                                                # Total terms in potential model

    # Generate model components
    rhsTerms = np.zeros([len(y), total_model_components])
    term_counter = 0
    lower_bounds = np.zeros(total_model_components)                                     # Lower bounds for parameters
    upper_bounds = np.zeros(total_model_components)                                     # Upper bounds for parameters

    # Define RHS terms coresponding to each model component
    for iy in range(0, y_terms_max):
        rhsTerms[:, iy] = y ** (iy+1)

    bounds = np.vstack([lower_bounds, upper_bounds]).transpose()                        # Bounds for parameters

    shifted_dydt = -(np.flip(np.ravel(dydt)) - np.max(np.ravel(dydt)))              # Reorient derivative
    sparse_Lasso = LassoCV(fit_intercept=False, max_iter=10000, positive=True)      # Cross validated sparse lasso fit
    bounds[:, 0] = 0.0                                                              # Enforce positive constraints
    bounds[:, 1] = np.inf                                                           # Enforce positive constraints

    sparse_Lasso.fit(rhsTerms, shifted_dydt)                                        # Fit model components to derivative via LASSO

    nonzeroIndices = np.nonzero(sparse_Lasso.coef_)[0]                                  # Non-zero model component coefficients

    # Perform model pruning for each non-zero model component (provided more than one model component)
    if nonzeroIndices.size > 1:
        startError = lsq_linear(rhsTerms[:, nonzeroIndices], shifted_dydt,              # Initial error in model
                                bounds=(bounds[nonzeroIndices, 0], bounds[nonzeroIndices, 1])).cost
        relevantIndices = pruneModel(rhsTerms[:, nonzeroIndices], shifted_dydt,
                                     startError, bounds[nonzeroIndices, :])             # Prune model terms
        coefs = lsq_linear(rhsTerms[:, nonzeroIndices[relevantIndices]], shifted_dydt,  # Calculate new model coefficients
                           bounds=(bounds[nonzeroIndices[relevantIndices], 0],
                                   bounds[nonzeroIndices[relevantIndices], 1])).x
        coefStorage = np.zeros(rhsTerms.shape[1])
        coefStorage[nonzeroIndices[relevantIndices]] = coefs
        coefs = coefStorage                                                             # New model coefficients
    else:
        coefs = lsq_linear(rhsTerms[:, nonzeroIndices], shifted_dydt,                   # Else just calculate coefficients
                           bounds=(bounds[nonzeroIndices, 0],
                                   bounds[nonzeroIndices, 1])).x
        coefStorage = np.zeros(rhsTerms.shape[1])
        coefStorage[nonzeroIndices] = coefs
        coefs = coefStorage                                                             # Calculate new model coefficients

    intercept = 0.0                                                                     # Intercept in model

    # Re-transform parameters
    coefs = -1.0*coefs
    intercept = np.max(np.ravel(dydt))

    # Plot neural network derivative and sparse model approximation
    fig, ax1 = plt.subplots()
    ax1.plot(y, np.flip(dydt), linewidth=3, label="Neural network derivative")
    ax1.plot(y, np.inner(coefs, rhsTerms) + intercept, linestyle='--', linewidth=3, label="LASSO prediction")
    ax1.set(xlabel="Particles per cell", ylabel="Particles per cell per time")
    ax1.legend(loc="upper right")
    plt.show()

    # Scale model components by maximum output (i.e. undo initial scaling)
    for iy in range(0, y_terms_max):
        rhsTerms[:, iy] *= maxOutput**(iy+1)

    y0 = min(y)                                                                 # Initial condition for dynamical systems model
    time_ratio = maxTime/(np.ravel(timeCheck)[-1] - np.ravel(timeCheck)[0])     # Ratio of time in transformed and untransformed models
    ode_time = scalerMethod.inverse_transform(np.ravel(timeCheck))              # Define time range for dynamical systems model

    # Scale coefficients and initial condition to be consistent with input data
    for im in range(0, total_model_components):
        coefs[im] /= maxOutput**im

    intercept *= maxOutput/time_ratio
    coefs /= time_ratio
    y0 *= maxOutput

    # Solve differential equation model
    sol = odeint(odesystem, y0, ode_time, args=(coefs, intercept))

    # Plot comparison between LASSO prediction and neural network prediction of the association rate
    fig, ax1 = plt.subplots()
    ax1.plot(y*maxOutput, np.flip(dydt)/time_ratio*maxOutput, linewidth=3, label="Neural network derivative")
    ax1.plot(y*maxOutput, np.inner(coefs, rhsTerms) + intercept, linestyle='--', linewidth=3, label="LASSO prediction")
    ax1.set(xlabel="Particles per cell", ylabel="Particles per cell per time")
    ax1.legend(loc="upper right")
    ax1.set_xlim((0, maxOutput))
    plt.savefig('./Results/DerivativeFit.eps')
    plt.show()

    plotTime = np.flip(scalerMethod.inverse_transform(timeData))                        # Recover data plotting time
    plotTimeCheck = scalerMethod.inverse_transform(timeCheck)                           # Recover model plotting time
    plotTrain = (1.0 - meanTrain) * maxOutput                                           # Rescale training data
    plotValidate = (1.0 - meanValidate) * maxOutput                                     # Rescale validation data
    plotTest = (1.0 - meanTest) * maxOutput                                             # Rescale testing data
    plotPredictions = maxOutput * predictions                                           # Rescale predictions

    # Calculate error bars from synthetic data
    errorbars = np.zeros(timeValues.size)
    for i in range(0, timeValues.size):
        errorbars[i] = np.std(npData[readData[:, 1].reshape(-1, 1) == timeValues[i]])

    # Plot ODE solution against test dataset
    fig, ax2 = plt.subplots()
    ax2.plot(plotTimeCheck, sol, linewidth=3, color="grey", label="Equation learning prediction")
    ax2.errorbar(plotTime, plotTest, yerr=errorbars, fmt='o', mec="red", mfc="red", color="red",
                 label="Test data")
    ax2.set(xlabel="Time (h)", ylabel="Particles per cell")
    ax2.set_xlim((0-0.2, maxTime+0.2))
    ax2.set_ylim((0, np.ceil(maxOutput+3)))
    ax2.legend(loc="lower right")
    plt.savefig('./Results/LearnedModelFit.eps')
    plt.show()

    # Plot neural network prediction against test dataset
    fig, ax3 = plt.subplots()
    ax3.plot(plotTimeCheck, plotPredictions, linewidth=3, color="blue", label="Neural network prediction")
    ax3.errorbar(plotTime, plotTest, yerr=errorbars, fmt='o', mec="red", mfc="red", color="red",
                 label="Test data")
    ax3.set(xlabel="Time (h)", ylabel="Particles per cell")
    ax3.set_xlim((0-0.2, maxTime+0.2))
    ax3.set_ylim((0, np.ceil(maxOutput+3)))
    ax3.legend(loc="lower right")
    plt.savefig('./Results/NeuralNetworkFit.eps')
    plt.show()

    # Plot evolution of training and validation loss functions
    fig, ax4 = plt.subplots()
    ax4.plot(outputModel.history["loss"][5:])
    ax4.plot(outputModel.history["val_loss"][5:])
    ax4.set(xlabel="Epoch", ylabel="Error")
    plt.yscale("log")
    plt.savefig('./Results/ValidationFit.eps')
    plt.show()

    # Store which model components are used
    modelNum = 0
    for i in range(0, y_terms_max):
        if coefs[i] != 0:
            nTimesUsed[i + 1] += 1

    # Store if 0th order term is used
    if intercept != 0:
        nTimesUsed[0] += 1

    # Calculate which model number is used (number = binary representation, 1 if model component used, 0 otherwise)
    for i in range(0, y_terms_max):
        if coefs[i] != 0:
            modelNum += 2 ** (i + 1)

    if intercept != 0:
        modelNum += 1

    modelCounter[modelNum] += 1
    storeModel[iReplicate] = modelNum

    # Print the form of the learned model
    tc = 0
    learnedEquation = "dN/dt = "
    learnedEquation += "+ %f " % (intercept)
    for iy in range(0, y_terms_max):
        if coefs[iy] != 0:
            learnedEquation += "+ %f N^%d " % (coefs[iy], iy+1)

    print(learnedEquation)
