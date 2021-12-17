import numpy as np

# Code to process the experimental data for "Biologically-informed neural networks and equation
# # learning reveal nano-engineered particle-cell interactions" by Johnston and Faria.

def load_and_process(parameters, t):
    # Load data for fuchs data set
    if parameters["dataSource"] == "fuchs":
        # Load experimental data
        loadString = "Flow Data/%s_%s_%s_%d_%d.0m_data_r0.csv" % (parameters["dataSource"],
                                                                  parameters["particleType"],
                                                                  parameters["particleFunc"],
                                                                  parameters["concentration"],
                                                                  t)
        loadedData = np.genfromtxt(loadString, delimiter=",", usemask=True).data
        loadedData = loadedData[1:, :]
        dataSSC = loadedData[:, 3].reshape(-1, 1)                               # Data for side scatter channel
        dataPC = loadedData[:, 5].reshape(-1, 1)                                # Data for particle signal channel

        # Remove outlying data (i.e. outside upper quartile + 1.5x interquartile range / lower quartile - 1.5x interquartile range)
        lower = np.percentile(dataPC, 25)
        upper = np.percentile(dataPC, 75)
        iqr = upper - lower
        cutoffLower = max(lower - 1.5 * iqr, 0.0)
        cutoffUpper = upper + 1.5 * iqr
        keepIndices = (dataPC >= cutoffLower) & (dataPC <= cutoffUpper)
        dataPC = dataPC[keepIndices].reshape(-1, 1)
        dataSSC = dataSSC[keepIndices].reshape(-1, 1)

        # Load control data
        loadControlString = "Flow Data/%s_%s_%s_%d_0.0m_ctrl_r0.csv" % (parameters["dataSource"],
                                                                        parameters["particleType"],
                                                                        parameters["particleFunc"],
                                                                        parameters["concentration"])
        loadedControl = np.genfromtxt(loadControlString, delimiter=",", usemask=True).data
        loadedControl = loadedControl[1:, :]
        controlSSC = loadedControl[:, 3].reshape(-1, 1)                 # Data for side scatter channel
        controlPC = loadedControl[:, 5].reshape(-1, 1)                  # Data for particle signal channel

        # Calculate relationship between log(side scatter channel) and log(particle channel)
        logControlSSC = np.log(controlSSC[(controlSSC > 0.0) & (controlPC > 0.0)])
        logControlPC = np.log(controlPC[(controlSSC > 0.0) & (controlPC > 0.0)])
        fit_parameters = np.linalg.lstsq(np.array([logControlSSC, np.ones(logControlSSC.size)]).T, logControlPC, rcond=None)

        # Remove outlying data
        dataPC = dataPC[dataSSC > 0.0].reshape(-1, 1)
        dataSSC = dataSSC[dataSSC > 0.0].reshape(-1, 1)

        # Shift data based on SSC-PC relationship to remove signal from cells
        backgroundShift = np.exp(fit_parameters[0][0]*np.log(dataSSC) + fit_parameters[0][1])
        loadedData = dataPC - backgroundShift
        loadedData[loadedData < 0.0] = 0.0

        # Set zero time data to zero, applying appropriate weighting.
        if t == 0.0:
            loadedData = 0.0 * loadedData
            loadedData = np.repeat(loadedData, 1).reshape(-1, 1)

    # Load data for leo data set
    elif parameters["dataSource"] == "leo":
        # Load experimental data (data already converted to particles per cell)
        loadString = "Flow Data/%s_%s_%d.0m_particles_per_cell_r%d.csv" % (parameters["dataSource"],
                                                                           parameters["particleType"],
                                                                           t,
                                                                        parameters["concentration"])
        loadedData = np.genfromtxt(loadString, delimiter=",", usemask=True).data

        loadedData = loadedData[1:, 1]

        # Remove outlying data (i.e. outside upper quartile + 1.5x interquartile range / lower quartile - 1.5x interquartile range)
        lower = np.percentile(loadedData, 25)
        upper = np.percentile(loadedData, 75)
        iqr = upper - lower
        cutoffLower = max(lower - 1.5 * iqr, 0.0)
        cutoffUpper = upper + 1.5 * iqr
        keepIndices = (loadedData >= cutoffLower) & (loadedData <= cutoffUpper)
        loadedData = loadedData[keepIndices].reshape(-1, 1)

        # Set zero time data to zero, applying appropriate weighting.
        if t == 0.0:
            loadedData = 0.0 * loadedData
            loadedData = np.repeat(loadedData, 5).reshape(-1, 1)

    return loadedData
