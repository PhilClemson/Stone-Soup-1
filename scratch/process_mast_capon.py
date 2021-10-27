# -*- coding: utf-8 -*-
import numpy as np
import math
from os import path
from datetime import datetime
import cProfile as profile
from matplotlib.patches import Ellipse

from stonesoup.detector import beamformers
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.array import CovarianceMatrix
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track

from stonesoup.initiator.simple import MultiMeasurementInitiator, SimpleMeasurementInitiator
from stonesoup.tracker.simple import MultiTargetTracker
from stonesoup.deleter.error import CovarianceBasedDeleter
from stonesoup.deleter.time import UpdateTimeStepsDeleter
from stonesoup.types.state import GaussianState
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
import matplotlib.pyplot as plt
from stonesoup.plotter import Plotter

pr = profile.Profile()
pr.disable()

def plot_cov_ellipse(cov, pos, nstd=1, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.
    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta,
                    alpha=0.4, **kwargs)

    ax.add_artist(ellip)
    return ellip


def plot_tracks(tracks, show_error=True, ax=None, color='r', label='Tracks'):
    if not ax:
        ax = plt.gca()
    ax.plot([], [], '-', color=color, label=label)
    for track in tracks:
        data = np.array([state.state_vector for state in track.states])
        ax.plot(data[:, 0], data[:, 2], '-', color=color)
        if show_error:
            plot_cov_ellipse(track.state.covar[[0, 2], :][:, [0, 2]],
                             track.state.mean[[0, 2], :], edgecolor=color,
                             facecolor='none', ax=ax)


def plot_detections(detections, ax=None, label='Detections'):
    if not ax:
        ax = plt.gca()
    data = np.array([detection.state_vector for detection in detections])
    ax.plot(data[:, 0], data[:, 1], 'bx', label=label)



if __name__ == '__main__':
    detections = set()
    tracks = set()
    fig = plt.figure()

    # import data files (converted from .mat to .csv using Matlab "writematrix()" function)
    data_file = "mast_1.csv"
    xpos_file = "mast_1_X.csv"
    ypos_file = "mast_1_Y.csv"
    zpos_file = "mast_1_Z.csv"

    # store positions over time in sensor_pos array
    X = np.loadtxt(xpos_file, delimiter=',')
    Y = np.loadtxt(ypos_file, delimiter=',')
    Z = np.loadtxt(zpos_file, delimiter=',')
    sensor_pos = np.empty([X.shape[1], 3, X.shape[0]])
    for i in range(0, X.shape[0]):
        for j in range(0, X.shape[1]):
            sensor_pos[j, 0, i] = X[i, j]
            sensor_pos[j, 1, i] = Y[i, j]
            sensor_pos[j, 2, i] = Z[i, j]

    # specify detectors
    window = 750  # size of sliding window in samples
    detector = beamformers.CaponBeamformer(data_file, sensor_loc=sensor_pos, fs=750, omega=115, wave_speed=1481)

    # initialise tracker
    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.1), ConstantVelocity(0.1)])

    # initialise the measurement model
    measurement_model_covariance = CovarianceMatrix([[0.25, 0.25]])
    measurement_model = LinearGaussian(4, [0, 2], measurement_model_covariance)

    # Predictor Updater
    predictor = KalmanPredictor(transition_model)
    updater = KalmanUpdater(measurement_model)

    # Hypothesiser/Data Associator
    hypothesiser = DistanceHypothesiser(predictor, updater, measure=Mahalanobis(), missed_distance=1)
    data_associator = GNNWith2DAssignment(hypothesiser)

    # Initiator
    s_prior_state = GaussianState([[0], [0], [0], [0]], np.diag([5, 0.5, 5, 0.5]))
    initiator = SimpleMeasurementInitiator(s_prior_state, measurement_model)

    # Deleter
    # covariance_limit_for_delete = 10
    # deleter = CovarianceBasedDeleter(covar_trace_thresh=covariance_limit_for_delete)
    deleter = UpdateTimeStepsDeleter(5)

    tracker = MultiTargetTracker(
        initiator=initiator,
        deleter=deleter,
        detector=detector,
        data_associator=data_associator,
        updater=updater,
    )

    tracks = set()
    for time, ctracks in tracker:
        tracks.update(ctracks)

        detections = detector.current[1]

        # Plot
        plt.clf()
        plot_tracks(tracks-ctracks, color='grey', label='Deleted Tracks')  # Plot the deleted tracks
        plot_tracks(ctracks)  # Plot the current tracks
        plot_detections(detections)  # Plot the detections
        ax = plt.gca()
        ax.set_xlim((-10, 10))
        ax.set_ylim((-10, 10))
        plt.legend()
        plt.pause(0.01)


