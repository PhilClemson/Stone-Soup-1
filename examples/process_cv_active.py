# -*- coding: utf-8 -*-
import numpy as np
import math
from os import path
from datetime import datetime
import cProfile as profile
from matplotlib.patches import Ellipse
from typing import Sequence

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
from stonesoup.types.array import StateVector

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

    # import data files
    data_file="active_square_noisy3.csv"
    pulse_file="pulse2.csv"
    
    num_sensors = 9

    # define static sensor positions
    X = [0, 50, 100, 0, 50, 100, 0, 50, 100]
    Y = [0, 0, 0, 50, 50, 50, 100, 100, 100]
    Z = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    sensor_pos_sequence = []
    for i in range(0, 1):
        sensor_pos_xyz = []
        for j in range(0, num_sensors):
            sensor_pos_xyz.append(X[j])
            sensor_pos_xyz.append(Y[j])
            sensor_pos_xyz.append(Z[j])
        sensor_pos_sequence.append(StateVector(sensor_pos_xyz))
    sensor_pos: Sequence[StateVector] = sensor_pos_sequence
    
    # specify detectors
    window = 5000  # size of sliding window in samples     
    detector = beamformers.ActiveBeamformer(data_file, pulse_file, sensor_loc=sensor_pos, fs=2000,
                                            wave_speed=1481, max_vel=100, window_size=window,
                                            nbins=[100, 100, 100, 1])
    
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
        plt.pause(0.1)
