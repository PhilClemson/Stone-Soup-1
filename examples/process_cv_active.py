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
    #data_file="lcas_sim.csv"
    #pulse_file="lcas_pulse.csv"
    #data_file="tenPulses_LFM.csv"
    #pulse_file="tenPulses_LFM_pulse.csv"
    
    num_sensors = 9
    
    num_windows = 4

    # define static sensor positions
    #X = [0, 0.210, 0.420, 0.630, 0.840, 1.05, 1.26, 1.47, 1.68, 1.89, 2.10, 2.31, 2.52, 2.73, 2.94, 3.15, 3.36, 3.57, 3.78, 3.99, 4.20, 4.41, 4.62, 4.83, 5.04, 5.25, 5.46, 5.67, 5.88, 6.09, 6.30, 6.51]
    #Y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #Z = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #X = [0, -0.25, -0.5, -0.75, -1, -1.25, -1.5, -1.75, -2, -2.25, -2.5, -2.75, -3, -3.25, -3.5, -3.75, -4, -4.25, -4.5, -4.75, -5, -5.25, -5.5, -5.75, -6, -6.25, -6.5, -6.75, -7, -7.25, -7.5, -7.75, -8, -8.25, -8.5, -8.75, -9, -9.25, -9.5, -9.75, -10, -10.25, -10.5, -10.75, -11, -11.25, -11.5, -11.75, -12, -12.25, -12.5, -12.75, -13, -13.25, -13.5, -13.75, -14, -14.25, -14.5, -14.75, -15, -15.25, -15.5, -15.75, -16, -16.25, -16.5, -16.75, -17, -17.25, -17.5, -17.75, -18, -18.25, -18.5, -18.75, -19, -19.25, -19.5, -19.75, -20, -20.25, -20.5, -20.75, -21, -21.25, -21.5, -21.75, -22, -22.25, -22.5, -22.75, -23, -23.25, -23.5, -23.75, -24, -24.25, -24.5, -24.75]
    #Y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #Z = [-80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80]
    #X = [0, -0.25, -0.5, -0.75, -1, -1.25, -1.5, -1.75, -2, -2.25, -2.5, -2.75, -3, -3.25, -3.5, -3.75, -4, -4.25, -4.5, -4.75, -5, -5.25, -5.5, -5.75, -6, -6.25, -6.5, -6.75, -7, -7.25, -7.5, -7.75, -8, -8.25, -8.5, -8.75, -9, -9.25, -9.5, -9.75, -10, -10.25, -10.5, -10.75, -11, -11.25, -11.5, -11.75, -12, -12.25]
    #Y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #Z = [-80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80]
    X = [0, 50, 100, 0, 50, 100, 0, 50, 100]
    Y = [0, 0, 0, 50, 50, 50, 100, 100, 100]
    Z = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    sensor_pos_sequence = []
    for i in range(0, num_windows):
        sensor_pos_xyz = []
        for j in range(0, num_sensors):
            sensor_pos_xyz.append(X[j])
            sensor_pos_xyz.append(Y[j])
            sensor_pos_xyz.append(Z[j])
        sensor_pos_sequence.append(StateVector(sensor_pos_xyz))
    sensor_pos: Sequence[StateVector] = sensor_pos_sequence
    
    # define static source positions
    X=[0]
    Y=[0]
    Z=[-80]
    source_pos_sequence = []
    for i in range(0, num_windows):
        source_pos_xyz = [X, Y, Z]
        source_pos_sequence.append(StateVector(source_pos_xyz))
    source_pos: Sequence[StateVector] = source_pos_sequence
    
    # specify detectors
    #window = 165000  # size of sliding window in samples     
    window = 5000
    #detector = beamformers.ActiveBeamformer(data_file, pulse_file, sensor_loc=sensor_pos, fs=2000,
    #                                        wave_speed=1481, max_vel=100, window_size=window,
    #                                        nbins=[100, 100, 100, 1])
    #detector = beamformers.ActiveBeamformer(data_file, pulse_file, sensor_loc=sensor_pos, fs=8187,
    #                                        wave_speed=1481, max_vel=100, window_size=window,
    #                                        nbins=[10, 1, 100, 1])
    detector = beamformers.ActiveBeamformer(data_file, pulse_file, sensor_loc=sensor_pos, fs=15000,
                                            source_loc = source_pos, wave_speed=1510, max_vel=50, window_size=window,
                                            nbins=[100, 1, 100, 101])
    
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
