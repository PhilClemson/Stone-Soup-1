# -*- coding: utf-8 -*-
# !python
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


# pr.disable()

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


def plot_projected_tracks(tracks, coord=0, nstd=1, show_error=True, ax=None, color='r', label='Tracks'):
    if not ax:
        ax = plt.gca()
    ax.plot([], [], '-', color=color, label=label)
    for track in tracks:
        means = np.array([state.state_vector[coord] for state in track.states])
        times = np.array([state.timestamp for state in track.states])
        ax.plot(times, means, '-', color=color)
        if show_error:
            stdevs = np.array([nstd * np.sqrt(state.covar[coord, coord]) for state in track.states])
            ax.plot(times, means + stdevs, '--', color=color)
            ax.plot(times, means - stdevs, '--', color=color)

def plot_projected_detections(detections, coord=0, ax=None, color='blue', label='Detection'):
    for detection in detections:
        data = measurement_model.inverse_function(detection)
        ax.plot(detection.timestamp, data[coord], marker='.', color=color, label=label)


def plot_projected_gt(gt, times, coord=0, ax=None, label='Target'):
    data = []
    for _ in times:
        data.append(gt[coord])
    ax.plot(times, data, 'gx-', label=label)

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


def plot_detections(detections, ax=None, color='blue', label='Detections'):
    if not ax:
        ax = plt.gca()
    if type(measurement_model) == CartesianToElevationBearing:
        # Measurement model is not invertible (range info is missing).
        data = [detection.state_vector for detection in detections]
        x_s, y_s, z_s = measurement_model.translation_offset
        for direction in data:
            elevation, bearing = direction
            r = 10000  # "infinite range"
            x = [x_s, r * np.cos(bearing)]
            y = [y_s, r * np.sin(bearing)]
            ax.plot(x, y, 'b-', label='Bearings')
    else:
        # Measurement model is invertible (range info is present to recover the location).
        data = np.array([measurement_model.inverse_function(detection) for detection in detections])
        ax.plot(data[:, 0], data[:, 2], linestyle="", marker=".", color=color, label=label)


if __name__ == '__main__':
    import copy
    import csv
    from scipy.io import savemat
    from datetime import timedelta
    from stonesoup.models.measurement.nonlinear import CartesianToElevationBearing
    from stonesoup.models.measurement.nonlinear import CartesianToElevationBearingRange
    from stonesoup.models.measurement.nonlinear import CartesianToElevationBearingRangeRate
    from stonesoup.simulator.simple import MultiTargetGroundTruthSimulator
    from stonesoup.simulator.simple import SimpleDetectionSimulator
    from stonesoup.initiator.simple import MultiMeasurementInitiator
    from stonesoup.plotter import Plotter
    from stonesoup.reader.generic import CSVDetectionReader
    from stonesoup.predictor.kalman import UnscentedKalmanPredictor
    from stonesoup.updater.kalman import UnscentedKalmanUpdater
    from stonesoup.predictor.particle import ParticlePredictor
    from stonesoup.resampler.particle import SystematicResampler
    from stonesoup.updater.particle import ParticleUpdater
    from stonesoup.initiator.simple import GaussianParticleInitiator
    from stonesoup.initiator.simple import SinglePointInitiator

    seed = 2022  # np.random.seed(seed)
    np.random.seed(seed)

    """Input data"""
    data_file = "tenPulses_LFM.csv"
    pulse_file = "tenPulses_LFM_pulse.csv"

    num_sensors = 100

    num_windows = 10

    # define static sensor positions
    X = [0, -0.25, -0.5, -0.75, -1, -1.25, -1.5, -1.75, -2, -2.25, -2.5, -2.75, -3, -3.25, -3.5, -3.75, -4, -4.25, -4.5,
         -4.75, -5, -5.25, -5.5, -5.75, -6, -6.25, -6.5, -6.75, -7, -7.25, -7.5, -7.75, -8, -8.25, -8.5, -8.75, -9,
         -9.25, -9.5, -9.75, -10, -10.25, -10.5, -10.75, -11, -11.25, -11.5, -11.75, -12, -12.25, -12.5, -12.75, -13,
         -13.25, -13.5, -13.75, -14, -14.25, -14.5, -14.75, -15, -15.25, -15.5, -15.75, -16, -16.25, -16.5, -16.75, -17,
         -17.25, -17.5, -17.75, -18, -18.25, -18.5, -18.75, -19, -19.25, -19.5, -19.75, -20, -20.25, -20.5, -20.75, -21,
         -21.25, -21.5, -21.75, -22, -22.25, -22.5, -22.75, -23, -23.25, -23.5, -23.75, -24, -24.25, -24.5, -24.75]
    Y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Z = [-80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80,
         -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80,
         -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80,
         -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80,
         -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80]
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
    X = [0]
    Y = [0]
    Z = [-80]
    source_pos_sequence = []
    for i in range(0, num_windows):
        source_pos_xyz = [X, Y, Z]
        source_pos_sequence.append(StateVector(source_pos_xyz))
    source_pos: Sequence[StateVector] = source_pos_sequence

    window = 165000  # size of sliding window in samples

    """Pass data and parameters to detector"""
    detector = beamformers.ActiveBeamformer(data_file, pulse_file, sensor_loc=sensor_pos, fs=15000,
                                            source_loc=source_pos, wave_speed=1510, max_vel=50,
                                            window_size=window, nbins=[100, 1, 1000, 1])

    """Ground truth: model setup"""
    # Set a constant velocity transition model for the targets
    transition_model = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(0.1), ConstantVelocity(0.1), ConstantVelocity(0.1)], seed=seed)

    # Define the Gaussian State from which new targets are sampled on initialisation
    x_0, y_0, z_0 = 2500, 2500, 0
    sigma_x, sigma_y, sigma_z = 10, 10, 0.1
    sigma_x_dot, sigma_y_dot, sigma_z_dot = 5, 5, 1

    initial_state_mean = StateVector([[x_0], [0], [y_0], [0], [z_0], [0]])
    initial_state_cov = CovarianceMatrix(np.diag([sigma_x ** 2, sigma_x_dot ** 2,
                                                  sigma_y ** 2, sigma_y_dot ** 2,
                                                  sigma_z ** 2, sigma_z_dot ** 2]))
    timestamp_initial = datetime.now()
    initial_target_state = GaussianState(
        initial_state_mean,
        initial_state_cov,
        timestamp_initial
    )

    """Measurements: model setup"""
    # Setting up a comprehensive measurement model (here: CartesianToElevationBearingRangeRate)
    noise_covar = CovarianceMatrix(np.array(np.diag([
        np.deg2rad(1) ** 2,  # elevation_var
        np.deg2rad(0.15) ** 2,  # bearing_var
        25 ** 2,  # range_var
        1 ** 2  # rangerate_var
    ])))
    rotation_offset = StateVector([
        [np.deg2rad(0)],
        [np.deg2rad(0)],
        [np.deg2rad(-90)]
    ])  # orientation of the array's boresight (oriented along x axis already)
    translation_offset = StateVector([
        [-24.75 / 2],
        [0],
        [-80]
    ])  #
    model_args_generation = {
        'ndim_state': 6,
        'mapping': np.array([0, 2, 4]),
        'velocity_mapping': np.array([1, 3, 5]),
        'noise_covar': noise_covar,
        'rotation_offset': rotation_offset,
        'translation_offset': translation_offset}  # centre of the array

    """Setting up a tracking algorithm"""
    # Announce candidate measurement models
    model_class_dict = {
        0: CartesianToElevationBearingRangeRate,  # measures range as well as rangerate
        1: CartesianToElevationBearingRange,
        2: CartesianToElevationBearing}

    # Ask for the number of a specific measurement model
    model_num = None
    # while model_num not in {0, 1, 2}:
    #     model_num = int(input(
    #         "Select the measurement model class: \n 0 - for {}, \n 1 - for {}, \n 2 - for {}.\n".format(
    #             model_class_dict[0], model_class_dict[1], model_class_dict[2])))
    model_num = 0

    # Assemble the measurement model from its class and (a part of) arguments used in generation
    model_args = copy.deepcopy(model_args_generation)
    if model_num in {1, 2}:
        del model_args['velocity_mapping']  # 'velocity_mapping' only required for model_num=0, and dropped elsewhere
        ndim_meas = np.shape(model_args['noise_covar'])[0] - model_num  # max. dimension to preserve
        model_args['noise_covar'] = model_args['noise_covar'][0:ndim_meas, 0:ndim_meas]  # trim covariance matrix
    measurement_model = model_class_dict[model_num](**model_args)

    # Choose filtering solution (UKF vs PF)
    use_ukf = None
    # while use_ukf not in {True, False}:
    #     use_ukf = bool(input("Use unscented Kalman filtering (and not particle filtering)? (0 - no, 1 - yes)\n"))
    #     if model_num == 2 and use_ukf == True:
    #         use_ukf = False
    #         print('Kalman filtering not possible with this measurement model. \nUsing particle filtering instead.')
    use_ukf = True

    # Predictor/Updater
    transition_model = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(0.1), ConstantVelocity(0.1), ConstantVelocity(0.1)])
    if use_ukf:
        predictor = UnscentedKalmanPredictor(transition_model)
        updater = UnscentedKalmanUpdater(measurement_model=measurement_model)
    else:
        predictor = ParticlePredictor(transition_model)
        updater = ParticleUpdater(measurement_model, resampler=SystematicResampler())

    # Hypothesiser/Data Associator
    missed_distance = 5
    hypothesiser = DistanceHypothesiser(predictor, updater, measure=Mahalanobis(), missed_distance=missed_distance)
    data_associator = GNNWith2DAssignment(hypothesiser)

    # Deleter
    # covariance_limit_for_delete = 10
    # deleter = CovarianceBasedDeleter(covar_trace_thresh=covariance_limit_for_delete)
    deleter = UpdateTimeStepsDeleter(5)

    # Initiator
    if use_ukf:
        initiator = SimpleMeasurementInitiator(prior_state=initial_target_state, measurement_model=measurement_model)
    else:
        number_particles = 1000
        single_point_initiator = SinglePointInitiator(
            prior_state=GaussianState(initial_target_state.state_vector.squeeze(), initial_target_state.covar),
            measurement_model=measurement_model)
        initiator = GaussianParticleInitiator(
            initiator=single_point_initiator,
            number_particles=number_particles,
            use_fixed_covar=False)

    min_points = 3
    initiator = MultiMeasurementInitiator(
        prior_state=initial_target_state,
        measurement_model=measurement_model,
        deleter=deleter,
        data_associator=data_associator,
        updater=updater,
        min_points=min_points,
        initiator=initiator)

    tracker = MultiTargetTracker(
        initiator=initiator,
        deleter=deleter,
        detector=detector,
        data_associator=data_associator,
        updater=updater)

    # Tracking loop
    tracks = set()
    detections_aggregated = set()
    alive = []
    gt = [1500, 0, 1500, 0]
    color = iter(plt.cm.rainbow(np.linspace(0, 1, 9)))

    for time, ctracks in tracker:
        tracks.update(ctracks)
        detections = detector.current[1]
        detections_aggregated.update(detections)
        alive.append(time)
        # Plot
        # plt.clf()
        plt.close('all')
        fig1, ax = plt.subplots()
        ax.plot(gt[0], gt[2], 'xg', label='Target')  # visualizing target
        plot_detections(detections, color=next(color))  # Plot the detections
        plot_tracks(tracks - ctracks, color='grey', label='Deleted Tracks')  # Plot the deleted tracks
        plot_tracks(ctracks)  # Plot the current tracks

        ax.set_xlabel('x-coordinate, [m]')
        ax.set_ylabel('y-coordinate, [m]')
        ax.set_xlim((1169.5908871554304, 1648.917306635766))
        ax.set_ylim((644.0407050947242, 1540.7599664240608))

        # ax.plot([-24.75, 0], [0, 0], '-g', label='Sonar Array')  # visualizing array extension/orientation

        # Introduce plot legend with no repreated entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        fig2 = plt.figure()
        ax1 = fig2.add_subplot(211)
        ax2 = fig2.add_subplot(212)

        ax1.set_ylabel('x-coordinate, [m]')
        ax1.tick_params(labelbottom=False)
        starttime = alive[0]
        endtime = alive[0]+timedelta(seconds=88)
        ax1.set_xlim((starttime, endtime))
        ax1.set_ylim((1168.9575004351245, 1648.9474679081613))

        ax2.set_ylabel('y-coordinate, [m]')
        ax2.tick_params(labelbottom=False)
        ax2.set_xlabel('Time')
        ax2.set_xlim((starttime, endtime))
        ax2.set_ylim((644.0407050947242, 1540.7599664240608))

        # plot_projection(tracks - ctracks, ax=ax1, color='grey', label='Deleted Tracks')  # Plot the deleted tracks
        plot_projected_detections(detections_aggregated, coord=0, ax=ax1)
        plot_projected_detections(detections_aggregated, coord=2, ax=ax2)
        plot_projected_tracks(tracks, ax=ax1, coord=0, show_error=False)
        plot_projected_gt(gt, alive, ax=ax1, coord=0)
        plot_projected_tracks(tracks, ax=ax2, coord=2, show_error=False)
        plot_projected_gt(gt, alive, ax=ax2, coord=2)

        plt.pause(0.5)

    print()
