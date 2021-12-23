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
    if type(measurement_model) ==  CartesianToElevationBearing:
        #Measurement model is not invertible (range info is missing).
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
        ax.plot(data[:, 0], data[:, 2], 'bx', label=label)


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


    """Ground truth: model setup"""
    # Set a constant velocity transition model for the targets
    transition_model = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(0.5), ConstantVelocity(0.5), ConstantVelocity(0.1)], seed=seed)

    # Define the Gaussian State from which new targets are sampled on initialisation
    x_0, y_0, z_0 = 2000, 2000, 0
    sigma_x, sigma_y, sigma_z = 10, 10, 1
    sigma_x_dot, sigma_y_dot, sigma_z_dot = 5, 5, 1

    initial_state_mean = StateVector([[x_0], [0], [y_0], [0], [z_0], [0]])
    initial_state_cov = CovarianceMatrix(np.diag([sigma_x**2, sigma_x_dot**2,
                                                  sigma_y**2, sigma_y_dot**2,
                                                  sigma_z**2, sigma_z_dot**2]))
    timestamp_initial = datetime.now()
    initial_target_state = GaussianState(
        initial_state_mean,
        initial_state_cov,
        timestamp_initial
    )

    # Set the ground thruth simulator
    groundtruth_sim = MultiTargetGroundTruthSimulator(
        transition_model=transition_model,  # target transition model
        initial_state=initial_target_state,  # add our initial state for targets
        timestep=timedelta(seconds=1),  # time between measurements
        number_steps=60,  # 1 minute
        birth_rate=0.10,  # 10% chance of a new target being born
        death_probability=0.01, # 1% chance of a target being killed
        seed=seed)


    """Measurements: model setup"""
    # Setting up a comprehensive measurement model (here: CartesianToElevationBearingRangeRate)
    noise_covar = CovarianceMatrix(np.array(np.diag([
        np.deg2rad(1)**2,  # elevation_var
        np.deg2rad(0.15)**2,  # bearing_var
        25**2,  # range_var
        1**2  # rangerate_var
        ])))
    rotation_offset = StateVector([
        [np.deg2rad(0)],
        [np.deg2rad(0)],
        [np.deg2rad(0)]
    ]) # orientation of the array's boresight (oriented along x axis already)
    translation_offset = StateVector([
        [0],
        [40],
        [0]
    ]) # centre shifted in y direction
    model_args_generation = {
        'ndim_state': 6,
        'mapping': np.array([0, 2, 4]),
        'velocity_mapping': np.array([1, 3, 5]),
        'noise_covar': noise_covar,
        'rotation_offset': rotation_offset,
        'translation_offset': translation_offset}  # centre of the array

    measurement_model_generation = CartesianToElevationBearingRangeRate(**model_args_generation)
    detection_probability=0.9
    clutter_area = np.array([
        [np.deg2rad(-10), np.deg2rad(10)],  # in elevation
             [np.deg2rad(0), np.deg2rad(90)],  # in bearing
             [0, 4000],  # in range
             [-1, 1] ]) # in velocity
    clutter_rate = 2
    groundtruth_sim_det = copy.deepcopy(groundtruth_sim)
    detector_sim = SimpleDetectionSimulator(
        groundtruth=groundtruth_sim_det,
        measurement_model=measurement_model_generation,
        detection_probability=detection_probability,  # probability of detection
        meas_range=clutter_area,  # clutter area
        clutter_rate=clutter_rate,
        seed=seed)


    """Export ground truth (+ideal measurement) data into a .mat file"""

    groundtruth_sim_export = copy.deepcopy(groundtruth_sim)  # make a copy to ensure the seed is the same
    timestamp_list, truth_list, ideal_measurements_list = [], [], []  # timestamps will be saved as floats
    for time, ctracks in groundtruth_sim_export:
        # add current timestamp
        timestamp_list.append(time.timestamp())
        
        # add current (ground truth) states
        current_states = [track.state_vector for track in ctracks]
        truth_list.append(current_states)

        # add current ideal measurements (of current_states)
        current_ideal_measurements = []
        for track in ctracks:
            # elevation(theta),bearing(phi),range,rangerate
            ideal_measurement = measurement_model_generation.function(track, noise=False)
            cleaned_measurement = [float(i) for i in ideal_measurement]  # Removes Bearing() and Elevation()
            current_ideal_measurements.append(cleaned_measurement)
        ideal_measurements_list.append(current_ideal_measurements)

    savemat("ground_truth.mat", {'timestamps': timestamp_list,
                                 'truth': np.array(truth_list, dtype=object),
                                 'ideal_measurements': np.array(ideal_measurements_list, dtype=object)})


    """Export measurement data into a .csv file"""

    path_detections = 'detections.csv'
    with open(path_detections, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        time_field = 'time'
        state_vector_fields_generation = ['elevation', 'bearing', 'range', 'rangerate']
        header = [time_field] + state_vector_fields_generation
        writer.writerow(header)  # write the header

        detector_sim_export = copy.deepcopy(detector_sim)
        for time, detections in detector_sim_export:
            for _, val in enumerate(detections):
                det_data = {
                    'time': val.timestamp.timestamp(),
                    'elevation': val.state_vector[0],
                    'bearing': val.state_vector[1],
                    'range': val.state_vector[2],
                    'rangerate': val.state_vector[3]}
                writer.writerow([det_data[key] for key in det_data])  # write the data
    f.close()


    # """Intermediate visualization"""
    # 
    # truths = set()
    # groundtruth_sim_viz = copy.deepcopy(groundtruth_sim)
    # for time, ctracks in groundtruth_sim_viz:
    #     truths.update(ctracks)
    # 
    # all_measurements = []
    # detector_sim_viz = copy.deepcopy(detector_sim)
    # for time, detections in detector_sim_viz:
    #     for detection in detections:
    #         all_measurements.append(detection)
    # 
    # plotter = Plotter()
    # plotter.plot_measurements(all_measurements, [0, 2], color='g')
    # plotter.plot_ground_truths(truths, [0, 2])
    # plt.plot([0, 0], [0, 80], '-r')  # visualizing array extension/orientation
    # ax = plt.gca()
    # ax.set_xlim((-10, 3500))
    # ax.set_ylim((-10, 2500))
    # plt.show()


    """Setting up a tracking algorithm"""
    # Announce candidate measurement models
    model_class_dict = {
        0: CartesianToElevationBearingRangeRate, # measures range as well as rangerate
        1: CartesianToElevationBearingRange,
        2: CartesianToElevationBearing}

    # Ask for the number of a specific measurement model
    model_num = None
    while model_num not in {0, 1, 2}:
        model_num = int(input(
            "Select the measurement model class: \n 0 - for {}, \n 1 - for {}, \n 2 - for {}.\n".format(
                model_class_dict[0], model_class_dict[1], model_class_dict[2])))

    # Assemble the measurement model from its class and (a part of) arguments used in generation
    model_args = copy.deepcopy(model_args_generation)
    if model_num in {1, 2}:
        del model_args['velocity_mapping']  # 'velocity_mapping' only required for model_num=0, and dropped elsewhere
        ndim_meas = np.shape(model_args['noise_covar'])[0] - model_num  # max. dimension to preserve
        model_args['noise_covar'] = model_args['noise_covar'][0:ndim_meas, 0:ndim_meas] # trim covariance matrix
    measurement_model = model_class_dict[model_num](**model_args)

    # Import measurement data from a .csv file
    state_vector_fields = state_vector_fields_generation[:-model_num or None]
    detector = CSVDetectionReader(
        path=path_detections,
        state_vector_fields=state_vector_fields,
        time_field=time_field,
        timestamp=True
    )

    # Choose filtering solution (UKF vs PF)
    use_ukf = None
    while use_ukf not in {True, False}:
        use_ukf = bool(input("Use unscented Kalman filtering (and not particle filtering)? (0 - no, 1 - yes)\n"))
        if model_num == 2 and use_ukf == True:
            use_ukf = False
            print('Kalman filtering not possible with this measurement model. \nUsing particle filtering instead.')

    # Predictor/Updater
    transition_model = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(0.5), ConstantVelocity(0.5), ConstantVelocity(0.1)])
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
    groundtruth = set()
    groundtruth_gen = groundtruth_sim.groundtruth_paths_gen()
    for time, ctracks in tracker:

        # Getting access to ground truth
        gt_time, ctruths = None, None
        while gt_time != time:
            gt_time, ctruths = next(groundtruth_gen)
            groundtruth.update(ctruths)

        tracks.update(ctracks)
        detections = detector.current[1]

        # Plot
        plt.clf()
        plot_tracks(groundtruth, show_error=False, color='black', label='Ground Truth')
        plot_tracks(tracks - ctracks, color='grey', label='Deleted Tracks')  # Plot the deleted tracks
        plot_tracks(ctracks)  # Plot the current tracks
        plot_detections(detections)  # Plot the detections

        ax = plt.gca()
        ax.set_xlim((-10, 3500))
        ax.set_ylim((-10, 2500))
        ax.set_title(time - timestamp_initial)
        plt.plot([0, 0], [0, 80], '-g', label='Sonar Array')  # visualizing array extension/orientation

        # Introduce plot legend with no repreated entries
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        plt.pause(0.5)

    print()
