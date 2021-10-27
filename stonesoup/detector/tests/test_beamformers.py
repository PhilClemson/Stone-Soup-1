# -*- coding: utf-8 -*-
import numpy as np
from os import path
from datetime import datetime

import pytest

from stonesoup.detector import beamformers
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                               ConstantVelocity
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.array import CovarianceMatrix
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.types.state import GaussianState
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track


@pytest.fixture(params=[beamformers.CaponBeamformer, beamformers.RJMCMCBeamformer])
def detector(request):
    data_file = path.join(path.dirname(__file__), "fixed_target_example.csv")
    extra_args = {'seed': 12345} if request.param is beamformers.RJMCMCBeamformer else {}
    return request.param(csv_path=data_file, sensor_loc=[[0, 0, 0], [0, 10, 0], [0, 20, 0], [10, 0, 0],
                                                [10, 10, 0], [10, 20, 0], [20, 0, 0], [20, 10, 0],
                                                [20, 20, 0]], fs=2000, omega=50, wave_speed=1481,
                                                **extra_args)


def test_fixed_target_beamformer(detector):
    truth = [0.8, 0.2]

    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.01),
                                                              ConstantVelocity(0.01)])

    covar = CovarianceMatrix(np.array([[1, 0], [0, 1]]))
    measurement_model = LinearGaussian(ndim_state=4, mapping=[0, 2], noise_covar=covar)

    predictor = KalmanPredictor(transition_model)

    updater = KalmanUpdater(measurement_model)

    prior = GaussianState([[0.5], [0], [0.5], [0]], np.diag([1, 0, 1, 0]),
                          timestamp=datetime.now())
    
    # get number of samples in data file
    L = 0
    for row in open(detector.csv_path):
        L += 1

    window = 800  # size of sliding window in samples        
    winstarts = np.linspace(0, L-window, num=int(L/window), dtype=int)

    track = Track()
    
    for t in winstarts:
        detector.update(start_index=t,end_index=t+window-1)
        detector.detections_gen
        for timestep, detections in detector:
            for detection in detections:
                # check beamformer accuracy
                assert(np.abs(detection.state_vector[0] - truth[0]) < 0.4)
                assert(np.abs(detection.state_vector[1] - truth[1]) < 0.1)
                prediction = predictor.predict(prior, timestamp=detection.timestamp)
                hypothesis = SingleHypothesis(prediction, detection)  # Group prediction + measurement
                post = updater.update(hypothesis)
                track.append(post)
                prior = track[-1]
