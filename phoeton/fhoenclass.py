"""
Classes and other stuff
"""

import numpy as np
import pandas as pd


class FhoenClassification:
    """ The main FhoenClassification Class

    """

    def __init__(self,
                 y,
                 data,
                 omega=None,
                 ddsector=None,
                 maxiter=10):
        """ Initialize parmeters which both simple and advanced model use

        Parameters:
        ------------
        y:       String
            Predictor, must be present in >>data<<
        data:   pandas.DataFrame
            Must at least contain 'y'
        omega:   String
            Concomitant, must be present in >>data<<
        ddsector: Array like
            Must contain two values to filter for wind direction
        maxiter:    Integer
            Number of iteration loops
        """

        Pandas DataFrame
        self.data = data


class SimpleFhoenClassification(FhoenClassification):

    # Fit the logistic model
    def fit(self):
        self.fhoen_probability = 1
