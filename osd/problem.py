# -*- coding: utf-8 -*-
''' Data Handler Module

This module contains a class for defining a signal demixing optimization problem


Author: Bennet Meyers
'''



class Problem():
    def __init__(self, data, components):
        self.data = data
        self.components = components

    def demix(self):
        pass

    def optimize_parameters(self):
        pass

    def holdout_validation(self):
        pass