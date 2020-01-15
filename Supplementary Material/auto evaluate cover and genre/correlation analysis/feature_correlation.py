#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas
import matplotlib.pyplot as plt


import pandas as pd
import numpy as np

rs = np.random.RandomState(0)
data = pd.read_csv("Classical_Sun_Cult-The_Ritual.csv")
corr = data.corr()
corr.style.background_gradient(cmap='coolwarm')
