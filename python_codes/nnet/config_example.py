# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 08:08:34 2016

Purpose: Configuration File for Statistical Parametric Speech Synthesis

@author: Sivanand Achanta
"""


from configparser import ConfigParser

parser = ConfigParser()
parser.read('config_spss.ini')


# paths
matpath = parser['paths'].get('matpath')

# ints
din = parser['ints'].getint('din')
dout = parser['ints'].getint('dout')

# flags
mvniflag = parser['flags'].getboolean('mvniflag')
mvnoflag = parser['flags'].getboolean('mvnoflag')
maxminiflag = parser['flags'].getboolean('maxminiflag')
maxminoflag = parser['flags'].getboolean('maxminoflag')
verboseflag = parser['flags'].getboolean('verboseflag')
