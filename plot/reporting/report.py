#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 16:11:12 2022

@author: svrizzi
"""

#adapted from https://towardsdatascience.com/how-to-create-pdf-reports-with-python-the-essential-guide-c08dd3ebf2ee

simulation = 0

from hood_report import construct
plots_per_page = construct()

from hood_report import PDF
pdf = PDF()

DIR_REPORTS = 'Reports'

def WriteReport(simulation):
    
    fname = 'Report {simulation}'
    
    [pdf.print_page(elem) for elem in plots_per_page]    
        
    pdf.output(f'{DIR_REPORTS}/{fname}', 'F')