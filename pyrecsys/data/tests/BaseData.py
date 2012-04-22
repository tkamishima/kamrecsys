#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from pyrecsys.datasets import load_pci_sample

data = load_pci_sample()

if data.to_iid(0, 'Mick LaSalle') == 5:
    print("Ok")
else:
    print("No")

try:
    x = data.to_iid(0, 'Dr. X')
except ValueError as detail:
    print("Ok", detail)
else:
    print("No")

if data.to_eid(1, 4) == 'The Night Listener':
    print("Ok")
else:
    print("No")

try:
    x = data.to_eid(1, 100)
except ValueError as detail:
    print("Ok", detail)
else:
    print("No")
