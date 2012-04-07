#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pyrecsys.datasets import load_pci_sample

data = load_pci_sample()

if data.to_iid(0, 'Mick LaSalle') == 5:
    print "Ok"
else:
    print "No"

try:
    x = data.to_iid(0, 'Dr. X')
except ValueError, detail:
    print "Ok", detail
else:
    print "No", detail

if data.to_eid(1, 4) == 'The Night Listener':
    print "Ok"
else:
    print "No"

try:
    x = data.to_eid(1, 100)
except ValueError, detail:
    print "Ok", detail
else:
    print "No", detail
