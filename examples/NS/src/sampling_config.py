# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author:Marcel Tuan
@Blog: https://github.com/AlfredMarcel
 
@File: sampling_config.py
@Time: 2022/7/20 15:49
"""

from easydict import EasyDict as edict

src_sampling_config = edict({
    'domain': edict({
        'random_sampling': True,
        'size': 2500,
        'sampler': 'uniform'
    }),
    'IC': edict({
        'random_sampling': True,
        'size': 250,
        'sampler': 'uniform',
    }),
    'time': edict({
        'random_sampling': True,
        'size': 250,
        'sampler': 'uniform',
    }),
})


bc_sampling_config = edict({
    'BC': edict({
        'random_sampling': True,
        'size': 250,
        'sampler': 'uniform',
        'with_normal': False
    }),
    'time': edict({
        'random_sampling': True,
        'size': 250,
        'sampler': 'uniform',
    }),
})