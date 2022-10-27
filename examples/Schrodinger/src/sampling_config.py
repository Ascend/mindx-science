from easydict import EasyDict as edict


line_config = edict({
        'domain': edict({
            'size': 20000,
            'random_sampling': False,
        }),
        'BC': edict({
            'size': 100,
            'random_sampling': False,
        }),
        'IC': edict({
            'size': 100,
            'random_sampling': False,
        }),
        'time': edict({
            'size': 20000,
            'random_sampling': True,
        })
    })
