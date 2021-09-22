"""
Fangrui Liu @ UBCO ISDPRL

Copyright Reserved 2020
NO DISTRIBUTION AGREEMENT PROVIDED
"""


REGISTRY = {
    'DATALOADER': {},
    'TRAINER': {},
    'MODEL': {},
    'BACKBONE': {},
}

def REGISTER_DATALOADER(name):
    def decorator(f):
        REGISTRY['DATALOADER'][name] = f
        return f
    return decorator

def REGISTER_MODEL(name):
    def decorator(f):
        REGISTRY['MODEL'][name] = f
        return f
    return decorator

def REGISTER_TRAINER(name):
    def decorator(f):
        REGISTRY['TRAINER'][name] = f
        return f
    return decorator

def REGISTER_BACKBONE(name):
    def decorator(f):
        REGISTRY['BACKBONE'][name] = f
        return f
    return decorator

def print_loaded(reg, prefix=''):
    for n in reg.keys():
        print(prefix + n)
        if type(reg[n]) is dict:
            print_loaded(reg[n], prefix=prefix+'\t')