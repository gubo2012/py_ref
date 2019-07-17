# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:54:55 2019

@author: dxuser22
"""

# Decorators
from functools import wraps


def my_logger(orig_func):
    import logging
    from datetime import datetime
    logging.basicConfig(filename='logs\\{}.log'.format(orig_func.__name__), level=logging.INFO)

    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        logging.info(
            '{}, ran with args: {}, and kwargs: {}'.format(datetime.now(), args, kwargs))
        return orig_func(*args, **kwargs)

    return wrapper


def my_timer(orig_func):
    import time

    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = orig_func(*args, **kwargs)
        t2 = time.time() - t1
        print('{} ran in: {} sec'.format(orig_func.__name__, t2))
        return result

    return wrapper