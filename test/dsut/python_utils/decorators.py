"""

Some useful decorators. Other useful decorators can be found in the module functools e.g. lru_cache.

"""
#################################### libraries and modules ####################################
# region

import typing
import time
import functools
import copy

# endregion
#################################### decorator functions ####################################
# region

def no_overwrite(global_dict:dict = None, modules: typing.List[str] = None) -> typing.Callable:
    """
    Function decorator that can be used to NOT overwrite functions if they already exist.
    Main use case is not overwriting imported functions from another file (useful for configuration vs main).

    :param global_dict: dictionary to be passed from main see ValueError below
    :param modules: list of namespaces to check
    :return:
    """
    if modules is None:
        modules = ['__main__']

    if global_dict is None:
        raise ValueError(f"You have to pass 'global_dict = dict(globals())")

    print("! Warning -> make sure to pass 'global_dict = dict(globals())' to the no_overwrite decorator.")

    def outer_wrapper(func):
        """
        """
        # get the function name to check
        f_name = func.__name__

        # get defined functions in set of modules
        l = []
        for key, value in global_dict.items():
            if callable(value) and value.__module__ in modules:
                l.append(key)

        # if name already in searched modules 'namespace'
        # then don't overwrite and get copy of current function (important to do this at def time and not run time)
        # runtime = recursive loop
        if f_name in l:
            print(f"function '{f_name}' will not be overwritten")

            # get copy of function (prevent recursive loop)
            # todo: test if copy was necessary or just getting copy at definition time rather than run time
            value = copy.deepcopy(global_dict[f_name])

            @functools.wraps(func)
            def wrapper(*args,**kwargs):
                return value(*args, **kwargs)

        else:
            @functools.wraps(func)
            def wrapper(*args,**kwargs):
                return func(*args, **kwargs)

        return wrapper

    return outer_wrapper

def timer(time_unit: str = None) -> typing.Callable:
    """
    A decorator to print the timing of functions. Useful for outside of Jupyter magic %timeit
    :param time_unit: 'seconds', 'minutes', or 'hours'
    :return:
    """
    if time_unit is None:
        time_unit = 'seconds'
    def wrapper(func):
        @functools.wraps(func)
        def wrapper_timer(*args, **kwargs):
            start_time = time.perf_counter()
            value = func(*args, **kwargs)
            end_time = time.perf_counter()
            run_time = end_time - start_time

            if time_unit == 'hours':
                print(f"Finished '{func.__name__}' in {run_time / 3600:.4f} hrs")
            elif time_unit == 'minutes':
                print(f"Finished '{func.__name__}' in {run_time / 60:.4f} mins")
            elif time_unit == 'seconds':
                print(f"Finished '{func.__name__}' in {run_time :.4f} secs")
            return value
        return wrapper_timer
    return wrapper

def tryexcept(func) -> typing.Callable:
    """
    Decorator to wrap funciton in try-except. Useful to keep program going
    :param func:
    :return:
    """
    @functools.wraps(func)
    def wrapper_tryexcept(*args, **kwargs):
        try:
            value = func(*args, **kwargs)
            return value
        except:
            print(f'Error occured in {func.__name__!r}')
    return wrapper_tryexcept

def toggle_off(func) -> typing.Callable:
    """
    A decorator to toggle a function off. Useful for shutting off the functionality
    of a funciton in one place rather than finding all occurrences.
    :param func:
    :return:
    """
    @functools.wraps(func)
    def wrapper_func(*args, **kwargs):
        return None
    return wrapper_func

# endregion

# EOF

