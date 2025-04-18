import numpy as np
import tensorflow as tf
import locale

locale.setlocale(locale.LC_ALL, '')  # Uncomment if needed

_params = {}
_param_aliases = {}

def param(name, *args, **kwargs):
    """
    A wrapper for `tf.Variable` which enables parameter sharing in models.

    Creates and returns TensorFlow variables similarly to `tf.Variable`, 
    except if you try to create a param with the same name as a 
    previously-created one, `param(...)` will just return the old one instead of 
    making a new one.

    This constructor also adds a `param` attribute to the shared variables it 
    creates, so that you can easily search a graph for all params.
    """
    if name not in _params:
        kwargs['name'] = name
        param_var = tf.Variable(*args, **kwargs)
        param_var.param = True
        _params[name] = param_var
    result = _params[name]
    i = 0
    while result in _param_aliases:
        # print(f'Following alias {i}: {result} to {_param_aliases[result]}')
        i += 1
        result = _param_aliases[result]
    return result

def params_with_name(name):
    return [p for n, p in _params.items() if name in n]

def delete_all_params():
    _params.clear()

def alias_params(replace_dict):
    for old, new in replace_dict.items():
        # print(f"Aliasing {old} to {new}")
        _param_aliases[old] = new

def delete_param_aliases():
    _param_aliases.clear()

def print_model_settings(locals_):
    print("Uppercase local vars:")
    all_vars = [(k, v) for (k, v) in locals_.items() if (k.isupper() and k not in {'T', 'SETTINGS', 'ALL_SETTINGS'})]
    all_vars = sorted(all_vars, key=lambda x: x[0])
    for var_name, var_value in all_vars:
        print("\t{}: {}".format(var_name, var_value))

def print_model_settings_dict(settings):
    print("Settings dict:")
    all_vars = sorted(settings.items(), key=lambda x: x[0])
    for var_name, var_value in all_vars:
        print("\t{}: {}".format(var_name, var_value))
