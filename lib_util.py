from typing import Dict


def with_grid_values(vars_and_values: Dict, f):
    var_names = list(vars_and_values.keys())
    n_vars = len(var_names)

    def rec(cur_var, cur_dict):
        if cur_var == n_vars:
            f(**cur_dict)
            return

        var_name = var_names[cur_var]
        for v in vars_and_values[var_name]:
            cur_dict[var_name] = v
            rec(cur_var + 1, cur_dict)

        del cur_dict[var_name]

    rec(0, {})
