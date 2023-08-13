def instance_vars_identical(instance_one, instance_two):
    return hash(frozenset(vars(instance_one).items())) == hash(frozenset(vars(instance_two).items()))