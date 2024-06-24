from copy import deepcopy
from dotmap import DotMap


def create_SAC_algorithm(variant, *args, **kwargs):
    from .sac import SAC

    algorithm = SAC(*args, **kwargs)

    return algorithm


def create_SQL_algorithm(variant, *args, **kwargs):
    from .sql import SQL

    algorithm = SQL(*args, **kwargs)

    return algorithm

def create_MVE_algorithm(variant, *args, **kwargs):
    from .mve_sac import MVESAC

    algorithm = MVESAC(*args, **kwargs)

    return algorithm

def create_MOPO_algorithm(variant, *args, **kwargs):
    from mopo.algorithms.mopo import MOPO

    print('create_MOPO_algorithm args', args)
    print('create_MOPO_algorithm kwargs', kwargs)
    algorithm = MOPO(*args, **kwargs)

    return algorithm


ALGORITHM_CLASSES = {
    'SAC': create_SAC_algorithm,
    'SQL': create_SQL_algorithm,
    'MOPO': create_MOPO_algorithm,
}


def get_algorithm_from_variant(variant,
                               *args,
                               **kwargs):
    algorithm_params = variant['algorithm_params']
    algorithm_type = algorithm_params['type']
    algorithm_kwargs = deepcopy(algorithm_params['kwargs'])
    
    # TODO Alan: The call to toDict() is only needed for local running
    # Clearly slightly different lib versions are being used
    if isinstance(algorithm_kwargs, DotMap):
        algorithm_kwargs = algorithm_kwargs.toDict()

    algorithm = ALGORITHM_CLASSES[algorithm_type](
        variant, *args, **algorithm_kwargs, **kwargs)

    return algorithm
