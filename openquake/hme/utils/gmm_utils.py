from openquake.hazardlib.calc import ground_motion_fields


def get_gsims(gsim_lt, trts):
    # use gsim_lt.values, which is a defaultict
    pass


def gmf_from_rupture(
    rupture,
    sites=None,
    imts=None,
    gsim=None,
    truncation_level=None,
    realizations=None,
    correlation_model=None,
    seed=0,
):

    # should probably do something here

    return ground_motion_fields(
        rupture,
        sites=sites,
        imts=imts,
        gsim=gsim,
        truncation_level=truncation_level,
        realizations=realizations,
        correlation_model=correlation_model,
        seed=seed,
    )
