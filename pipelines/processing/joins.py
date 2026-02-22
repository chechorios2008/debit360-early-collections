def build_base_dataframe(
    clientes,
    moras,
    tanque_movimiento,
    canales,
    gestiones,
    excedente,
):

    df = clientes

    join_keys = ["num_doc", "obl17", "f_analisis"]

    df = df.join(moras, on=join_keys, how="left")
    df = df.join(tanque_movimiento, on=join_keys, how="left")
    df = df.join(canales, on=join_keys, how="left")
    df = df.join(gestiones, on=join_keys, how="left")
    df = df.join(excedente, on=join_keys, how="left")

    return df    