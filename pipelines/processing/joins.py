from pyspark.sql import functions as F


def build_base_dataframe(clientes, moras, tanque_movimiento, canales, gestiones, excedente):
    df = clientes
    join_keys = ["num_doc", "obl17", "f_analisis"]

    # Lista de dataframes a unir
    dfs_to_join = [moras, tanque_movimiento, canales, gestiones, excedente]

    for incoming_df in dfs_to_join:
        df = df.join(incoming_df, on=join_keys, how="left")

    # SENIOR TIP: Llenar nulos post-join para que el Feature Engineering no propague Nulls
    # Solo llenamos con 0 las columnas que no son llaves
    fill_cols = [c for c in df.columns if c not in join_keys]
    df = df.fillna(0, subset=fill_cols)

    return df