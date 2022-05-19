"""Utilities to get KPI mappings details."""


def get_kpi_mapping_category(df):
    """Get kpi mapping and category from kpi df."""
    _kpi_mapping = {str(i[0]): i[1] for i in df[["kpi_id", "question"]].values}
    kpi_mapping = {(float(key)): value for key, value in _kpi_mapping.items()}

    # Category where the answer to the question should originate from
    kpi_category = {
        i[0]: [j.strip() for j in i[1].split(", ")]
        for i in df[["kpi_id", "kpi_category"]].values
    }

    _kpi_mapping_model = {
        str(i[0]): (i[1], [j.strip() for j in i[2].split(",")])
        for i in df[["kpi_id", "question", "sectors"]].values
    }
    kpi_mapping_model = {
        (float(key)): value for key, value in _kpi_mapping_model.items()
    }

    return {
        "KPI_MAPPING": kpi_mapping,
        "KPI_CATEGORY": kpi_category,
        "KPI_MAPPING_MODEL": kpi_mapping_model,
    }
