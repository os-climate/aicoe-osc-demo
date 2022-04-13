def get_kpi_mapping_category(df):
    """
    Get kpi mapping and category from kpi df
    """
    _KPI_MAPPING = {str(i[0]): i[1] for i in df[["kpi_id", "question"]].values}
    KPI_MAPPING = {(float(key)): value for key, value in _KPI_MAPPING.items()}
    
    # Category where the answer to the question should originate from
    KPI_CATEGORY = {
    i[0]: [j.strip() for j in i[1].split(", ")]
    for i in df[["kpi_id", "kpi_category"]].values
    }
    
    _KPI_MAPPING_MODEL = {
    str(i[0]): (i[1], [j.strip() for j in i[2].split(",")])
    for i in df[["kpi_id", "question", "sectors"]].values
    }
    KPI_MAPPING_MODEL = {(float(key)): value for key, value in _KPI_MAPPING_MODEL.items()}
    
    return {"KPI_MAPPING": KPI_MAPPING,
            "KPI_CATEGORY": KPI_CATEGORY,
            "KPI_MAPPING_MODEL": KPI_MAPPING_MODEL}