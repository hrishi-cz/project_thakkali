"""
Sampling Bridge Layer: Materialize lazy data for Integrator.
"""

def materialize_sample(lazy_data, n=500):
    """
    Convert lazy data → small in-memory sample for Integrator.
    Handles Polars LazyFrame, Dask DataFrame, PyTorch Dataset, or pandas DataFrame.
    """
    try:
        # Polars LazyFrame
        if hasattr(lazy_data, "collect"):
            return lazy_data.head(n).collect().to_pandas()

        # Dask DataFrame
        if hasattr(lazy_data, "compute"):
            return lazy_data.head(n).compute()

        # PyTorch Dataset (images)
        if hasattr(lazy_data, "__getitem__") and hasattr(lazy_data, "__len__"):
            return [lazy_data[i] for i in range(min(n, len(lazy_data)))]

        # Already in-memory (pandas, list, etc)
        return lazy_data

    except Exception as e:
        print("Sampling failed:", e)
        return None
