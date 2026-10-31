"""HEC-RAS geometry (``.g##.hdf``) and plan (``.p##.hdf``) HDF files straight to archive
GeoParquet, through the HEC-RAS engine's Arrow layers (ras-commander-fast-hdf) and terrio's
archive GeoParquet writer, with no GeoDataFrame in between::

    import pyarrow as pa
    import terrio.hecras

    found = terrio.hecras.archive_layers("Model.p01.hdf", geometry=False)
    for layer in found["layers"]:
        print(layer["name"], layer["rows"], pa.table(layer["table"]).schema)
    print(found["skipped"])  # what the engine refused, to fill another way

    report = terrio.hecras.write_archive("Model.g01.hdf", "Model.g01.parquet",
                                         extra_layers=[("storage_areas_text", table)])

Available when terrio is built with the ``hecras`` feature (the default).
"""

from terrio._terrio import hecras_archive_layers as archive_layers
from terrio._terrio import write_hecras_archive as write_archive

__all__ = ["archive_layers", "write_archive"]
