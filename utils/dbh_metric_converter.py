import pandas as pd


def estimate_dbh(input_csv: str, output_csv: str) -> pd.DataFrame:
    """
    Reads a CSV file, computes estimated DBH (cm) for each row,
    and writes a new CSV with the `estimated_dbh` column appended.

    Formula:
        W (mm) = (n * S * D_mm) / (N * f)
        estimated_dbh (cm) = W (mm) / 10

    Where:
        n     = dbh_width     (pixels)
        S     = sensor_width  (mm)
        D_mm  = length * 10   (length is in cm, converted to mm)
        N     = image_width   (pixels)
        f     = focal_length  (mm)

    Parameters
    ----------
    input_csv  : path to the input CSV file
    output_csv : path to write the output CSV file

    Returns
    -------
    pd.DataFrame with estimated_dbh column included
    """
    df = pd.read_csv(input_csv)

    D_mm = df["length"] * 10  # cm → mm
    W_mm = (df["dbh_width"] * df["sensor_width"] * D_mm) / (df["image_width"] * df["focal_length"])

    df["estimated_dbh"] = (W_mm / 10).round(4)  # mm → cm

    df.to_csv(output_csv, index=False)
    return df