import logging
import pandas as pd
from pathlib import Path
from itertools import combinations
from argparse import ArgumentParser

from diffuse_neutrino_flux import Spectrum

logger = logging.getLogger("diffuse_neutrino_flux.data.bpl_2dscan_to_4d")


def process_2d_scan_contour(
    contour_file: str | Path,
    spectrum_name: str,
    outfile_path: str | Path,
    delimiter: str = ",",
    decimal: str = ".",
) -> pd.DataFrame:
    """
    Process a 2D scan contour file and convert it to a 4D format. The 2D file must be a CSV file with a
    multicolumn header containing the 2D scans in columns labeled <param1>-<param2>, <param1>-<param3>, etc.

    Parameters
    ----------
    contour_file : str or Path
        Path to the contour file.
    spectrum_name : str
        Name of the spectrum.
    outfile_path : str or Path
        Path to save the processed DataFrame as a CSV file.
    delimiter : str, optional
        Delimiter used in the CSV file, by default ","
    decimal : str, optional
        Decimal character used in the CSV file, by default "."

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the processed contour data in 4D format.
    """

    contour_path = Path(contour_file).expanduser()
    assert contour_path.exists(), f"Contour file {contour_path} does not exist"

    # Read the contour data
    logger.info(f"Reading contour data from {contour_path}")
    df = pd.read_csv(contour_path, header=[0, 1], delimiter=delimiter, decimal=decimal)
    logger.debug(f"Columns in contour file: {df.columns}")

    # Forward-fill missing top-level labels
    new_columns = []
    current_top = None
    for top, sub in df.columns:
        if not top.startswith("Unnamed"):
            current_top = top
        new_columns.append((current_top, sub))
    df.columns = pd.MultiIndex.from_tuples(new_columns)

    # load the spectrum
    measurements = Spectrum.load_summary_file()
    if spectrum_name not in measurements:
        raise ValueError(f"Spectrum name '{spectrum_name}' not found in measurements.")
    s = Spectrum.from_dict(measurements[spectrum_name])

    # Check if the required columns are present
    required_columns = [
        [f"{p1}-{p2}", f"{p2}-{p1}"] for p1, p2 in combinations(s.paramater_names, 2)
    ]
    for cols in required_columns:
        if all([col not in df.columns for col in cols]):
            raise ValueError(
                f"None of required column '{cols}' not found in the contour file."
            )

    param_names = s.paramater_names
    best_fit = s.best_fit

    # Collect all slices
    logger.info("Processing 2D scans into 4D format")
    all_slices = []
    for p1, p2 in combinations(param_names, 2):
        scan_keys = [f"{p1}-{p2}", f"{p2}-{p1}"]
        scan_key_exists = [
            scan_key in df.columns.get_level_values(0) for scan_key in scan_keys
        ]
        if not any(scan_key_exists):
            raise ValueError(
                f"Required columns '{scan_keys}' not found in the contour file."
            )
        if all(scan_key_exists):
            logger.warning(f"Both columns '{scan_keys}' found. Using '{scan_keys[0]}'.")
        scan_key = scan_keys[0] if scan_key_exists[0] else scan_keys[1]

        # Extract the 2D scan for this pair
        p1, p2 = scan_key.split("-")
        scan_df = df[scan_key].copy()
        scan_df = scan_df.rename(columns={"X": p1, "Y": p2})
        scan_df = scan_df.dropna(subset=[p1, p2])

        # Fill in best-fit values for the other parameters
        for p_other in set(param_names) - {p1, p2}:
            scan_df[p_other] = best_fit[p_other]

        # Keep track of which scan this row came from
        scan_df["scan"] = scan_key

        all_slices.append(scan_df)

    # Concatenate into one big DataFrame
    full_df = pd.concat(all_slices, ignore_index=True)
    logger.info(
        f"Processed DataFrame shape: {full_df.shape}, columns: {full_df.columns.tolist()}"
    )
    logger.info(f"Saving processed contour data to {outfile_path}")
    outfile_path = Path(outfile_path).expanduser()
    full_df.to_csv(outfile_path, index=False)
    logger.info("Done.")


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Process a 2D scan contour file and convert it to a 4D format."
    )
    parser.add_argument(
        "contour_file", type=str, help="Path to the contour file (CSV format)."
    )
    parser.add_argument(
        "spectrum_name",
        type=str,
        help="Name of the spectrum (as in measurements.json).",
    )
    parser.add_argument(
        "outfile_path", type=str, help="Path to save the processed CSV file."
    )
    parser.add_argument(
        "--delimiter",
        type=str,
        default=",",
        help="Delimiter used in the CSV file (default: ',').",
    )
    parser.add_argument(
        "--decimal",
        type=str,
        default=".",
        help="Decimal character used in the CSV file (default: '.').",
    )
    parser.add_argument(
        "--loglevel",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: 'INFO').",
    )

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=args.loglevel,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    process_2d_scan_contour(
        contour_file=args.contour_file,
        spectrum_name=args.spectrum_name,
        outfile_path=args.outfile_path,
        delimiter=args.delimiter,
        decimal=args.decimal,
    )
