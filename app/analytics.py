
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


# --------------------------------------------------
# LOAD ENERGY DATASETS
# --------------------------------------------------
def load_energy_data():

    energy = pd.read_csv(
        DATA_DIR / "fact_energy_indicators.csv"
    )

    generation = pd.read_csv(
        DATA_DIR / "fact_generation.csv"
    )

    areas = pd.read_csv(
        DATA_DIR / "dim_area.csv"
    )

    sources = pd.read_csv(
        DATA_DIR / "dim_source.csv"
    )

    years = pd.read_csv(
        DATA_DIR / "dim_year.csv"
    )

    return {
        "energy": energy,
        "generation": generation,
        "areas": areas,
        "sources": sources,
        "years": years
    }


# --------------------------------------------------
# DATA CLEANING
# --------------------------------------------------
def prepare_energy_data():

    data = load_energy_data()

    energy = data["energy"].copy()
    generation = data["generation"].copy()

    energy["Year"] = pd.to_numeric(
        energy["Year"],
        errors="coerce"
    )

    generation["Year"] = pd.to_numeric(
        generation["Year"],
        errors="coerce"
    )

    energy["Generation (TWh)"] = pd.to_numeric(
        energy["Generation (TWh)"],
        errors="coerce"
    )

    generation["Generation (TWh)"] = pd.to_numeric(
        generation["Generation (TWh)"],
        errors="coerce"
    )

    return {
        "energy": energy,
        "generation": generation,
        "areas": data["areas"],
        "sources": data["sources"],
        "years": data["years"]
    }


# --------------------------------------------------
# DATASET SUMMARY
# --------------------------------------------------
def get_dataset_summary():

    data = prepare_energy_data()

    energy = data["energy"]
    generation = data["generation"]

    return {
        "total_areas": int(energy["Area"].nunique()),
        "start_year": int(energy["Year"].min()),
        "end_year": int(energy["Year"].max()),
        "energy_records": int(len(energy)),
        "generation_records": int(len(generation)),
        "electricity_sources": sorted(
            generation["Electricity source"]
            .dropna()
            .unique()
            .tolist()
        )
    }


# --------------------------------------------------
# COUNTRY GENERATION ANALYSIS
# --------------------------------------------------
def get_country_generation(country, year=None):

    data = prepare_energy_data()
    generation = data["generation"]

    # Match the requested country
    country_data = generation[
        (generation["Area"].str.lower() == country.lower()) &
        (generation["Area type"] == "Country or economy")
    ].copy()

    if year is not None:
        country_data = country_data[
            country_data["Year"] == int(year)
        ]

    if country_data.empty:
        return {
            "error": f"No generation data found for {country}."
        }

    # Group by electricity source
    result = (
        country_data.groupby("Electricity source")[
            "Generation (TWh)"
        ]
        .sum()
        .sort_values(ascending=False)
    )

    return {
        "country": country,
        "year": year if year is not None else "All available years",
        "generation_by_source": result.round(3).to_dict(),
        "total_generation_twh": round(
            float(result.sum()), 3
        )
    }
